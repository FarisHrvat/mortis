"""
MORTIS I/O Module
=================
Loading spatial metabolomics data from .h5ad.csv, and .xlsx files. Handles
single files, paired tissue/background exports, and whole folders (which get
paired up by filename).

Public API
----------
read_metabolomics_data(file_path)
    Load a single file into an AnnData object.

load_from_folder(folder_path)
    Scan a folder, auto-pair tissue/background files, and return a list of
    AnnData objects ready for preprocessing.

check_rois(adatas)
    Validate that every AnnData in a list has ROI labels assigned.

save_spatial_data(adatas, output_dir, prefix)
    Save processed AnnData objects to disk as .h5ad files.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List, Optional

import anndata as ad
import numpy as np
import pandas as pd

from .exceptions import FileFormatError, InvalidParameterError, MissingROIError, listing

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS = {
    ".h5ad",
    ".csv", ".tsv", ".txt", ".tab",      # delimiter is sniffed, not assumed
    ".xlsx", ".xlsm", ".xls",
    ".parquet", ".pq",
    ".rds",                               # needs pyreadr, see _read_rds
}

#: Delimiters worth trying when a text export does not say which it used.
#: Semicolon first after comma because that is what a European locale writes
#: when the decimal separator is a comma.
_DELIMITERS = (",", ";", "\t", "|")


def make_writable(adata: ad.AnnData) -> ad.AnnData:
    """
    Convert pandas extension string columns back to plain object dtype, in
    place, so the object can be written to ``.h5ad`` by any anndata version.

    Newer pandas hands back nullable ``StringArray`` for text columns, and
    anndata refuses to write those unless you opt in, on the grounds that
    versions below 0.11 cannot read them. The result is a file written on one
    machine that cannot be re-saved on another, which is exactly the situation
    a container or a cluster node puts you in.

    Plain object dtype is what every anndata version has always understood, so
    normalising to it costs nothing and removes the whole class of problem.
    """
    for frame in (adata.obs, adata.var):
        for column in frame.columns:
            if isinstance(frame[column].dtype, pd.StringDtype):
                frame[column] = frame[column].astype(object)
        if isinstance(frame.index.dtype, pd.StringDtype):
            frame.index = frame.index.astype(object)
    return adata


def _sniff_delimiter(file_path: Path) -> str:
    """Work out which delimiter a text export used.

    Instrument software disagrees: SCiLS writes tabs, a European Excel writes
    semicolons, METASPACE writes commas. csv.Sniffer is the first try, and
    when it cannot decide we pick whichever candidate splits the header and
    the first data row into the same number of fields, most fields winning.
    """
    with file_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(64 * 1024)
    if not sample.strip():
        raise FileFormatError(f"'{file_path.name}' is empty.")

    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(_DELIMITERS)).delimiter
    except csv.Error:
        pass

    lines = [ln for ln in sample.splitlines() if ln.strip()][:2]
    best, best_fields = None, 1
    for candidate in _DELIMITERS:
        counts = [len(next(csv.reader([ln], delimiter=candidate))) for ln in lines]
        if counts and len(set(counts)) == 1 and counts[0] > best_fields:
            best, best_fields = candidate, counts[0]
    if best is None:
        raise FileFormatError(
            f"Could not work out the delimiter in '{file_path.name}'. Tried "
            f"comma, semicolon, tab and pipe, and none of them split the "
            f"header and the first row into the same number of columns. If it "
            f"uses something else, read it with pandas and hand the frame to "
            f"mortis.from_dataframe()."
        )
    return best


def _read_text_table(file_path: Path) -> pd.DataFrame:
    """Read a delimited text export, sniffing the delimiter and the decimals."""
    sep = _sniff_delimiter(file_path)
    frame = pd.read_csv(file_path, sep=sep, encoding="utf-8", engine="python")
    # A European export writes "12,5" for twelve and a half, which pandas reads
    # as text. Left alone it would coerce to NaN and land in the matrix as a
    # zero, so look for it and re-read with the matching decimal mark.
    if _looks_like_comma_decimals(frame):
        frame = pd.read_csv(file_path, sep=sep, decimal=",",
                            encoding="utf-8", engine="python")
    return frame


_COMMA_DECIMAL = re.compile(r"^\s*-?\d{1,3}(?:\.\d{3})*,\d+\s*$")


def _looks_like_comma_decimals(frame: pd.DataFrame) -> bool:
    """True when a text column is really numbers written the European way."""
    for column in frame.columns:
        values = frame[column]
        # pandas 2 calls a text column 'object', pandas 3 calls it 'str'.
        if pd.api.types.is_numeric_dtype(values):
            continue
        sample = values.dropna().astype(str).head(20)
        if len(sample) and sample.map(lambda v: bool(_COMMA_DECIMAL.match(v))).all():
            return True
    return False


def _read_rds(file_path: Path) -> pd.DataFrame:
    """Read an R .rds holding a data frame, via pyreadr."""
    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError(
            "Reading .rds files needs pyreadr, which is not part of the "
            "default install:\n\n    pip install 'mortis-spatial[rds]'\n\n"
            "Alternatively, save the object from R as a csv or parquet, which "
            "MORTIS reads without an extra dependency."
        ) from exc
    try:
        result = pyreadr.read_r(str(file_path))
    except Exception as exc:
        raise FileFormatError(
            f"pyreadr could not read '{file_path.name}'. It handles a data "
            f"frame saved with saveRDS(); an S4 object, a Seurat object or a "
            f"list will not come through.\nOriginal error: {exc}"
        ) from exc
    frames = [v for v in result.values() if isinstance(v, pd.DataFrame)]
    if not frames:
        raise FileFormatError(
            f"'{file_path.name}' holds no data frame. saveRDS() of a matrix or "
            "a list gives something pyreadr cannot turn into a table; convert "
            "it to a data.frame in R first."
        )
    return frames[0]


#: Column names different exporters use for the pixel coordinates.
_X_ALIASES = ("x", "X", "x_pos", "xpos", "x_coord", "column", "col", "Column")
_Y_ALIASES = ("y", "Y", "y_pos", "ypos", "y_coord", "row", "Row")


def _normalise_coordinate_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whatever the exporter called the coordinates to 'x' and 'y'.

    SCiLS writes 'x'/'y', METASPACE writes 'x'/'y', but a plain image export
    often writes 'Row'/'Column' and an R pipeline tends to write 'X'/'Y'.
    Matching case-insensitively saves the user a rename they should not have
    had to think about.
    """
    if "x" in df.columns and "y" in df.columns:
        return df
    lowered = {str(c).lower(): c for c in df.columns}
    rename = {}
    for target, aliases in (("x", _X_ALIASES), ("y", _Y_ALIASES)):
        if target in df.columns:
            continue
        for alias in aliases:
            found = lowered.get(alias.lower())
            if found is not None and found not in rename:
                rename[found] = target
                break
    return df.rename(columns=rename) if rename else df


def from_dataframe(
    df: pd.DataFrame,
    x: str = "x",
    y: str = "y",
    sample: Optional[str] = None,
) -> ad.AnnData:
    """
    Build an AnnData from a table you have already read yourself.

    The escape hatch for a format MORTIS does not read: load it with whatever
    library does, hand the frame over here, and the rest of the package works
    as usual. Every column that is not a coordinate is treated as a compound.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per pixel. Needs the two coordinate columns; everything else
        is taken as intensities.
    x, y : str
        Names of the coordinate columns. Default ``'x'`` and ``'y'``.
    sample : str, optional
        Written to ``adata.obs['sample']`` so merged cohorts keep track of
        where a pixel came from.

    Returns
    -------
    anndata.AnnData
        With coordinates in ``.obsm['spatial']`` and compounds as
        ``.var_names``.

    Raises
    ------
    InvalidParameterError
        If a coordinate column is missing, or nothing is left to treat as a
        compound.

    Examples
    --------
    >>> frame = pd.read_stata("export.dta")
    >>> adata = mt.from_dataframe(frame, x="X", y="Y")
    """
    missing = [c for c in (x, y) if c not in df.columns]
    if missing:
        raise InvalidParameterError(
            f"Coordinate column(s) {missing} are not in this frame. Columns "
            f"present: {listing(df.columns)}. Pass the names you use, for "
            f"example from_dataframe(df, x='Row', y='Column')."
        )
    compounds = [c for c in df.columns if c not in (x, y)]
    if not compounds:
        raise InvalidParameterError(
            "This frame has the two coordinate columns and nothing else, so "
            "there are no intensities to analyse."
        )
    obs = pd.DataFrame(
        {"x": df[x].to_numpy(dtype=np.float32), "y": df[y].to_numpy(dtype=np.float32)},
        index=[f"{int(a)}_{int(b)}" for a, b in zip(df[x], df[y])],
    )
    if sample is not None:
        obs["sample"] = sample
    adata = ad.AnnData(
        X=df[compounds].to_numpy(dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=[str(c) for c in compounds]),
    )
    adata.obs_names_make_unique()
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    return adata


def _read_tabular(file_path: Path) -> ad.AnnData:
    """Parse a delimited text, Excel, parquet or RDS file into an AnnData."""
    suffix = file_path.suffix.lower()
    try:
        if suffix in {".csv", ".tsv", ".txt", ".tab"}:
            df = _read_text_table(file_path)
        elif suffix in {".parquet", ".pq"}:
            try:
                df = pd.read_parquet(file_path)
            except ImportError as exc:
                raise ImportError(
                    "Reading parquet needs pyarrow, which is not part of the "
                    "default install:\n\n    pip install "
                    "'mortis-spatial[fast-io]'"
                ) from exc
        elif suffix == ".rds":
            df = _read_rds(file_path)
        else:
            try:
                # python-calamine is a fast optional reader; fall back to
                # openpyxl (a hard dependency) if it isn't installed.
                df = pd.read_excel(file_path, engine="calamine")
            except ImportError:
                df = pd.read_excel(file_path, engine="openpyxl")
    except (FileFormatError, ImportError):
        raise
    except Exception as exc:
        raise FileFormatError(
            f"Could not read '{file_path.name}'. MORTIS accepts "
            f"{sorted(_SUPPORTED_EXTENSIONS)}, and the file has to be a table "
            f"with one row per pixel.\nOriginal error: {exc}"
        ) from exc

    df = _normalise_coordinate_names(df)

    missing = [c for c in ("x", "y") if c not in df.columns]
    if missing:
        raise FileFormatError(
            f"'{file_path.name}' is missing required column(s): {missing}. "
            "Every spatial metabolomics file must have 'x' and 'y' columns."
        )

    metabolite_cols = [c for c in df.columns if c not in ("x", "y")]
    if not metabolite_cols:
        raise FileFormatError(
            f"'{file_path.name}' has 'x' and 'y' columns but no metabolite "
            "columns. Please check your export settings."
        )

    obs_df = df[["x", "y"]].copy()

    # Build pixel IDs as "x_y". A comprehension beats df.apply here by a wide
    # margin on the 100k+ row exports these files usually are.
    x_vals = obs_df["x"].astype(int).values
    y_vals = obs_df["y"].astype(int).values
    obs_df.index = [f"{x}_{y}" for x, y in zip(x_vals, y_vals)]

    X = _intensities(df, metabolite_cols, file_path)
    var_df = pd.DataFrame(index=metabolite_cols)
    var_df.index.name = None

    adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
    adata.obsm["spatial"] = obs_df[["x", "y"]].to_numpy(dtype=np.float32)
    adata.uns["source_file"] = file_path.name
    return adata


def _intensities(df: pd.DataFrame, columns: List[str], file_path: Path) -> np.ndarray:
    """Turn the compound columns into a float32 matrix, loudly.

    Blanks are a genuine zero, so those are filled. Text that is not a number
    is not: coercing it to zero quietly would hand back a matrix full of
    absent compounds that were only ever a parsing mistake, and nothing
    downstream could tell the difference.
    """
    block = df[columns]
    coerced = block.apply(pd.to_numeric, errors="coerce")
    non_blank = block.apply(lambda col: col.astype(str).str.strip() != "")
    unparseable = coerced.isna() & block.notna() & non_blank
    if unparseable.to_numpy().any():
        per_column = unparseable.sum()
        worst = per_column[per_column > 0].sort_values(ascending=False)
        examples = []
        for name in list(worst.index)[:3]:
            bad = block.loc[unparseable[name], name]
            if len(bad):
                examples.append(f"{name!r} has {int(worst[name])}, such as {bad.iloc[0]!r}")
        raise FileFormatError(
            f"'{file_path.name}' has values in its compound columns that are "
            f"not numbers: {'; '.join(examples)}. MORTIS will not read those "
            f"as zero, because an absent compound and an unreadable one mean "
            f"different things. Common causes are a decimal comma that did "
            f"not survive the export, a thousands separator, or a text label "
            f"like 'n.d.' or 'below LOD' sitting in a numeric column."
        )
    return coerced.fillna(0).to_numpy(dtype=np.float32)


def _read_h5ad(file_path: Path) -> ad.AnnData:
    """Load an .h5ad file, ensuring spatial coordinates are present."""
    try:
        adata = ad.read_h5ad(file_path)
    except Exception as exc:
        raise FileFormatError(
            f"Could not read '{file_path.name}' as an H5AD file.\nOriginal error: {exc}"
        ) from exc

    if "spatial" not in adata.obsm:
        if "x" in adata.obs.columns and "y" in adata.obs.columns:
            adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy(dtype=np.float32)
        else:
            raise FileFormatError(
                f"'{file_path.name}' has no spatial coordinates. "
                "Expected either 'adata.obsm[\"spatial\"]' or 'x'/'y' columns "
                "in adata.obs."
            )

    # Files written by MORTIS <=0.5.0 can carry a '_perf_cache' in .uns holding
    # gigabytes of stale dense copies of .X (see analysis._get_X). Drop it on
    # read so it doesn't get dragged along into everything saved afterwards.
    if adata.uns.pop("_perf_cache", None) is not None:
        print(
            f"[MORTIS] Dropped a stale '_perf_cache' from '{file_path.name}' "
            "(written by an older MORTIS version; it held redundant copies of .X)."
        )

    if "source_file" not in adata.uns:
        adata.uns["source_file"] = file_path.name
    return adata


def _classify_name(stem: str) -> str:
    """Return 'tissue', 'background', or 'unknown' based on filename stem."""
    s = stem.lower()
    if re.search(r"(^|[_\-])tissue([_\-]|$)", s):
        return "tissue"
    if re.search(r"(^|[_\-])(background|bg)([_\-]|$)", s):
        return "background"
    return "unknown"


def _base_name(stem: str) -> str:
    """Strip tissue/background/bg tokens to get a pairing key."""
    return re.sub(r"[_\-]?(tissue|background|bg)[_\-]?", "", stem.lower()).strip("_-")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_metabolomics_data(file_path: str) -> ad.AnnData:
    """
    Load a single spatial metabolomics file into an AnnData object.

    Supported formats: ``.h5ad``, ``.csv``, ``.xlsx``.

    Parameters
    ----------
    file_path : str
        Path to the file to load.

    Returns
    -------
    anndata.AnnData
        AnnData with:

        * ``adata.X``: float32 intensity matrix (pixels x metabolites)
        * ``adata.obs``: pixel metadata including 'x' and 'y' coordinates
        * ``adata.var``: metabolite names as index
        * ``adata.obsm['spatial']``: (N, 2) float32 coordinate array

    Raises
    ------
    FileFormatError
        If the file extension is unsupported, required columns are missing,
        or the file cannot be parsed.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: '{file_path}'. Please check the path and try again."
        )
    ext = path.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise FileFormatError(
            f"Unsupported file extension '{ext}'. "
            f"MORTIS accepts: {sorted(_SUPPORTED_EXTENSIONS)}."
        )
    if ext == ".h5ad":
        return _read_h5ad(path)
    return _read_tabular(path)


def load_annotation_scores(
    adata: ad.AnnData,
    feature_table_path: str,
    compound_col: str = "Compound",
    score_col: str = "Identification Score",
    target_col: str = "score",
) -> ad.AnnData:
    """
    Merge per-compound annotation confidence scores from a separate feature
    table into ``adata.var[target_col]``.

    Many MSI facilities export spatial intensities (one row per pixel, one
    column per metabolite, the ``*_tissue.xlsx`` / ``*_background.xlsx`` /
    ``.csv`` files handled by :func:`read_metabolomics_data`) and annotation
    metadata (one row per metabolite, with identification confidence score,
    adduct, chemical formula, HMDB ID, etc.) as two *separate* files. This
    function performs the join needed before calling :func:`filter_by_score`,
    which otherwise requires ``adata.var['score']`` to already be populated
    (as it is for ``.h5ad`` exports from METASPACE/SCiLS, but not for raw
    ``.xlsx``/``.csv`` tissue exports).

    Parameters
    ----------
    adata : anndata.AnnData
        Data loaded via :func:`read_metabolomics_data` / :func:`load_from_folder`.
    feature_table_path : str
        Path to a ``.xlsx`` or ``.csv`` feature table with a compound-name
        column and a score column.
    compound_col : str, optional
        Column in the feature table matching ``adata.var_names``.
        Default: ``'Compound'``.
    score_col : str, optional
        Column in the feature table containing the annotation confidence
        score. Default: ``'Identification Score'``.
    target_col : str, optional
        Column created in ``adata.var``. Default: ``'score'``.

    Returns
    -------
    anndata.AnnData
        AnnData with ``adata.var[target_col]`` populated. Metabolites with
        no match in the feature table get ``NaN``, which
        :func:`filter_by_score` treats as failing any positive threshold.

    Raises
    ------
    FileNotFoundError
        If ``feature_table_path`` does not exist.
    FileFormatError
        If the file cannot be parsed or is missing ``compound_col`` /
        ``score_col``.

    Examples
    --------
    >>> adata = mortis.read_metabolomics_data("sample_tissue.xlsx")
    >>> adata = mortis.load_annotation_scores(adata, "feature_table.xlsx")
    >>> adata = mortis.filter_by_score(adata, min_score=0.5)
    """
    path = Path(feature_table_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Feature table not found: '{feature_table_path}'. "
            "Please check the path and try again."
        )
    try:
        if path.suffix.lower() == ".csv":
            table = pd.read_csv(path)
        else:
            table = pd.read_excel(path)
    except Exception as exc:
        raise FileFormatError(
            f"Could not read '{path.name}' as a feature table.\n"
            f"Original error: {exc}"
        ) from exc

    missing = [c for c in (compound_col, score_col) if c not in table.columns]
    if missing:
        raise FileFormatError(
            f"'{path.name}' is missing required column(s): {missing}. "
            f"Available columns: {table.columns.tolist()}."
        )

    score_map = dict(zip(table[compound_col].astype(str), table[score_col]))
    scores = adata.var_names.to_series().map(score_map)
    n_matched = int(scores.notna().sum())
    adata.var[target_col] = scores.to_numpy(dtype=float)

    print(
        f"[MORTIS] Annotation scores merged from '{path.name}': "
        f"{n_matched} / {adata.n_vars} metabolites matched -> adata.var['{target_col}']"
    )
    if n_matched < adata.n_vars:
        print(
            f"[MORTIS] Warning: {adata.n_vars - n_matched} metabolite(s) had no match "
            "in the feature table and will have NaN scores (filter_by_score drops these)."
        )
    return adata


def load_from_folder(folder_path: str) -> List[ad.AnnData]:
    """
    Scan a folder for spatial metabolomics files, auto-pair tissue and
    background files by name, and return a list of AnnData objects.

    Pairing logic
    -------------
    Files whose names contain ``tissue`` are paired with files whose names
    contain ``background`` or ``bg`` that share the same base name.
    Paired files are merged into a single AnnData with ``is_tissue`` and
    ``is_background`` boolean columns in ``.obs``.

    Parameters
    ----------
    folder_path : str
        Path to the folder to scan.

    Returns
    -------
    list of anndata.AnnData
        One AnnData per sample (paired or unpaired).

    Raises
    ------
    FileNotFoundError
        If ``folder_path`` does not exist.
    FileFormatError
        If any file cannot be parsed.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: '{folder_path}'. Please check the path and try again."
        )

    all_files = sorted(
        f
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in _SUPPORTED_EXTENSIONS
    )
    if not all_files:
        print(
            f"[MORTIS] No supported files found in '{folder_path}'. "
            f"Accepted extensions: {sorted(_SUPPORTED_EXTENSIONS)}."
        )
        return []

    h5ads = [f for f in all_files if f.suffix.lower() == ".h5ad"]
    tabulars = [f for f in all_files if f.suffix.lower() in {".csv", ".xlsx"}]

    adatas: List[ad.AnnData] = []

    for f in h5ads:
        print(f"[MORTIS] Loaded H5AD: {f.name}")
        adatas.append(_read_h5ad(f))

    pairs: dict[str, dict[str, Path]] = {}
    unpaired: List[Path] = []

    for f in tabulars:
        role = _classify_name(f.stem)
        if role == "unknown":
            unpaired.append(f)
            continue
        base = _base_name(f.stem)
        pairs.setdefault(base, {})[role] = f

    for base, pair in pairs.items():
        if "tissue" in pair and "background" in pair:
            print(
                f"[MORTIS] Auto-Paired: {pair['tissue'].name} "
                f"+ {pair['background'].name}"
            )
            adata_t = _read_tabular(pair["tissue"])
            adata_t.obs["is_tissue"] = True
            adata_t.obs["is_background"] = False

            adata_b = _read_tabular(pair["background"])
            adata_b.obs["is_tissue"] = False
            adata_b.obs["is_background"] = True

            merged = ad.concat(
                [adata_t, adata_b],
                join="outer",
                fill_value=0.0,
            )
            merged.obsm["spatial"] = merged.obs[["x", "y"]].to_numpy(dtype=np.float32)
            merged.uns["source_file"] = pair["tissue"].name
            adatas.append(merged)
        else:
            for role, f in pair.items():
                unpaired.append(f)

    for f in unpaired:
        print(f"[MORTIS] Loaded unpaired file: {f.name}")
        adatas.append(_read_tabular(f))

    return adatas


def check_rois(adatas: List[ad.AnnData]) -> None:
    """
    Verify that every AnnData in the list has ROI labels assigned.

    ROI labels are the ``is_tissue`` and ``is_background`` boolean columns
    in ``adata.obs``.  They are set automatically when tissue/background
    files are paired by :func:`load_from_folder`, or manually via
    :func:`draw_ROIs`.

    Parameters
    ----------
    adatas : list of anndata.AnnData
        The samples to check.

    Raises
    ------
    MissingROIError
        If any sample is missing ROI labels, with a list of which samples
        are affected and instructions on how to fix it.
    """
    missing_indices = [
        i
        for i, a in enumerate(adatas)
        if "is_tissue" not in a.obs.columns or "is_background" not in a.obs.columns
    ]
    if missing_indices:
        raise MissingROIError(
            f"{len(missing_indices)} sample(s) are missing ROI labels "
            f"(indices: {missing_indices}). "
            "Fix: run MORTIS.draw_ROIs(adata) for each unpaired sample, "
            "or load paired tissue/background files via load_from_folder()."
        )


def save_spatial_data(
    adatas: List[ad.AnnData],
    output_dir: str = ".",
    prefix: str = "mortis_sample",
) -> List[Path]:
    """
    Save a list of processed AnnData objects to disk as ``.h5ad`` files.

    Parameters
    ----------
    adatas : list of anndata.AnnData
        The samples to save.
    output_dir : str, optional
        Directory where files will be written.  Created if it does not exist.
        Default: current working directory.
    prefix : str, optional
        Filename prefix.  Files are named ``{prefix}_1.h5ad``,
        ``{prefix}_2.h5ad``, etc.  Default: ``"mortis_sample"``.

    Returns
    -------
    list of pathlib.Path
        Paths of the saved files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for i, adata in enumerate(adatas, start=1):
        path = out / f"{prefix}_{i}.h5ad"
        adata.write_h5ad(path)
        print(f"[MORTIS] Saved: {path}")
        saved.append(path)
    return saved

