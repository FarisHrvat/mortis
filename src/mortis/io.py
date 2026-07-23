"""
MORTIS I/O Module
=================
Fast, robust loading of spatial metabolomics data from .h5ad, .csv, and .xlsx
files.  Handles single files, paired tissue/background files, and entire
folders with automatic pairing.

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

import re
from pathlib import Path
from typing import List

import anndata as ad
import numpy as np
import pandas as pd

from .exceptions import FileFormatError, MissingROIError

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS = {".h5ad", ".csv", ".xlsx"}


def _read_tabular(file_path: Path) -> ad.AnnData:
    """Parse a CSV or XLSX file into an AnnData object using high-performance engines."""
    try:
        if file_path.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(file_path, engine="pyarrow")
            except ImportError:
                df = pd.read_csv(file_path)
        else:
            try:
                # python-calamine is a fast optional reader; fall back to
                # openpyxl (a hard dependency) if it isn't installed.
                df = pd.read_excel(file_path, engine="calamine")
            except ImportError:
                df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as exc:
        raise FileFormatError(
            f"Could not read '{file_path.name}'. "
            f"Make sure the file is a valid CSV or Excel file.\n"
            f"Original error: {exc}"
        ) from exc

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

    # Ultra-fast index creation utilizing Python List Comprehensions
    x_vals = obs_df["x"].astype(int).values
    y_vals = obs_df["y"].astype(int).values
    obs_df.index = [f"{x}_{y}" for x, y in zip(x_vals, y_vals)]

    X = df[metabolite_cols].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(dtype=np.float32)
    var_df = pd.DataFrame(index=metabolite_cols)
    var_df.index.name = None

    adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
    adata.obsm["spatial"] = obs_df[["x", "y"]].to_numpy(dtype=np.float32)
    adata.uns["source_file"] = file_path.name
    return adata


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

        * ``adata.X``  — float32 intensity matrix (pixels × metabolites)
        * ``adata.obs`` — pixel metadata including 'x' and 'y' coordinates
        * ``adata.var`` — metabolite names as index
        * ``adata.obsm['spatial']`` — (N, 2) float32 coordinate array

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
    column per metabolite — the ``*_tissue.xlsx`` / ``*_background.xlsx`` /
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
        f"{n_matched} / {adata.n_vars} metabolites matched → adata.var['{target_col}']"
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
