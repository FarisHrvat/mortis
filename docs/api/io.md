# I/O

## `read_metabolomics_data`

```python
mt.read_metabolomics_data(file_path: str) -> anndata.AnnData
```

Load a single spatial metabolomics file (`.h5ad`, `.csv`, or `.xlsx`)
into an `AnnData` object.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file_path` | `str` | :material-check: | Path to the file |

**Returns:** `AnnData` with `adata.X` (float32 intensity matrix, pixels x
metabolites), `adata.obs` (pixel metadata incl. `x`/`y`), `adata.var`
(metabolite names as the index), `adata.obsm['spatial']` (an (N, 2)
float32 coordinate array).

**Raises:** `FileNotFoundError` if the path doesn't exist;
[`FileFormatError`](exceptions.md#fileformaterror) if the extension is
unsupported or required columns (`x`, `y`, at least one metabolite) are
missing.

```python
adata = mt.read_metabolomics_data("sample_tissue.xlsx")
adata = mt.read_metabolomics_data("sample.h5ad")
```

---

## `load_from_folder`

```python
mt.load_from_folder(folder_path: str) -> list[anndata.AnnData]
```

Scan a folder for spatial metabolomics files and auto-pair `*_tissue.*`
with `*_background.*`/`*_bg.*` files sharing a base name.

**Pairing logic:** `421ID_tissue.xlsx` pairs with `421ID_background.xlsx`
(or `421ID_bg.xlsx`). Paired files are merged into one `AnnData` with
boolean `is_tissue`/`is_background` columns added to `.obs`. Files that
don't match this pattern are loaded individually.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `folder_path` | `str` | :material-check: | Folder to scan |

**Returns:** `list[AnnData]`, one entry per sample (paired or unpaired).

**Raises:** `FileNotFoundError` if the folder doesn't exist;
[`FileFormatError`](exceptions.md#fileformaterror) if any file can't be parsed.

```python
adatas = mt.load_from_folder("./data")
# [MORTIS] Auto-Paired: 488IMb_tissue.xlsx + 488IMb_background.xlsx
```

!!! warning "Sanity-check your pairs"
    If a "tissue" and "background" file are accidentally identical or
    swapped (it happens, e.g. a copy-paste mistake when exporting from
    the instrument software), `filter_background()` will correctly
    reject every metabolite (fold-change ≈ 1.0 everywhere). If that
    happens, don't assume MORTIS is broken, check
    `stats[0]['fold_change'].max()`; if it's suspiciously close to
    `1.0`, compare the two source files directly (e.g. an md5 checksum)
    before debugging further.

---

## `load_annotation_scores`

```python
mt.load_annotation_scores(
    adata: anndata.AnnData,
    feature_table_path: str,
    compound_col: str = "Compound",
    score_col: str = "Identification Score",
    target_col: str = "score",
) -> anndata.AnnData
```

Merge per-compound annotation confidence scores from a **separate**
feature table into `adata.var[target_col]`.

Many MSI facilities export spatial intensities (pixels x metabolites,
what `read_metabolomics_data` reads) and annotation metadata (one row
per metabolite, with `Compound`, `Chemical Formula`, `Adduct`, `HMDB ID`,
`Identification Score`, etc.) as two **separate** files. This performs
the join needed before [`filter_by_score()`](filtering.md#filter_by_score),
which otherwise requires `adata.var['score']` to already exist, true
for `.h5ad` exports from METASPACE/SCiLS, but not for raw `.xlsx`/`.csv`
tissue exports.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adata` | `AnnData` |, | Loaded via `read_metabolomics_data`/`load_from_folder` |
| `feature_table_path` | `str` |, | Path to a `.xlsx`/`.csv` feature table |
| `compound_col` | `str` | `"Compound"` | Column matching `adata.var_names` |
| `score_col` | `str` | `"Identification Score"` | Column with the confidence score |
| `target_col` | `str` | `"score"` | Column created in `adata.var` |

**Returns:** `AnnData` with `adata.var[target_col]` populated.
Metabolites with no match get `NaN` (which `filter_by_score` treats as
failing any positive threshold, i.e. unmatched metabolites are dropped,
not silently kept).

**Raises:** `FileNotFoundError`; [`FileFormatError`](exceptions.md#fileformaterror)
if `compound_col`/`score_col` aren't in the table.

```python
adata = mt.read_metabolomics_data("sample_tissue.xlsx")
adata = mt.load_annotation_scores(adata, "feature_table.xlsx")
adata = mt.filter_by_score(adata, min_score=0.5)
```

---

## `check_rois`

```python
mt.check_rois(adatas: list[anndata.AnnData]) -> None
```

Verify every `AnnData` in a list has `is_tissue`/`is_background` ROI
labels (set automatically by `load_from_folder`'s auto-pairing, or
manually via [`draw_ROIs()`](roi.md#draw_rois)).

**Raises:** [`MissingROIError`](exceptions.md#missingroierror) listing
which sample indices are missing labels, if any.

```python
try:
    mt.check_rois(adatas)
except mt.MissingROIError as e:
    print(e)  # tells you exactly which samples need ROIs drawn
    adatas = mt.draw_ROIs_for_folder(adatas)
```

---

## `save_spatial_data`

```python
mt.save_spatial_data(
    adatas: list[anndata.AnnData],
    output_dir: str = ".",
    prefix: str = "mortis_sample",
) -> list[pathlib.Path]
```

Save a list of `AnnData` objects to `{prefix}_1.h5ad`, `{prefix}_2.h5ad`....
Use this for saving raw/lightly-processed multi-sample data early in a
pipeline; for a single fully-processed sample, use
[`save_adata()`](analysis-multisample.md#save_adata) instead.

```python
paths = mt.save_spatial_data(adatas, output_dir="results", prefix="patient1")
```

