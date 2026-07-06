# Exceptions

All exceptions subclass `mortis.exceptions.MortisError` and are
exported at the top level (`mt.MissingROIError`, etc.), so you never
need a separate import.

| Exception | When raised | How to fix |
|---|---|---|
| [`MortisError`](#mortiserror) | Base class for all MORTIS errors | — |
| [`MissingROIError`](#missingroierror) | ROI labels missing from `.obs` | Run [`draw_ROIs()`](roi.md#draw_rois) or load paired tissue/background files |
| [`MissingSpatialError`](#missingspatialerror) | No `adata.obsm['spatial']` | Ensure `x`/`y` columns exist in your file |
| [`NotPreprocessedError`](#notpreprocessederror) | Analysis called before preprocessing | Run [`preprocess()`](preprocessing.md#preprocess) first |
| [`NoClustersError`](#noclusterserror) | Clustering step missing | Run [`cluster()`](analysis-clustering.md#cluster) first |
| [`NoEmbeddingError`](#noembeddingerror) | PCA / neighbours / UMAP missing | Run [`run_pca()`](preprocessing.md#run_pca) / [`run_neighbors()`](preprocessing.md#run_neighbors) / [`run_umap()`](preprocessing.md#run_umap) first |
| [`InsufficientSamplesError`](#insufficientsampleserror) | Too few pixels per group | Need ≥ 3 pixels per group |
| [`InvalidParameterError`](#invalidparametererror) | Bad parameter value | Check the error message for valid values |
| [`FileFormatError`](#fileformaterror) | Unreadable or malformed file | Check the file has `x`, `y`, and metabolite columns |

## Every exception carries an actionable message

```python
try:
    adata = mt.filter_by_score(adata, min_score=5.0)
except mt.InvalidParameterError as e:
    print(e)
# min_score must be between 0.0 and 2.0, got 5.0.
# Typical values: 0.3 (moderate), 0.5 (high), 0.8 (very high).
```

## Recommended pattern: catch, then fix, then retry

```python
try:
    mt.check_rois(adatas)
except mt.MissingROIError as e:
    print(e)  # tells you exactly which sample indices need ROIs
    adatas = mt.draw_ROIs_for_folder(adatas)

try:
    adata, markers = mt.find_markers(adata)
except mt.NoClustersError:
    adata = mt.cluster(adata, resolution=0.5)
    adata, markers = mt.find_markers(adata)
```

---

## `MortisError`

Base class for every exception MORTIS raises. Catch this if you want a
single `except` clause for "something in MORTIS went wrong," rather
than enumerating every specific exception type.

## `MissingROIError`

Raised when an `AnnData` is missing `is_tissue`/`is_background` labels,
required before [`filter_background()`](preprocessing.md#filter_background).

**Fix:** run [`draw_ROIs()`](roi.md#draw_rois)/[`draw_ROIs_for_folder()`](roi.md#draw_rois_for_folder),
or load paired tissue/background files via [`load_from_folder()`](io.md#load_from_folder)
so ROI labels are assigned automatically.

## `MissingSpatialError`

Raised when an `AnnData` has no spatial coordinates
(`adata.obsm['spatial']`).

**Fix:** ensure your source file has `x`/`y` columns, or set
`adata.obsm['spatial']` manually before calling a spatial function.

## `NotPreprocessedError`

Raised when a downstream analysis function is called on data that
hasn't been normalized/log-transformed yet.

**Fix:** run [`preprocess()`](preprocessing.md#preprocess), or the
individual [`tic_normalize()`](preprocessing.md#tic_normalize) +
[`log1p_transform()`](preprocessing.md#log1p_transform) steps, first.

## `NoClustersError`

Raised when cluster labels are required but haven't been computed.

**Fix:** run [`cluster()`](analysis-clustering.md#cluster) (or
[`spatial_domains()`](analysis-clustering.md#spatial_domains)/
[`cluster_nmf()`](analysis-clustering.md#cluster_nmf)) first.

## `NoEmbeddingError`

Raised when a dimensionality-reduction embedding (PCA/UMAP) or the kNN
graph is required but hasn't been computed.

**Fix:** run [`run_pca()`](preprocessing.md#run_pca) /
[`run_neighbors()`](preprocessing.md#run_neighbors) /
[`run_umap()`](preprocessing.md#run_umap) first, as appropriate.

## `InsufficientSamplesError`

Raised when a statistical comparison needs more pixels per group than
are actually present (e.g. [`compare_groups()`](analysis-de.md#compare_groups)
needs ≥ 3 pixels per group).

**Fix:** ensure your grouping column has enough pixels in each group,
or merge/subset your data differently.

## `InvalidParameterError`

Raised when a parameter value is outside its accepted range or of the
wrong type. The error message specifies which parameter is invalid and
what values are accepted.

## `FileFormatError`

Raised when a file can't be parsed — an unrecognized extension, or
missing required columns (`x`, `y`, at least one metabolite column).

**Fix:** verify your file's extension is `.h5ad`/`.csv`/`.xlsx` and that
it has the required columns.
