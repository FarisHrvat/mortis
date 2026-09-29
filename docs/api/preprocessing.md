# Preprocessing

See also: [Choosing a Normalization Method](../concepts/normalization.md) ·
[Batch Correction & Verifying It Worked](../concepts/batch-correction.md) ·
[Performance & Threading](../concepts/performance.md)

## `filter_background`

```python
mt.filter_background(
    adatas: list[anndata.AnnData],
    cutoff: float = 1.5,
    mode: str = "sample",
) -> tuple[list[anndata.AnnData], list[dict]]
```

Remove metabolites whose tissue signal doesn't exceed background signal
by `cutoff`x.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `adatas` | `list[AnnData]` |, | Must have `is_tissue`/`is_background` in `.obs` (see [ROI Selection](roi.md)) |
| `cutoff` | `float` | `1.5` | Minimum tissue/background fold-change to keep a metabolite. Typical range: 1.5 (lenient) - 3.0 (strict) |
| `mode` | `str` | `"sample"` | `"sample"`, fold-change computed per sample independently. `"group"`, pooled across all samples first |

**Returns:** `(list[AnnData], list[dict])`, tissue-only `AnnData` per
sample, and a QC stats dict per sample (feed directly to
[`plot_qc()`](plotting.md#plot_qc)).

**Raises:** [`MissingROIError`](exceptions.md#missingroierror);
[`InvalidParameterError`](exceptions.md#invalidparametererror) if
`cutoff <= 0` or `mode` isn't `"sample"`/`"group"`.

```python
clean, stats = mt.filter_background(adatas, cutoff=2.0, mode="sample")
mt.plot_qc(stats[0], clean[0], sample_name="Patient 1")
```

---

## `tic_normalize`

```python
mt.tic_normalize(adata, target_sum: float | None = None, copy: bool = False) -> anndata.AnnData
```

Total Ion Current normalization, every pixel is scaled so its row sum
matches a common target. Default target is the **median row sum across
all pixels** (not `1.0`, see [Choosing a Normalization Method](../concepts/normalization.md)
for why).

```python
adata = mt.tic_normalize(adata)                  # default: median target
adata = mt.tic_normalize(adata, target_sum=1.0)   # classic "proportion" TIC
```

## `median_normalize`

```python
mt.median_normalize(adata, target_value: float | None = None, copy: bool = False) -> anndata.AnnData
```

Alternative to TIC: scales each pixel by its own **median non-zero
intensity** rather than the sum. More robust when a few very
high-intensity ions dominate a pixel's total (common with abundant
membrane lipids). See [Choosing a Normalization Method](../concepts/normalization.md).

```python
adata = mt.median_normalize(adata)
```

## `log1p_transform`

```python
mt.log1p_transform(adata, copy: bool = False) -> anndata.AnnData
```

`log(1 + x)` variance-stabilizing transform, compresses the dynamic
range so a handful of extreme values don't dominate downstream PCA.

## `scale`

```python
mt.scale(adata, max_value: float = 10.0, copy: bool = False) -> anndata.AnnData
```

Zero-mean, unit-variance scaling per metabolite, clipped to `max_value`.
Stores the pre-scale values in `adata.layers['log1p']`, used
automatically by marker discovery, differential expression, and
plotting functions so results/figures are reported in interpretable
units, not z-scores.

---

## `run_pca`

```python
mt.run_pca(
    adata, n_comps: int = 50, random_state: int = 0,
    use_hardware: bool = True, copy: bool = False,
) -> anndata.AnnData
```

PCA dimensionality reduction → `adata.obsm['X_pca']`.

| Parameter | Default | Description |
|---|---|---|
| `n_comps` | `50` | Capped automatically at `min(n_obs, n_vars) - 1` |
| `use_hardware` | `True` | Use GPU (`cuml`/CUDA) automatically if importable, else CPU |

**Raises:** [`InvalidParameterError`](exceptions.md#invalidparametererror)
if fewer than 2 metabolites remain.

!!! tip "Speed"
    The CPU path respects `MORTIS_N_JOBS` (see [Performance & Threading](../concepts/performance.md)).
    Set it before starting Python, not after.

## `run_neighbors`

```python
mt.run_neighbors(
    adata, n_neighbors: int = 30, n_pcs: int = 30,
    metric: str = "cosine", random_state: int = 0, copy: bool = False,
) -> anndata.AnnData
```

Computes the kNN graph in PCA space, required before clustering/UMAP.

**Raises:** [`NoEmbeddingError`](exceptions.md#noembeddingerror) if
`adata.obsm['X_pca']` doesn't exist yet.

!!! tip "The first call in a process is slow, and that is normal"
    See [Performance & Threading](../concepts/performance.md#2-the-first-run_neighborsrun_umap-call-in-a-process-is-slow-for-an-unrelated-reason)
    for why: it's a one-time Numba compilation cost, not a bug.

Pass `use_rep="X_pca_harmony"` after [`run_harmony()`](#run_harmony) to
build the graph on the batch-corrected embedding instead of raw PCA.

## `run_umap`

```python
mt.run_umap(
    adata, min_dist: float = 0.3, spread: float = 1.0,
    random_state: int = 0, use_hardware: bool = True, copy: bool = False,
) -> anndata.AnnData
```

UMAP embedding → `adata.obsm['X_umap']`.

**Raises:** [`NoEmbeddingError`](exceptions.md#noembeddingerror) if the
neighbour graph hasn't been computed.

---

## `correct_batches`

```python
mt.correct_batches(
    adata, batch_key: str = "sample", covariates: list[str] | None = None,
    recompute_pca: bool = True, use_hardware: bool = True, copy: bool = False,
) -> anndata.AnnData
```

Batch correction via **ComBat**, applied to the expression matrix
itself. `covariates` (e.g. a biological condition you don't want
"corrected away") are protected from correction if they have ≥ 2
unique values; otherwise MORTIS reverts to standard ComBat and prints a
warning rather than crashing. Recomputes PCA afterward by default.

See [Batch Correction & Verifying It Worked](../concepts/batch-correction.md)
for when to prefer this over `run_harmony`.

```python
adata = mt.correct_batches(adata, batch_key="sample", covariates=["condition"])
```

## `run_harmony`

```python
mt.run_harmony(
    adata, batch_key: str = "sample", adjusted_basis: str = "X_pca_harmony",
    copy: bool = False,
) -> anndata.AnnData
```

Batch correction via [Harmony](https://doi.org/10.1038/s41592-019-0619-0)
(Korsunsky et al. 2019, *Nature Methods*), corrects the PCA embedding
rather than the expression matrix. Generally faster and more robust
than ComBat with many batches/samples.

!!! important "The corrected embedding is separate, point downstream steps at it"
    Harmony's output is stored under a **new** key (default
    `X_pca_harmony`), not overwritten onto `X_pca`:
    ```python
    adata = mt.run_harmony(adata, batch_key="sample")
    adata = mt.run_neighbors(adata, use_rep="X_pca_harmony")
    adata = mt.run_umap(adata)
    ```

**Raises:** [`NoEmbeddingError`](exceptions.md#noembeddingerror) if PCA
hasn't been run; [`InvalidParameterError`](exceptions.md#invalidparametererror)
for an unrecognized `batch_key`.

**Citation:** Korsunsky I, et al. Fast, sensitive and accurate
integration of single-cell data with Harmony. *Nature Methods*, 2019,
16:1289-1296.

---

## `preprocess`

```python
mt.preprocess(
    adata, n_pcs: int = 50, n_pcs_neighbors: int = 10, n_neighbors: int = 15,
    metric: str = "euclidean", do_tic: bool = True,
    normalize_method: str = "tic", do_log1p: bool = True,
    target_sum: float | None = None, scale_data: bool = False,
    max_value: float = 10.0, use_hardware: bool = True, copy: bool = False,
) -> anndata.AnnData
```

Convenience wrapper chaining: normalize (`normalize_method="tic"|"median"`)
→ log1p → (optional) scale → PCA → kNN graph, in one call.

```python
adata = mt.preprocess(adata)                                    # defaults
adata = mt.preprocess(adata, normalize_method="median", n_pcs=30)
adata = mt.preprocess(adata, do_tic=False)                       # already normalized upstream
```

