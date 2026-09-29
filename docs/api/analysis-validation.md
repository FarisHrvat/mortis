# Analysis. Validation & QC

These functions answer "did that actually work?", for clustering
choices and for batch correction, instead of assuming it did.

## `cluster_validation`

```python
mt.cluster_validation(
    adata, cluster_key: str = "cluster", use_rep: str = "X_pca",
    sample_size: int | None = 10000, random_state: int = 0,
) -> float
```

Silhouette score (scikit-learn) for an existing clustering: how well
separated the clusters are in `use_rep` space, in `[-1, 1]` (higher =
better separated). Use this to choose/report a Leiden `resolution`
objectively instead of eyeballing the spatial map.

`sample_size` subsamples pixels for speed (exact silhouette is O(n²));
pass `None` to use every pixel.

**Raises:** [`NoClustersError`](exceptions.md#noclusterserror);
[`NoEmbeddingError`](exceptions.md#noembeddingerror);
[`InvalidParameterError`](exceptions.md#invalidparametererror) if fewer
than 2 clusters exist.

```python
for res in [0.1, 0.3, 0.5, 1.0]:
    adata = mt.cluster(adata, resolution=res, key_added=f"cluster_{res}")
    score = mt.cluster_validation(adata, cluster_key=f"cluster_{res}")
    print(f"resolution={res}: silhouette={score:.3f}")
```

## `compare_clusterings`

```python
mt.compare_clusterings(labels_a, labels_b) -> dict[str, float]
```

Adjusted Rand Index (ARI) and Adjusted Mutual Information (AMI) between
two cluster label arrays of the same pixels, e.g. checking resolution
stability, or comparing two samples' cluster assignments. Both scores
are `1.0` for identical labelings and ~`0` for random/independent ones;
unlike raw accuracy, both are invariant to how cluster IDs are permuted.

**Raises:** [`InvalidParameterError`](exceptions.md#invalidparametererror)
if the two arrays have different lengths.

```python
result = mt.compare_clusterings(adata.obs["cluster_0.3"], adata.obs["cluster_0.5"])
print(f"ARI={result['ari']:.3f}, AMI={result['ami']:.3f}")
```

---

## `batch_mixing_score`

```python
mt.batch_mixing_score(
    adata, batch_key: str = "sample", use_rep: str = "X_pca",
    n_neighbors: int = 30, copy: bool = False,
) -> anndata.AnnData
```

LISI (Local Inverse Simpson's Index; Korsunsky et al. 2019, the Harmony
paper), verifies whether batch correction ([`run_harmony()`](preprocessing.md#run_harmony),
[`correct_batches()`](preprocessing.md#correct_batches)) **actually
worked**, rather than assuming it did.

For each pixel, computes the effective number of distinct batches
represented among its nearest neighbours in `use_rep` space:
`1.0` = neighbourhood is a single batch (no mixing) · approaching the
true batch count = well mixed. Adds `adata.obs['lisi_score']`.

**Raises:** [`NoEmbeddingError`](exceptions.md#noembeddingerror);
[`InvalidParameterError`](exceptions.md#invalidparametererror).

```python
adata = mt.batch_mixing_score(adata, batch_key="sample", use_rep="X_pca")
print("before harmony:", adata.obs["lisi_score"].mean())

adata = mt.run_harmony(adata, batch_key="sample")
adata = mt.batch_mixing_score(adata, batch_key="sample", use_rep="X_pca_harmony")
print("after harmony:", adata.obs["lisi_score"].mean())
```

On the real 14-sample Responder/Non-Responder cohort (see the
[Cohort Comparison tutorial](../tutorials/04-cohort-comparison.md)):
mean LISI went from **2.13 → 2.69** out of a maximum of 14 (one score
per batch) after Harmony, real improvement, not a dramatic one,
exactly the kind of honest number you should expect to see and should
report rather than assume.

See [Batch Correction & Verifying It Worked](../concepts/batch-correction.md)
for the full picture.

