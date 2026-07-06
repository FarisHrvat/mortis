# Analysis — Spatial Statistics

See also: [Spatial Statistics Explained](../concepts/spatial-statistics.md)
for a plain-language guide to which statistic answers which question.

## `spatial_autocorrelation`

```python
mt.spatial_autocorrelation(
    adata, n_neighbors: int = 6, batch_key: str = "sample",
    use_fdr: bool = False, copy: bool = False,
) -> tuple[anndata.AnnData, pandas.DataFrame]
```

Computes **Moran's I** and **Geary's C** for every metabolite in one
call (both reuse the same spatial weights, so it's cheap to get both).

- **Moran's I** > 0 → clustered ; ≈ 0 → random ; < 0 → dispersed (checkerboard-like)
- **Geary's C** < 1 → clustered ; ≈ 1 → random ; > 1 → dispersed

They usually agree on direction but weight things differently — Geary's
C is more sensitive to sharp *local* discontinuities, Moran's I to the
*overall* global pattern.

**Returns:** `(AnnData, DataFrame)` with `morans_i`, `z_score`, `pval`,
`pval_adj`, `geary_c`, `geary_z_score`, `geary_pval`, `geary_pval_adj` —
sorted by Moran's I descending. Also written to `adata.var`.

**Verified against [esda/PySAL](https://pysal.org/esda/)** (an
independent published implementation) — see
[`tests/test_correctness_vs_reference.py`](https://github.com/FarisHrvat/mortis/blob/main/tests/test_correctness_vs_reference.py).

```python
adata, morans = mt.spatial_autocorrelation(adata, n_neighbors=6)
mt.plot_morans(morans, n_top=20)
```

<div class="mortis-figure" markdown>
![Moran's I bar chart](../assets/img/morans_bar_healthy.png)
<figcaption>Top 20 spatially variable metabolites by Moran's I</figcaption>
</div>

## `spatial_de`

```python
mt.spatial_de(adata, n_top: int = 50, n_neighbors: int = 6, use_fdr: bool = False, copy: bool = False) -> tuple[anndata.AnnData, pandas.DataFrame]
```

Convenience wrapper: the top-`n_top` significant (p < 0.05) spatially
variable metabolites by Moran's I, in one call.

---

## `local_moran`

```python
mt.local_moran(adata, metabolite: str, n_neighbors: int = 6, batch_key: str = "sample", copy: bool = False) -> tuple[anndata.AnnData, pandas.DataFrame]
```

LISA (Local Indicators of Spatial Association) for **one** metabolite:
classifies each pixel as a High-High/Low-Low hotspot, a High-Low/Low-High
**spatial outlier**, or not significant.

**Returns:** `(AnnData, DataFrame[x, y, value, local_i, z_score, pval, lisa_type])`.
Adds `adata.obs[f'{metabolite}_lisa']` / `_lisa_type`.

```python
adata, lisa = mt.local_moran(adata, "Palmitic acid")
mt.plot_spatial(adata, color="Palmitic acid_lisa_type")
```

## `getis_ord_gi`

```python
mt.getis_ord_gi(adata, metabolite: str, n_neighbors: int = 6, batch_key: str = "sample", copy: bool = False) -> tuple[anndata.AnnData, pandas.DataFrame]
```

Getis-Ord Gi* hotspot statistic — a per-pixel z-score testing whether a
pixel **and its neighbours together** (self included) form a
significant hot or cold cluster. The standard "hotspot map" statistic
in GIS/spatial-epidemiology tooling.

!!! note "Gi* vs. LISA — which one do I want?"
    `local_moran` (LISA) also flags spatial *outliers* (a high pixel
    surrounded by low neighbours). `getis_ord_gi` only flags concordant
    hot/cold clusters. If you specifically want a clean hotspot map
    without outlier noise, use Gi*; if outliers themselves are
    interesting to you, use LISA.

**Returns:** `(AnnData, DataFrame[x, y, value, gi_star, pval, hotspot_type])`.
`hotspot_type` is `"hot"`, `"cold"`, or `"NS"`.

**Verified against esda/PySAL's `G_Local`** — correlation > 0.999 on
synthetic ground truth (see the note in the function's own docstring
about a normalization-convention difference in absolute scale, which
doesn't affect which pixels are flagged).

```python
adata, gi_df = mt.getis_ord_gi(adata, "D-Glucose")
mt.plot_spatial(adata, color="D-Glucose_gi_type")
```

<div class="mortis-figure" markdown>
![Hotspot map](../assets/img/hotspot_map_healthy.png)
<figcaption>Getis-Ord Gi* hot/cold spot classification for the top spatially-variable metabolite</figcaption>
</div>

---

## `spatial_neighbors`

```python
mt.spatial_neighbors(adata, n_neighbors: int = 6, batch_key: str = "sample", copy: bool = False) -> anndata.AnnData
```

Build a symmetric physical-distance kNN graph in
`adata.obsp['spatial_connectivities']` — required before
[`neighborhood_enrichment()`](#neighborhood_enrichment) and used
internally by [`spatial_domains()`](analysis-clustering.md#spatial_domains)
/[`spatially_weighted_nmf()`](analysis-clustering.md#spatially_weighted_nmf).

**Raises:** [`MissingSpatialError`](exceptions.md#missingspatialerror).

## `neighborhood_enrichment`

```python
mt.neighborhood_enrichment(
    adata, cluster_key: str = "cluster", n_permutations: int = 1000,
    n_jobs: int | None = None, random_state: int = 0, copy: bool = False,
) -> tuple[anndata.AnnData, pandas.DataFrame]
```

Permutation test (Numba-JIT accelerated) for whether cluster pairs
co-localize (or avoid each other) more than expected by chance.

| Parameter | Default | Description |
|---|---|---|
| `n_jobs` | `None` | Threads for the permutation loop; `None` = `MORTIS_N_JOBS` env var or all CPU cores. Genuinely changes wall time — see [Performance & Threading](../concepts/performance.md). |

**Returns:** `(AnnData, DataFrame[cluster_a, cluster_b, observed, expected, zscore, pval])`.

**Raises:** [`NoClustersError`](exceptions.md#noclusterserror);
[`InvalidParameterError`](exceptions.md#invalidparametererror) if
`spatial_neighbors()` hasn't been run.

```python
adata = mt.spatial_neighbors(adata, n_neighbors=6)
adata, enrich = mt.neighborhood_enrichment(adata)
print(enrich.sort_values("zscore", ascending=False).head(10))
```

## `co_occurrence`

```python
mt.co_occurrence(
    adata, cluster_key: str = "cluster", n_bins: int = 25,
    max_dist: float | None = None, batch_key: str = "sample", copy: bool = False,
) -> pandas.DataFrame
```

Distance-**binned** co-occurrence probability between cluster/domain
labels (analogous to squidpy's `gr.co_occurrence`). Distinct from
`neighborhood_enrichment`: uses continuous physical distance bins
rather than a fixed-degree graph, revealing *at what distance scale*
two regions co-occur, not just whether they're graph-adjacent.

**Returns:** `DataFrame[cluster_a, cluster_b, bin, distance, ratio]` —
`ratio` is `P(b | within this distance of a) / P(b)`; > 1 means
enrichment, < 1 means depletion, at that distance.

```python
df = mt.co_occurrence(adata, cluster_key="domain", n_bins=25)
```

---

## `metabolite_colocalization`

```python
mt.metabolite_colocalization(
    adata, top_n: int = 50, corr_threshold: float = 0.4,
    use_spatial_smooth: bool = True, metric: str = "pearson",
) -> pandas.DataFrame
```

Pairwise ion-image similarity network among the top-`top_n` spatially
variable metabolites (run [`spatial_autocorrelation()`](#spatial_autocorrelation) first).

| Parameter | Default | Description |
|---|---|---|
| `metric` | `"pearson"` | `"pearson"` — correlation between (optionally spatially-smoothed) ion images. `"cosine"` — matches METASPACE's own colocalization metric / ColocML; more sensitive to shared on/off spatial patterns in sparse data. |

**Raises:** [`InvalidParameterError`](exceptions.md#invalidparametererror)
if `spatial_autocorrelation()` hasn't been run.

```python
edges = mt.metabolite_colocalization(adata, top_n=50, corr_threshold=0.4)
mt.plot_colocalization_network(edges)
```

<div class="mortis-figure" markdown>
![Colocalization network](../assets/img/colocalization_network_healthy.png)
<figcaption>Pearson colocalization network among the top 40 spatially variable metabolites</figcaption>
</div>

---

## `spatial_gradient`

```python
mt.spatial_gradient(
    adata, target_col: str, target_val: str, batch_key: str = "sample",
    bins: int = 15, max_dist: float = 1000.0,
) -> pandas.DataFrame
```

Bins all pixels by physical distance to the nearest pixel matching
`target_col == target_val` (e.g. distance from tumour core) and reports
mean metabolite intensity per distance bin — a metabolic gradient
profile.

```python
grad_df = mt.spatial_gradient(adata, target_col="cluster", target_val="0", max_dist=500.0)
mt.plot_spatial_gradient(grad_df, top_n=5)
```

<div class="mortis-figure" markdown>
![Spatial gradient](../assets/img/spatial_gradient_glass.png)
<figcaption>Top metabolites by intensity change with distance from cluster "0"</figcaption>
</div>
