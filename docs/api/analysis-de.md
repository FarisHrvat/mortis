# Analysis — Differential Expression

## `find_markers`

```python
mt.find_markers(
    adata, cluster_key: str = "cluster", method: str = "wilcoxon",
    n_top: int = 20, copy: bool = False,
) -> tuple[anndata.AnnData, pandas.DataFrame]
```

One-vs-rest marker metabolites for each cluster.

| Parameter | Default | Description |
|---|---|---|
| `method` | `"wilcoxon"` | `"wilcoxon"` (recommended, non-parametric) · `"t-test"` · `"logreg"` |
| `n_top` | `20` | Top markers reported per cluster |

**Returns:** `(AnnData, DataFrame[cluster, metabolite, score, pval, pval_adj, log2fc])`.
Raw result also stored in `adata.uns['rank_genes_groups']` (scanpy-compatible).

**Raises:** [`NoClustersError`](exceptions.md#noclusterserror);
[`NotPreprocessedError`](exceptions.md#notpreprocessederror);
[`InvalidParameterError`](exceptions.md#invalidparametererror) for an
unknown `method`.

```python
adata, markers = mt.find_markers(adata, n_top=20)
mt.plot_markers(adata, n_top=5)
mt.save_results(markers, "markers.csv")
```

<div class="mortis-img-grid" markdown>
<figure markdown>![Marker dotplot](../assets/img/marker_dotplot_488IMb.png)<figcaption><code>plot_markers()</code> — dot size = fraction expressing, colour = mean expression</figcaption></figure>
<figure markdown>![Marker heatmap](../assets/img/marker_heatmap_488IMb.png)<figcaption><code>plot_heatmap()</code> on the same markers, grouped by cluster</figcaption></figure>
</div>

---

## `compare_groups`

```python
mt.compare_groups(
    adata, groupby: str, group1: str, group2: str,
    method: str = "wilcoxon", copy: bool = False,
) -> tuple[anndata.AnnData, pandas.DataFrame]
```

Pairwise differential metabolite test between two groups, with FDR
correction (Benjamini-Hochberg).

| Parameter | Default | Description |
|---|---|---|
| `groupby` | — | Column in `adata.obs` defining the groups (e.g. `"condition"`, `"cluster"`) |
| `method` | `"wilcoxon"` | `"wilcoxon"`/`"mannwhitney"` (equivalent) or `"ttest"` |

**Returns:** `(AnnData, DataFrame)` with columns `metabolite`,
`mean_group1`, `mean_group2`, `log2fc`, `cohen_d`, `statistic`, `pval`,
`pval_adj`, `significant` (FDR < 0.05) — sorted by `pval_adj`.

**Raises:** [`InsufficientSamplesError`](exceptions.md#insufficientsampleserror)
(need ≥ 3 pixels/group); [`InvalidParameterError`](exceptions.md#invalidparametererror)
for an unknown `groupby`/group value.

```python
adata, results = mt.compare_groups(adata, groupby="condition", group1="healthy", group2="tumour")
sig = results[results["significant"]]
print(f"{len(sig)} significantly different metabolites")
mt.plot_volcano(results, group1="healthy", group2="tumour", show_table=True)
```

**Verified numerically identical** to calling
`scipy.stats.mannwhitneyu` + `statsmodels.stats.multitest.multipletests`
directly on the same data — see
[`tests/test_correctness_vs_reference.py`](https://github.com/FarisHrvat/mortis/blob/main/tests/test_correctness_vs_reference.py).

<div class="mortis-figure" markdown>
![Volcano plot](../assets/img/volcano_responder_vs_nonresponder.png)
<figcaption>Responder vs. Non-Responder cohort, with the optional top-hits table</figcaption>
</div>

---

## `multi_group_test`

```python
mt.multi_group_test(adata, groupby: str, method: str = "kruskal", copy: bool = False) -> tuple[anndata.AnnData, pandas.DataFrame]
```

Omnibus test across **≥ 2** groups — use this instead of many pairwise
`compare_groups()` calls when comparing more than two conditions.

| Parameter | Default | Description |
|---|---|---|
| `method` | `"kruskal"` | `"kruskal"` (Kruskal-Wallis, non-parametric) or `"anova"` |

**Returns:** `(AnnData, DataFrame)` with `statistic`, `pval`, `pval_adj`,
`significant`, and `eta_squared` (effect size).

**Raises:** [`InvalidParameterError`](exceptions.md#invalidparametererror)
for an unknown `method`/`groupby`, or fewer than 2 groups present.

```python
adata, results = mt.multi_group_test(adata, groupby="tissue_region", method="kruskal")
```
