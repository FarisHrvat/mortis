# Analysis — Multi-sample & Enrichment

## Multi-sample data management

### `subset_obs`

```python
mt.subset_obs(adata, obs_col: str, value: str | list[str], copy: bool = True) -> anndata.AnnData
```

Subset to one or more values of any `adata.obs` column (not just
"sample" despite similarly-named tools elsewhere — this works on
`condition`, `cluster`, `patient`, anything).

**Raises:** [`InvalidParameterError`](exceptions.md#invalidparametererror)
if `obs_col` doesn't exist or no rows match `value`.

```python
responders_only = mt.subset_obs(adata, "condition", "Responder")
two_conditions = mt.subset_obs(adata, "condition", ["Responder", "Non Responder"])
```

### `merge_samples`

```python
mt.merge_samples(
    adatas: list[anndata.AnnData], sample_labels: list[str] | None = None,
    sample_col: str = "sample", join: str = "inner",
) -> anndata.AnnData
```

Concatenate multiple samples into one `AnnData`, adding a `sample_col`
label. `join="inner"` keeps only metabolites present in every sample;
`join="outer"` keeps all metabolites (filling missing ones with `0.0`).

**Raises:** [`InvalidParameterError`](exceptions.md#invalidparametererror)
if `sample_labels` length doesn't match `len(adatas)`.

```python
merged = mt.merge_samples(adatas, sample_labels=["P1", "P2", "P3"], join="outer")
```

### `split_by_obs`

```python
mt.split_by_obs(adata, obs_col: str, copy: bool = True) -> dict[str, anndata.AnnData]
```

Inverse of `merge_samples` — splits one `AnnData` into a
`{value: AnnData}` dict by any obs column.

```python
by_condition = mt.split_by_obs(merged, "condition")
# {"Responder": AnnData(...), "Non Responder": AnnData(...)}
```

---

## Metabolite set scoring & enrichment

### `score_metabolite_set`

```python
mt.score_metabolite_set(adata, metabolites: list[str], score_name: str = "metabolite_set_score", copy: bool = False) -> anndata.AnnData
```

Per-pixel mean intensity across a metabolite set (pathway, lipid class,
etc.) → `adata.obs[score_name]`. Warns (doesn't fail) if some
metabolites aren't found; only raises if *none* are found.

**Raises:** [`InvalidParameterError`](exceptions.md#invalidparametererror)
if none of the metabolites are found.

```python
fatty_acids = ["Palmitic acid", "Stearic acid", "Oleic acid"]
adata = mt.score_metabolite_set(adata, fatty_acids, score_name="fatty_acids")
mt.plot_spatial(adata, color="fatty_acids", cmap="RdYlBu_r")
```

### `metabolite_set_enrichment`

```python
mt.metabolite_set_enrichment(
    results_df: pandas.DataFrame, metabolite_sets: dict[str, list[str]],
    score_col: str = "log2fc", min_set_size: int = 3,
    n_permutations: int = 1000, random_state: int = 0,
) -> pandas.DataFrame
```

GSEA-style enrichment of metabolite sets against a ranked differential
expression result ([`compare_groups()`](analysis-de.md#compare_groups) output).

**Returns:** `DataFrame` with enrichment score (`es`), normalized
enrichment score (`nes`), permutation `pval`/`pval_adj`, and
`leading_edge` metabolites.

```python
adata, de_results = mt.compare_groups(adata, "condition", "healthy", "tumour")
pathways = {"fatty_acid_metabolism": fatty_acids}
enrichment = mt.metabolite_set_enrichment(de_results, pathways)
```

### `lipid_class_summary`

```python
mt.lipid_class_summary(adata, groupby: str | None = None) -> pandas.DataFrame
```

Classifies metabolites by lipid-class name prefix (PC, PE, TG, SM, Cer,
LPC, FA, CE, ...) and reports per-class metabolite counts and mean
intensity, optionally split by `groupby`.

```python
lipid_summary = mt.lipid_class_summary(adata, groupby="condition")
```

---

## Diversity & unmixing

### `diversity_index`

```python
mt.diversity_index(adata, method: str = "shannon", copy: bool = False) -> anndata.AnnData
```

Per-pixel metabolomic heterogeneity, treating each pixel's intensities
as a compositional distribution over metabolites. `method="shannon"`
(entropy, higher = more even/diverse) or `"simpson"` (Gini-Simpson
index, bounded `[0, 1)`, more sensitive to a single dominant metabolite).

```python
adata = mt.diversity_index(adata, method="shannon")
mt.plot_spatial(adata, color="shannon_diversity", cmap="magma")
```

<div class="mortis-figure" markdown>
![Diversity map](../assets/img/diversity_map_healthy.png)
<figcaption>Per-pixel Shannon diversity — regions of even, "unremarkable" chemistry (high diversity, bright) vs. regions dominated by a few metabolites (low diversity, dark)</figcaption>
</div>

### `cluster_diversity`

```python
mt.cluster_diversity(adata, cluster_key: str = "cluster", groupby: str = "sample", method: str = "shannon") -> pandas.DataFrame
```

Region-level heterogeneity: diversity of cluster/domain **composition**
within each group (e.g. per sample) — how mixed vs. homogeneous each
sample's tissue-domain makeup is.

**Raises:** [`NoClustersError`](exceptions.md#noclusterserror);
[`InvalidParameterError`](exceptions.md#invalidparametererror).

```python
df = mt.cluster_diversity(adata, cluster_key="domain", groupby="sample")
```

### `unmix_pixels`

```python
mt.unmix_pixels(adata, reference_spectra: dict[str, dict[str, float]], copy: bool = False) -> anndata.AnnData
```

Non-negative least squares (NNLS) unmixing against a library of known
reference spectra — recovers, per pixel, the non-negative mixing
fractions that best reconstruct its intensities as a combination of the
references. MSI pixels routinely contain mixed signal from more than
one underlying tissue/cell population, so this is a standard MSI-native
analysis category when reference spectra are known a priori.

**Returns:** `AnnData` with `adata.obsm['X_unmixed']` (fractions summing
to 1 per pixel) and one `adata.obs['fraction_{name}']` per reference.

**Raises:** [`InvalidParameterError`](exceptions.md#invalidparametererror)
if `reference_spectra` is empty or shares no metabolites with `adata`.

```python
references = {
    "epithelium": {"met_a": 1.0, "met_b": 0.2, ...},
    "stroma": {"met_a": 0.1, "met_b": 0.9, ...},
}
adata = mt.unmix_pixels(adata, references)
mt.plot_spatial(adata, color="fraction_epithelium")
```

---

## `run_paga`

```python
mt.run_paga(adata, cluster_key: str = "cluster", copy: bool = False) -> anndata.AnnData
```

PAGA (partition-based graph abstraction) — coarse connectivity between
clusters, useful for understanding gradual metabolic transitions
between tissue regions.

**Raises:** [`NoClustersError`](exceptions.md#noclusterserror);
[`NoEmbeddingError`](exceptions.md#noembeddingerror).

---

## Saving results

### `save_results`

```python
mt.save_results(results_df: pandas.DataFrame, path: str) -> None
```

Any results `DataFrame` → CSV, creating parent directories as needed.

### `save_adata`

```python
mt.save_adata(adata, path: str) -> None
```

Full `AnnData` (UMAP, clusters, layers, `.uns`) → `.h5ad`.

```python
mt.save_results(markers, "results/markers.csv")
mt.save_adata(adata, "results/patient1_processed.h5ad")
```
