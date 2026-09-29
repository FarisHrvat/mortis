# Plotting

Every plot function returns a `matplotlib.figure.Figure`, modify it
further before saving/displaying if you want (see
[Plot Customization](#plot-customization) below).

## Shared style parameters

All plot functions accept these:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `figsize` | `tuple[float, float]` or `None` | auto | Figure size in inches |
| `dpi` | `int` | `300` | Resolution, use `300`+ for print, `100-150` for web/screen |
| `fontsize` | `int` | `11` | Base font size |
| `show_grid` | `bool` | `False` | Background grid |
| `show_axes_border` | `bool` | `True` | Border around each axes |
| `save` | `bool`, `str`, or `None` | `None` | `None` = display. `True` = save as `mortis_plot.pdf`. A path string = save there. Supports `.pdf`/`.png`/`.svg`/`.tiff`. |
| `show` | `bool` | `True` | Whether to also display interactively |

Spatial-scatter functions (`plot_spatial`, `plot_umap`,
`plot_embedding_grid`, `plot_qc`) accept marker size as either `s=`
(matplotlib-native) or `spot_size=` (more descriptive alias),
both work identically.

## `plot_qc`

```python
mt.plot_qc(stats: dict, clean_adata, sample_name: str = "Sample"...) -> matplotlib.figure.Figure
```

Four-panel QC report after [`filter_background()`](preprocessing.md#filter_background):
fold-change histogram + retention curve, signal-vs-noise scatter,
kept/removed pie chart, spatial TIC map.

```python
mt.plot_qc(stats[0], clean[0], sample_name="Patient 1", dpi=300, save="qc_patient1.pdf")
```

<div class="mortis-figure" markdown>
![QC report](../assets/img/qc_report.png)
<figcaption>A real QC report showing a 100% rejection, because the tissue and background source files were accidentally identical (see the <a href="../tutorials/01-qc-and-scoring/">tutorial</a> for the full story); this is exactly what that failure mode looks like, so you can recognize it if it happens to you.</figcaption>
</div>

---

## `plot_spatial`

```python
mt.plot_spatial(adata, color: str, cmap: str = "viridis", palette: str = "tab20"...) -> matplotlib.figure.Figure
```

Spatial scatter plot (an "ion image" when `color` is a metabolite,
a cluster/domain map when `color` is categorical), colour by any
`adata.obs` column or metabolite name; auto-detects categorical vs.
continuous.

```python
mt.plot_spatial(adata, color="cluster")
mt.plot_spatial(adata, color="Palmitic acid", cmap="hot")
mt.plot_spatial(adata, color="cluster", figsize=(10, 8), fontsize=14, show_grid=True)
```

## `plot_spatial_metabolite`

```python
mt.plot_spatial_metabolite(adata, metabolite: str, cmap: str = "hot"...) -> matplotlib.figure.Figure
```

Single-metabolite ion image, a thin convenience wrapper around
`plot_spatial` with the title pre-filled.

## `plot_embedding_grid`

```python
mt.plot_embedding_grid(adata, metabolites: list[str], ncols: int = 4, cmap: str = "viridis"...) -> matplotlib.figure.Figure
```

Small-multiples grid of ion images, the single most common main-figure
panel type in MSI papers.

```python
top_mets = markers["metabolite"].head(12).tolist()
mt.plot_embedding_grid(adata, top_mets, ncols=4)
```

<div class="mortis-figure" markdown>
![Ion image grid](../assets/img/ion_image_grid_488IMb.png)
<figcaption>Four marker metabolites side by side</figcaption>
</div>

---

## `plot_umap`

```python
mt.plot_umap(adata, color: str = "cluster", palette: str = "tab20", cmap: str = "viridis"...) -> matplotlib.figure.Figure
```

UMAP embedding, coloured by cluster/condition/metabolite.

**Raises:** [`NoEmbeddingError`](exceptions.md#noembeddingerror).

<div class="mortis-img-grid" markdown>
<figure markdown>![UMAP by cluster](../assets/img/umap_clusters_488IMb.png)<figcaption>Coloured by cluster</figcaption></figure>
<figure markdown>![UMAP by condition](../assets/img/umap_condition_groups.png)<figcaption>Coloured by condition (Responder vs. Non-Responder cohort)</figcaption></figure>
</div>

## `plot_markers`

```python
mt.plot_markers(adata, cluster_key: str = "cluster", n_top: int = 5...) -> matplotlib.figure.Figure
```

Dot plot of top marker metabolites per cluster (dot size = fraction
expressing, colour = mean expression).

**Raises:** [`NoClustersError`](exceptions.md#noclusterserror) if
[`find_markers()`](analysis-de.md#find_markers) hasn't run.

## `plot_volcano`

```python
mt.plot_volcano(
    results_df, group1: str = "Group 1", group2: str = "Group 2",
    fc_cutoff: float = 1.0, pval_cutoff: float = 0.05, n_label: int = 10,
    show_table: bool = False, n_table: int = 5...
) -> matplotlib.figure.Figure
```

Volcano plot for [`compare_groups()`](analysis-de.md#compare_groups)
output, with an optional top-hits table panel.

```python
mt.plot_volcano(results, group1="Healthy", group2="Tumour", show_table=True, n_table=5)
```

## `plot_heatmap`

```python
mt.plot_heatmap(adata, metabolites: list[str], groupby: str = "cluster", cmap: str = "RdBu_r"...) -> matplotlib.figure.Figure
```

Mean intensity heatmap across groups/clusters.

## `plot_violin`

```python
mt.plot_violin(adata, metabolites: str | list[str], groupby: str = "cluster"...) -> matplotlib.figure.Figure
```

Intensity distributions per group.

<div class="mortis-img-grid" markdown>
<figure markdown>![DE heatmap](../assets/img/de_heatmap_groups.png)<figcaption><code>plot_heatmap()</code> on the top differential metabolites</figcaption></figure>
<figure markdown>![DE violin](../assets/img/de_violin_groups.png)<figcaption><code>plot_violin()</code> on the top 3</figcaption></figure>
</div>

## `plot_cluster_composition`

```python
mt.plot_cluster_composition(adata, cluster_key: str = "cluster", groupby: str = "condition", normalize: bool = True...) -> matplotlib.figure.Figure
```

Stacked bar chart of cluster proportions per condition/sample.

<div class="mortis-figure" markdown>
![Cluster composition](../assets/img/cluster_composition_groups.png)
<figcaption>Cluster proportions per sample</figcaption>
</div>

## `plot_morans`

```python
mt.plot_morans(morans_df, n_top: int = 20...) -> matplotlib.figure.Figure
```

Horizontal bar chart of top spatially variable metabolites (output of
[`spatial_autocorrelation()`](analysis-spatial.md#spatial_autocorrelation)).

## `plot_spatial_gradient`

```python
mt.plot_spatial_gradient(gradient_df, top_n: int = 5...) -> matplotlib.figure.Figure
```

Shaded line plot of intensity vs. distance from a target region
(output of [`spatial_gradient()`](analysis-spatial.md#spatial_gradient)).

## `plot_colocalization_network`

```python
mt.plot_colocalization_network(edges_df...) -> matplotlib.figure.Figure
```

Network graph of co-localized metabolites (output of
[`metabolite_colocalization()`](analysis-spatial.md#metabolite_colocalization)).
Requires the `image-network` extra (`pip install "mortis-spatial[image-network]"`)
for `networkx`.

---

## Plot Customization

Every plot function returns the `Figure`, so customize freely:

```python
fig = mt.plot_umap(adata, color="cluster")
fig.axes[0].set_title("My Custom Title", fontsize=16)
fig.savefig("custom.pdf", bbox_inches="tight", dpi=300)
```

```python
# Publication style: no border, larger font, high DPI, vector format
mt.plot_umap(adata, color="cluster", figsize=(6, 5), dpi=300, fontsize=13,
             show_axes_border=False, save="pub_umap.pdf")

# TIFF (raster, required by some journals)
mt.plot_spatial(adata, color="cluster", dpi=600, save="figure.tiff")
```

