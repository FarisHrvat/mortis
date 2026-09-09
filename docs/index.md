---
title: MORTIS
hide:
  - navigation
  - toc
---

<div class="mortis-hero" markdown>

# MORTIS

<p class="tagline">
Downstream analysis for spatial metabolomics (imaging mass
spectrometry) — from raw instrument export to publication-ready figures,
in one consistent Python API built on <strong>AnnData</strong> and
<strong>scanpy</strong>.
</p>

[Get Started :material-arrow-right:](getting-started/installation.md){ .md-button .md-button--primary }
[See it on real data :material-image-multiple:](tutorials/index.md){ .md-button }

</div>

```python
import mortis as mt

adatas = mt.load_from_folder("./data")               # auto-pairs tissue + background
clean, stats = mt.filter_background(adatas, cutoff=1.5)
adata = mt.filter_by_score(clean[0], min_score=0.3)
adata = mt.preprocess(adata)                          # normalize -> log1p -> PCA -> kNN
adata = mt.cluster(adata, resolution=0.5)

mt.plot_spatial(adata, color="cluster", save="clusters.pdf")
```

## Why MORTIS?

<div class="mortis-grid" markdown>

<div class="mortis-card" markdown>
### :material-map-marker-radius: Spatially aware, not just borrowed
Moran's I, Geary's C, Getis-Ord Gi* hotspots, neighbourhood enrichment,
distance-binned co-occurrence, and spatially-smoothed "niche" clustering
— all built for pixel-grid tissue data, not repurposed single-cell tools.
</div>

<div class="mortis-card" markdown>
### :material-check-decagram: Verified, not just claimed
The custom spatial statistics are cross-checked against
[esda/PySAL](https://pysal.org/esda/) — an independent, published
implementation. The scanpy/sklearn/scipy wrappers are verified to
produce numerically identical results to calling those libraries
directly. See [the test suite](https://github.com/FarisHrvat/mortis/blob/main/tests/test_correctness_vs_reference.py).
</div>

<div class="mortis-card" markdown>
### :material-file-document-multiple: One consistent API
No `gr.`/`pl.`/`tl.` namespace gymnastics — every function is
`mt.function_name(adata, ...)`, returns what you'd expect, and raises a
specific, actionable exception (`mt.NoClustersError`, not a bare
`KeyError`) when a prerequisite step is missing.
</div>

<div class="mortis-card" markdown>
### :material-chart-box: Real facility data, not toy examples
Every tutorial in these docs runs on genuine MALDI-MSI exports — paired
tissue/background `.xlsx` files, METASPACE-annotated `.h5ad` samples,
and a Responder vs. Non-Responder clinical cohort — not synthetic
placeholder data. (The raw files themselves are private patient data
and are never published; only the derived plots and results are shown
here.)
</div>

</div>

## What a complete analysis looks like

<div class="mortis-img-grid" markdown>
<figure markdown>
  ![Spatial clusters](assets/img/spatial_clusters_healthy.png)
  <figcaption>Leiden clustering in chemistry space — <code>mt.cluster()</code> + <code>mt.plot_spatial()</code></figcaption>
</figure>
<figure markdown>
  ![Spatial domains](assets/img/spatial_domains_healthy.png)
  <figcaption>Spatially-smoothed tissue domains — <code>mt.spatial_domains()</code></figcaption>
</figure>
<figure markdown>
  ![Volcano plot](assets/img/volcano_responder_vs_nonresponder.png)
  <figcaption>Responder vs. Non-Responder differential expression — <code>mt.compare_groups()</code> + <code>mt.plot_volcano()</code></figcaption>
</figure>
<figure markdown>
  ![Hotspot map](assets/img/hotspot_map_healthy.png)
  <figcaption>Getis-Ord Gi* hotspot detection — <code>mt.getis_ord_gi()</code></figcaption>
</figure>
</div>

[See the full gallery, with every plot type and the code that produced it :material-arrow-right:](gallery.md){ .md-button }

## Two ways to read these docs

<div class="mortis-grid" markdown>
<div class="mortis-card" markdown>
<span class="mortis-badge beginner">New to Python</span>

Start at **[Installation](getting-started/installation.md)**, then
**[Quickstart](getting-started/quickstart.md)** — it assumes nothing,
explains every command before you run it, and gets you from a raw file
to your first figure without needing to already know Python.
</div>

<div class="mortis-card" markdown>
<span class="mortis-badge developer">Already know scanpy/AnnData</span>

Skip to the **[API Reference](api/index.md)** — every function,
parameter, default, exception, and a runnable example. Or jump straight
into a **[tutorial](tutorials/index.md)** to see the whole pipeline on
real data end to end.
</div>
</div>

---

*MORTIS is developed by [Faris Hrvat](https://github.com/FarisHrvat). Source
available on request; see [Citation](getting-started/installation.md#citation) for how to reference it.*
