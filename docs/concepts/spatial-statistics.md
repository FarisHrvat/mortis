# Spatial Statistics Explained

MORTIS has several spatial statistics that sound similar but answer
different questions. This page is a plain-language map between "what do
I want to know" and "which function answers that."

## "Is this metabolite spatially organized at all?"

→ [`mt.spatial_autocorrelation()`](../api/analysis-spatial.md#spatial_autocorrelation)
(Moran's I + Geary's C, computed together)

A single global number per metabolite. Positive Moran's I / Geary's C < 1
means the metabolite is spatially clustered somewhere in the tissue —
but this statistic doesn't tell you *where*.

## "Where, specifically, are this metabolite's hot/cold regions?"

→ [`mt.local_moran()`](../api/analysis-spatial.md#local_moran) (LISA) or
[`mt.getis_ord_gi()`](../api/analysis-spatial.md#getis_ord_gi) (Gi*)

Both give you a **per-pixel** answer. Use LISA if you also care about
spatial *outliers* (a high-value pixel isolated among low-value
neighbours); use Gi* for a cleaner hot/cold-only map (the more standard
choice in GIS/epidemiology hotspot mapping).

## "Do these two tissue regions/clusters tend to sit next to each other?"

→ [`mt.spatial_neighbors()`](../api/analysis-spatial.md#spatial_neighbors) +
[`mt.neighborhood_enrichment()`](../api/analysis-spatial.md#neighborhood_enrichment)

A **graph-based** permutation test: are cluster A and cluster B
significantly more (or less) often *directly adjacent* than chance
would predict?

## "At what physical distance scale do these regions co-occur?"

→ [`mt.co_occurrence()`](../api/analysis-spatial.md#co_occurrence)

A **distance-binned** alternative to neighbourhood enrichment — instead
of a single "adjacent or not" answer, you get a curve: the enrichment
ratio at 10 µm, 50 µm, 100 µm, etc. Useful when you suspect regions
interact at a specific distance rather than only when touching.

## "Do these two metabolites' spatial patterns look similar?"

→ [`mt.metabolite_colocalization()`](../api/analysis-spatial.md#metabolite_colocalization)

Builds a similarity network across many metabolites at once (Pearson or
cosine similarity between ion images), rather than testing one pair.

## "How does intensity change as you move away from a region?"

→ [`mt.spatial_gradient()`](../api/analysis-spatial.md#spatial_gradient)

A **directional/radial** profile: mean intensity binned by distance from
a chosen reference region (e.g. distance from tumour core).

## "Can I find contiguous tissue domains, not just chemical clusters?"

→ [`mt.spatial_domains()`](../api/analysis-clustering.md#spatial_domains) /
[`mt.spatial_domains_kmeans()`](../api/analysis-clustering.md#spatial_domains_kmeans)

These aren't statistics per se — they're spatially-aware *clustering*,
smoothing the feature space across physical neighbours before grouping,
so the result tends toward contiguous regions instead of scattered
chemical clusters.

## Summary table

| Question | Function | Output shape |
|---|---|---|
| Is metabolite X spatially organized at all? | `spatial_autocorrelation` | one number per metabolite |
| Where exactly are its hot/cold spots? | `local_moran` / `getis_ord_gi` | one number per pixel |
| Do clusters A and B sit next to each other? | `neighborhood_enrichment` | one number per cluster pair |
| At what distance do they co-occur? | `co_occurrence` | a curve per cluster pair |
| Do metabolites X and Y look spatially similar? | `metabolite_colocalization` | a similarity network |
| How does intensity change with distance from a region? | `spatial_gradient` | a curve per metabolite |
| Find contiguous tissue domains | `spatial_domains` | cluster labels, spatially smoothed |
