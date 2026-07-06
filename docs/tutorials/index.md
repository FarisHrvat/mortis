# Tutorials — Real Data Walkthroughs

Every plot in these tutorials was generated from genuine MALDI-MSI
facility exports — not synthetic placeholder data. The raw data files
themselves are private patient data and are **not** published anywhere;
only the derived plots, summary statistics, and code that produced them
are shown here.

<span class="mortis-badge beginner">New to Python?</span> Read
[Quickstart](../getting-started/quickstart.md) first — these tutorials
assume you already know what `adata`, `import mortis as mt`, etc. mean.

| # | Tutorial | What it covers | Data |
|---|---|---|---|
| 1 | [QC & Annotation Scoring](01-qc-and-scoring.md) | Background filtering, a real QC failure and how to recognize it, merging a separate annotation-score feature table, clustering, marker discovery | Paired tissue/background `.xlsx` export |
| 2 | [Single-Sample Deep Dive](02-single-sample-deep-dive.md) | Spatial domains vs. chemistry clustering, Moran's I, Getis-Ord Gi* hotspots, colocalization networks, diversity mapping | METASPACE-annotated `.h5ad` |
| 3 | [Large-Scale Microenvironments](03-large-scale-microenvironments.md) | NMF vs. spatially-weighted NMF, spatial gradients, cluster composition, at 30,000-pixel scale | Large `.h5ad` sample |
| 4 | [Cohort Comparison](04-cohort-comparison.md) | Merging many samples, differential expression, batch correction, and — critically — **verifying** batch correction worked | 14-sample Responder/Non-Responder cohort |

Each tutorial ends with a short **"What would I change for my own
data?"** section, since real datasets never look exactly like the
examples.
