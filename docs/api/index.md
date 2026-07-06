# API Reference

Every MORTIS function lives in a single flat namespace — `mt.function_name(...)`.
No `gr.`/`pl.`/`tl.` submodule gymnastics to remember. This section documents
every public function: what it does, every parameter and its default,
what it returns, what it raises, and a runnable example.

## Organized by what you're trying to do

| Category | What's in it |
|---|---|
| [I/O](io.md) | Load `.h5ad`/`.csv`/`.xlsx` files, auto-pair tissue/background, merge annotation scores, save results |
| [ROI Selection](roi.md) | Interactively draw tissue/background regions when files aren't pre-paired |
| [Preprocessing](preprocessing.md) | Background filtering, normalization, PCA, kNN graph, UMAP, batch correction |
| [Filtering](filtering.md) | Annotation-score filtering, DrugBank xenobiotic filtering |
| [Analysis — Clustering](analysis-clustering.md) | Leiden, NMF, spatially-aware domain clustering |
| [Analysis — Differential Expression](analysis-de.md) | Marker discovery, pairwise/multi-group tests |
| [Analysis — Spatial Statistics](analysis-spatial.md) | Moran's I, Geary's C, Getis-Ord Gi*, LISA, neighbourhood enrichment, co-occurrence, colocalization, gradients |
| [Analysis — Validation & QC](analysis-validation.md) | Silhouette/ARI/AMI cluster validation, batch-mixing (LISI) score |
| [Analysis — Multi-sample & Enrichment](analysis-multisample.md) | Merge/split/subset samples, metabolite set scoring & enrichment, lipid class summary, diversity, unmixing |
| [Plotting](plotting.md) | Every plot function, all shared style parameters |
| [Image](image.md) | Histology image loading, registration, overlay |
| [Exceptions](exceptions.md) | Every custom exception, when it's raised, how to fix it |

## Conventions used throughout

- The first positional argument is always `adata` (an `anndata.AnnData`).
- Functions that **compute a new embedding, graph, or cluster labels**
  (`run_pca`, `run_neighbors`, `run_umap`, `cluster`, `spatial_domains`, ...)
  return the modified `AnnData` alone.
- Functions that **compute a result table** (`find_markers`,
  `compare_groups`, `spatial_autocorrelation`, `neighborhood_enrichment`, ...)
  return a `(AnnData, pandas.DataFrame)` tuple — the `AnnData` with the
  result stashed in `.obs`/`.var`/`.uns` for later plotting, and the
  DataFrame for direct inspection/saving.
- `copy: bool = False` — every mutating function modifies `adata` in
  place by default (matching scanpy's convention); pass `copy=True` to
  get an independent copy back instead.
- `random_state: int = 0` — every stochastic function is seeded by
  default for reproducibility.
- Every function raises a **specific exception** (see
  [Exceptions](exceptions.md)) with an actionable message when a
  prerequisite step is missing, not a bare `KeyError`/`AttributeError`.

## Quick function finder

Not sure which function you need? Search for what you have vs. what you want:

| I have... | I want... | Use |
|---|---|---|
| A folder of tissue/background files | One `AnnData` per sample, auto-paired | [`mt.load_from_folder()`](io.md#load_from_folder) |
| A single `.xlsx`/`.csv`/`.h5ad` | It loaded as `AnnData` | [`mt.read_metabolomics_data()`](io.md#read_metabolomics_data) |
| Pixels including background noise | Only real tissue signal | [`mt.filter_background()`](preprocessing.md#filter_background) |
| A separate feature/annotation table | Scores merged into my data | [`mt.load_annotation_scores()`](io.md#load_annotation_scores) |
| Raw intensities | Normalized + PCA + kNN graph | [`mt.preprocess()`](preprocessing.md#preprocess) |
| A preprocessed sample | Tissue-region clusters | [`mt.cluster()`](analysis-clustering.md#cluster) |
| Clusters | Which metabolites define each one | [`mt.find_markers()`](analysis-de.md#find_markers) |
| Two conditions | What's differentially abundant | [`mt.compare_groups()`](analysis-de.md#compare_groups) |
| One metabolite | Is it spatially clustered? | [`mt.spatial_autocorrelation()`](analysis-spatial.md#spatial_autocorrelation) |
| One metabolite | Exactly *where* are its hot/cold spots | [`mt.getis_ord_gi()`](analysis-spatial.md#getis_ord_gi) or [`mt.local_moran()`](analysis-spatial.md#local_moran) |
| Multiple samples | Batch effects corrected | [`mt.run_harmony()`](preprocessing.md#run_harmony) / [`mt.correct_batches()`](preprocessing.md#correct_batches) |
| A batch-corrected embedding | Proof it actually worked | [`mt.batch_mixing_score()`](analysis-validation.md#batch_mixing_score) |
| A finished analysis | A figure | [Plotting](plotting.md) (pick by what you're showing) |
