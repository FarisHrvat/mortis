# Complete Workflow

<span class="mortis-badge developer">The full pipeline, every step, for readers already comfortable with Python</span>

This is the same pipeline as the [Quickstart](quickstart.md), extended
with every optional step MORTIS supports. Use it as a reference to copy
from, not something to run top-to-bottom on every project, skip
whatever doesn't apply to your data.

```python
import mortis as mt

# ── 1. Load ────────────────────────────────────────────────────────────
adatas = mt.load_from_folder("./data")             # auto-pairs tissue + background
adata = mt.read_metabolomics_data("sample_tissue.xlsx")  # or a single file

# ── 2. ROI assignment (if not auto-paired) ───────────────────────────
try:
    mt.check_rois(adatas)
except mt.MissingROIError:
    adatas = mt.draw_ROIs_for_folder(adatas)

# ── 3. Background filtering ────────────────────────────────────────────
clean, stats = mt.filter_background(adatas, cutoff=1.5, mode="sample")
mt.plot_qc(stats[0], clean[0], sample_name="Patient 1", save="qc.pdf")

# ── 4. Annotation score filtering ──────────────────────────────────────
# Facility .xlsx/.csv exports usually do NOT embed annotation scores --
# they arrive in a separate feature table. Merge it in first:
adata = mt.load_annotation_scores(clean[0], "feature_table.xlsx")
adata = mt.filter_by_score(adata, min_score=0.3)
# (.h5ad files exported from METASPACE/SCiLS already have adata.var['score']
# populated, so load_annotation_scores is not needed for those.)

# --- 5. Drug and xenobiotic removal (optional) ---
# This is opt-in QC, not a mandatory stage. If drug distribution is your
# analyte of interest, as in an in-situ PK study, skip it and use
# list_drug_matches() or score_metabolite_set() to find drug ions instead.
# Note that amino acids, taurine and cholesterol all carry drug identifiers,
# so check the endogenous column before removing anything.
matches = mt.list_drug_matches(adata)
print(matches[~matches.endogenous])
adata = mt.filter_drugs(adata)

# ── 6. Preprocess ───────────────────────────────────────────────────────
adata = mt.preprocess(adata, n_pcs=50, n_neighbors=15)
# step by step instead:
# adata = mt.tic_normalize(adata)          # or mt.median_normalize(adata)
# adata = mt.log1p_transform(adata)
# adata = mt.scale(adata, max_value=10.0)
# adata = mt.run_pca(adata, n_comps=50)
# adata = mt.run_neighbors(adata, n_neighbors=15, n_pcs=30)

# ── 7. Batch correction (multi-sample studies) ─────────────────────────
# adata = mt.correct_batches(adata, batch_key="sample")   # ComBat
# adata = mt.run_harmony(adata, batch_key="sample")       # Harmony
# adata = mt.run_neighbors(adata, use_rep="X_pca_harmony")
# adata = mt.batch_mixing_score(adata, batch_key="sample", use_rep="X_pca_harmony")

# ── 8. UMAP ───────────────────────────────────────────────────────────
adata = mt.run_umap(adata, min_dist=0.3)

# ── 9. Clustering ───────────────────────────────────────────────────────
adata = mt.cluster(adata, resolution=0.5)                     # chemistry space
adata = mt.spatial_domains(adata, resolution=0.5, alpha=0.5)  # + spatial context
mt.rename_clusters(adata, mapping={"0": "Tumour core", "1": "Stroma"})

# ── 10. Visualize clusters ──────────────────────────────────────────────
mt.plot_umap(adata, color="cluster", save="umap_clusters.pdf")
mt.plot_spatial(adata, color="cluster", save="spatial_clusters.pdf")
mt.plot_spatial(adata, color="domain", save="spatial_domains.pdf")
mt.plot_cluster_composition(adata, groupby="condition", save="composition.pdf")

# ── 11. Marker metabolites ──────────────────────────────────────────────
adata, markers = mt.find_markers(adata, n_top=20)
mt.plot_markers(adata, n_top=5, save="markers.pdf")
mt.save_results(markers, "results/markers.csv")

# ── 12. Differential expression ─────────────────────────────────────────
adata, results = mt.compare_groups(adata, groupby="condition", group1="healthy", group2="tumour")
mt.plot_volcano(results, group1="healthy", group2="tumour", show_table=True, n_table=5, save="volcano.pdf")
mt.plot_heatmap(adata, results.head(30)["metabolite"].tolist(), groupby="cluster", save="heatmap.pdf")
mt.plot_violin(adata, results.head(5)["metabolite"].tolist(), groupby="condition", save="violin.pdf")

# for >2 groups:
adata, multi_results = mt.multi_group_test(adata, groupby="condition", method="kruskal")

# ── 13. Spatial statistics ──────────────────────────────────────────────
adata, svgs = mt.spatial_de(adata, n_top=50)
mt.plot_morans(svgs, n_top=20, save="morans.pdf")

adata, lisa = mt.local_moran(adata, svgs.iloc[0]["metabolite"])
adata, gi_df = mt.getis_ord_gi(adata, svgs.iloc[0]["metabolite"])

adata = mt.spatial_neighbors(adata, n_neighbors=6)
adata, enrich = mt.neighborhood_enrichment(adata)
co_occ = mt.co_occurrence(adata, cluster_key="cluster")

edges = mt.metabolite_colocalization(adata, top_n=50, corr_threshold=0.4)
mt.plot_colocalization_network(edges, save="interactome.pdf")

# ── 14. Metabolite set scoring, enrichment, diversity ───────────────────
fatty_acids = ["Palmitic acid", "Stearic acid", "Oleic acid"]
adata = mt.score_metabolite_set(adata, fatty_acids, score_name="fatty_acids")
mt.plot_spatial(adata, color="fatty_acids", cmap="RdYlBu_r")

enrichment = mt.metabolite_set_enrichment(results, {"fatty_acid_metabolism": fatty_acids})
lipid_summary = mt.lipid_class_summary(adata, groupby="condition")

adata = mt.diversity_index(adata, method="shannon")
diversity_by_sample = mt.cluster_diversity(adata, cluster_key="cluster", groupby="sample")

# ── 15. Cluster validation ───────────────────────────────────────────────
silhouette = mt.cluster_validation(adata, cluster_key="cluster")
comparison = mt.compare_clusterings(adata.obs["cluster"], adata.obs["domain"])

# ── 16. Multi-sample data management ────────────────────────────────────
merged = mt.merge_samples([adata1, adata2], sample_labels=["S1", "S2"])
patient1 = mt.subset_obs(merged, "sample", "S1")
by_condition = mt.split_by_obs(merged, "condition")

# ── 17. Histology image overlay ─────────────────────────────────────────
adata = mt.load_image(adata, "histology.tif")
adata = mt.align_image(adata, scale_x=0.5, scale_y=0.5, offset_x=10)
adata = mt.extract_image_features(adata, radius=5)
mt.plot_image_overlay(adata, color="cluster", save="overlay.pdf")

# ── 18. Reproducibility receipt (optional) ──────────────────────────────
mt.export_manifest(
    "results/manifest.json",
    adata=adata,
    results={"abundance": ab, "organization": do},
    analysis="my_analysis",
)

# ── 19. Save ──────────────────────────────────────────────────────────────
mt.save_adata(adata, "results/patient1_processed.h5ad")
mt.save_results(results, "results/de_results.csv")
```

Every function used above is documented in full in the
[API Reference](../api/index.md), with parameters, defaults, exceptions,
and its own runnable example.

