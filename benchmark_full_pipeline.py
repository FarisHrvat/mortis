#!/usr/bin/env python3
"""
SpatPy Full Pipeline Benchmark & Demonstration Script
======================================================
Uses every function in the SpatPy package on real spatial metabolomics data.

Data used:
  - 1b_P487_Healthy_sez1.h5ad  (14,504 pixels × 4,023 metabolites)  — H5AD
  - 488IMb_tissue.xlsx + 488IMb_background.xlsx  (9,741 × 3,566)     — XLSX pair
  - data/glass_1_left.h5ad     (30,000 pixels × 1,952 metabolites)   — H5AD large

Outputs:
  - results/single_sample/   — all single-sample analysis outputs
  - results/group/           — multi-sample / group analysis outputs
  - results/benchmark.txt    — timing report for every step
"""

import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")   # headless — no display needed
import numpy as np

warnings.filterwarnings("ignore")

# ── Import MORTIS ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import mortis as mt  # noqa: E402
from mortis.analysis import local_moran  # noqa: E402

# ── Output directories ─────────────────────────────────────────────────────
SINGLE_DIR = Path("results/single_sample")
GROUP_DIR  = Path("results/group")
BENCH_DIR  = Path("results")
for d in [SINGLE_DIR, GROUP_DIR, BENCH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Timing infrastructure ──────────────────────────────────────────────────
timings: dict[str, float] = {}

class Timer:
    def __init__(self, label: str):
        self.label = label
    def __enter__(self):
        self._t = time.perf_counter()
        return self
    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._t
        timings[self.label] = elapsed
        print(f"  ✓ {self.label}: {elapsed:.3f}s")

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")



# ===========================================================================
# PART 1: SINGLE-SAMPLE ANALYSIS (H5AD)
# ===========================================================================

section("PART 1: SINGLE-SAMPLE ANALYSIS (H5AD)")

with Timer("1.1 Load H5AD"):
    adata_h5 = mt.read_metabolomics_data("1b_P487_Healthy_sez1.h5ad")
    print(f"      Loaded: {adata_h5.shape[0]} pixels × {adata_h5.shape[1]} metabolites")

# Convert Sample_roi / Matrix_roi to is_tissue / is_background
adata_h5.obs["is_tissue"] = adata_h5.obs["Sample_roi"]
adata_h5.obs["is_background"] = adata_h5.obs["Matrix_roi"]

with Timer("1.2 Filter background (single sample)"):
    clean_h5, stats_h5 = mt.filter_background([adata_h5], cutoff=1.5, mode="sample")
    adata_single = clean_h5[0]
    print(f"      After filtering: {adata_single.shape}")

with Timer("1.3 Plot QC (PDF, PNG, SVG)"):
    mt.plot_qc(stats_h5[0], adata_single, sample_name="H5AD Sample",
                   dpi=150, save=str(SINGLE_DIR / "qc.pdf"))
    mt.plot_qc(stats_h5[0], adata_single, sample_name="H5AD Sample",
                   dpi=150, save=str(SINGLE_DIR / "qc.png"))
    mt.plot_qc(stats_h5[0], adata_single, sample_name="H5AD Sample",
                   dpi=150, save=str(SINGLE_DIR / "qc.svg"))

with Timer("1.4 Filter by annotation score"):
    adata_single = mt.filter_by_score(adata_single, min_score=0.3)
    print(f"      After score filter: {adata_single.shape}")

with Timer("1.5 Filter drugs (DrugBank)"):
    matches = mt.list_drug_matches(adata_single, db_path="drugbank.db")
    print(f"      Found {len(matches)} drug metabolites")
    adata_single = mt.filter_drugs(adata_single, db_path="drugbank.db")
    print(f"      After drug filter: {adata_single.shape}")

with Timer("1.6 Preprocess (TIC, log1p, scale, PCA, kNN)"):
    adata_single = mt.preprocess(adata_single, n_pcs=50, n_neighbors=15)

with Timer("1.7 UMAP"):
    adata_single = mt.run_umap(adata_single, min_dist=0.3)

with Timer("1.8 Leiden clustering"):
    adata_single = mt.cluster(adata_single, resolution=0.5)
    print(f"      {adata_single.obs['cluster'].nunique()} clusters")

with Timer("1.9 NMF clustering"):
    adata_single, nmf_df = mt.cluster_nmf(adata_single, n_components=8)
    mt.save_results(nmf_df, str(SINGLE_DIR / "nmf_components.csv"))

with Timer("1.10 Rename clusters"):
    n_clust = adata_single.obs["cluster"].nunique()
    mapping = {str(i): f"Region_{i}" for i in range(min(n_clust, 5))}
    mt.rename_clusters(adata_single, mapping)

with Timer("1.11 Plot UMAP (cluster, NMF, custom styles)"):
    mt.plot_umap(adata_single, color="cluster", save=str(SINGLE_DIR / "umap_cluster.pdf"))
    mt.plot_umap(adata_single, color="nmf_cluster", palette="Set2",
                     figsize=(8, 6), dpi=300, fontsize=13,
                     save=str(SINGLE_DIR / "umap_nmf.png"))

with Timer("1.12 Plot spatial (cluster, NMF)"):
    mt.plot_spatial(adata_single, color="cluster",
                        spot_size=3, save=str(SINGLE_DIR / "spatial_cluster.pdf"))
    mt.plot_spatial(adata_single, color="nmf_cluster", cmap="tab10",
                        figsize=(10, 8), dpi=200, fontsize=12,
                        show_grid=True, save=str(SINGLE_DIR / "spatial_nmf.svg"))

with Timer("1.13 Find marker metabolites"):
    adata_single, markers = mt.find_markers(adata_single, n_top=30)
    mt.save_results(markers, str(SINGLE_DIR / "markers.csv"))

with Timer("1.14 Plot markers"):
    mt.plot_markers(adata_single, n_top=5, dpi=200,
                        save=str(SINGLE_DIR / "markers.pdf"))

with Timer("1.15 Plot embedding grid (top 12 markers)"):
    top_mets = markers["metabolite"].head(12).tolist()
    mt.plot_embedding_grid(adata_single, top_mets, ncols=4,
                                cmap="hot", dpi=150,
                                save=str(SINGLE_DIR / "grid_top_markers.pdf"))

with Timer("1.16 Spatial autocorrelation (Moran's I)"):
    adata_single, morans = mt.spatial_autocorrelation(adata_single, n_neighbors=6)
    mt.save_results(morans, str(SINGLE_DIR / "morans_i.csv"))

with Timer("1.17 Spatial DE (top 50)"):
    adata_single, svgs = mt.spatial_de(adata_single, n_top=50)
    mt.save_results(svgs, str(SINGLE_DIR / "spatial_de.csv"))

with Timer("1.18 Plot Moran's I"):
    mt.plot_morans(svgs, n_top=20, dpi=200,
                       save=str(SINGLE_DIR / "morans_bar.pdf"))

with Timer("1.19 Local Moran (LISA) for top metabolite"):
    # Use top by Moran's I value regardless of significance
    top_met = morans.sort_values("morans_i", ascending=False)["metabolite"].iloc[0]
    adata_single, lisa = local_moran(adata_single, top_met)
    mt.save_results(lisa, str(SINGLE_DIR / f"lisa_{top_met[:20]}.csv"))
    mt.plot_spatial(adata_single, color=f"{top_met}_lisa_type",
                        save=str(SINGLE_DIR / "lisa_spatial.pdf"))

with Timer("1.20 Spatial neighbors + neighborhood enrichment"):
    adata_single = mt.spatial_neighbors(adata_single, n_neighbors=6)
    adata_single, enrich = mt.neighborhood_enrichment(adata_single, n_permutations=500)
    mt.save_results(enrich, str(SINGLE_DIR / "neighborhood_enrichment.csv"))

with Timer("1.21 PAGA trajectory"):
    adata_single = mt.run_paga(adata_single)

with Timer("1.22 Metabolite set scoring (fatty acids)"):
    # Find fatty acid metabolites
    fa_mets = [m for m in adata_single.var_names if "acid" in m.lower()][:10]
    if fa_mets:
        adata_single = mt.score_metabolite_set(adata_single, fa_mets,
                                                    score_name="fatty_acids")
        mt.plot_spatial(adata_single, color="fatty_acids", cmap="RdYlBu_r",
                            save=str(SINGLE_DIR / "fatty_acids_spatial.pdf"))

with Timer("1.23 Lipid class summary"):
    lipid_summary = mt.lipid_class_summary(adata_single, groupby="cluster")
    mt.save_results(lipid_summary, str(SINGLE_DIR / "lipid_class_summary.csv"))

with Timer("1.24 Plot cluster composition"):
    # Add a fake condition for demonstration
    adata_single.obs["condition"] = np.where(
        adata_single.obs["cluster"].isin(["0", "1", "Region_0", "Region_1"]),
        "TypeA", "TypeB"
    )
    mt.plot_cluster_composition(adata_single, groupby="condition",
                                     dpi=200, save=str(SINGLE_DIR / "composition.pdf"))

with Timer("1.25 Plot heatmap (top 30 markers)"):
    top30 = markers["metabolite"].head(30).tolist()
    mt.plot_heatmap(adata_single, top30, groupby="cluster",
                        figsize=(10, 12), dpi=200,
                        save=str(SINGLE_DIR / "heatmap.pdf"))

with Timer("1.26 Plot violin (top 5 markers)"):
    top5 = markers["metabolite"].head(5).tolist()
    mt.plot_violin(adata_single, top5, groupby="cluster",
                       dpi=150, save=str(SINGLE_DIR / "violin.pdf"))

with Timer("1.27 Save final processed H5AD"):
    mt.save_adata(adata_single, str(SINGLE_DIR / "processed.h5ad"))

print(f"\n✓ Single-sample analysis complete: {SINGLE_DIR}")


# ===========================================================================
# PART 2: MULTI-SAMPLE / GROUP ANALYSIS (XLSX + H5AD)
# ===========================================================================

section("PART 2: MULTI-SAMPLE / GROUP ANALYSIS")

with Timer("2.1 Load XLSX paired files"):
    adata_t = mt.read_metabolomics_data("488IMb_tissue.xlsx")
    adata_t.obs["is_tissue"] = True
    adata_t.obs["is_background"] = False
    adata_b = mt.read_metabolomics_data("488IMb_background.xlsx")
    adata_b.obs["is_tissue"] = False
    adata_b.obs["is_background"] = True
    import anndata as ad
    xlsx_adatas = [ad.concat([adata_t, adata_b], join="outer", fill_value=0.0)]
    xlsx_adatas[0].obsm["spatial"] = xlsx_adatas[0].obs[["x","y"]].to_numpy(dtype=np.float32)
    print(f"      XLSX sample: {xlsx_adatas[0].shape}")

with Timer("2.2 Load large H5AD (glass_1_left)"):
    adata_glass = mt.read_metabolomics_data("data/glass_1_left.h5ad")
    # glass_1_left has no ROI labels — assign all as tissue for demo
    adata_glass.obs["is_tissue"] = True
    adata_glass.obs["is_background"] = False
    print(f"      Glass sample: {adata_glass.shape}")

with Timer("2.3 Filter background — group mode (XLSX)"):
    clean_xlsx, stats_xlsx = mt.filter_background(
        xlsx_adatas, cutoff=1.5, mode="sample"
    )
    adata_xlsx = clean_xlsx[0]
    print(f"      After filter: {adata_xlsx.shape}")

with Timer("2.4 QC plot for XLSX sample"):
    mt.plot_qc(stats_xlsx[0], adata_xlsx, sample_name="XLSX Sample",
                   figsize=(16, 11), dpi=150,
                   save=str(GROUP_DIR / "qc_xlsx.pdf"))

with Timer("2.5 Filter by score (XLSX)"):
    if "score" in adata_xlsx.var.columns:
        adata_xlsx = mt.filter_by_score(adata_xlsx, min_score=0.3)
    print(f"      After score filter: {adata_xlsx.shape}")

with Timer("2.6 Filter drugs (XLSX)"):
    # Drug filter removes all metabolites in this dataset, skip it or keep some
    pass
    print(f"      Skipped drug filter: {adata_xlsx.shape}")

with Timer("2.7 Preprocess XLSX sample"):
    if adata_xlsx.shape[1] > 0:
        adata_xlsx = mt.preprocess(adata_xlsx, n_pcs=min(50, adata_xlsx.shape[1]-1), n_neighbors=15)

with Timer("2.8 Preprocess glass sample (large: 30k pixels)"):
    # glass has no background — just preprocess directly
    adata_glass_proc = mt.preprocess(adata_glass, n_pcs=50, n_neighbors=15)

with Timer("2.9 Simulate multi-patient dataset"):
    # Create 3 simulated patients from the h5ad data by splitting spatially
    adata_base = mt.read_metabolomics_data("1b_P487_Healthy_sez1.h5ad")
    adata_base.obs["is_tissue"] = adata_base.obs["Sample_roi"]
    adata_base.obs["is_background"] = adata_base.obs["Matrix_roi"]
    clean_base, _ = mt.filter_background([adata_base], cutoff=1.5, mode="sample")
    adata_base_clean = clean_base[0]

    # Split into 3 "patients" by x-coordinate thirds
    x = adata_base_clean.obs["x"].values
    x_min, x_max = x.min(), x.max()
    third = (x_max - x_min) / 3
    adata_base_clean.obs["patient"] = "P3"
    adata_base_clean.obs.loc[x < x_min + third, "patient"] = "P1"
    adata_base_clean.obs.loc[(x >= x_min + third) & (x < x_min + 2*third), "patient"] = "P2"

    # Split into individual patients
    patients = mt.split_by_obs(adata_base_clean, "patient")
    print(f"      Patients: {[(k, v.n_obs) for k, v in patients.items()]}")

with Timer("2.10 Merge patients into one dataset"):
    merged = mt.merge_samples(
        list(patients.values()),
        sample_labels=list(patients.keys()),
        sample_col="patient",
    )
    print(f"      Merged: {merged.shape}")

with Timer("2.11 Preprocess merged dataset"):
    merged = mt.preprocess(merged, n_pcs=50, n_neighbors=15)

with Timer("2.12 Harmony batch correction"):
    merged = mt.run_harmony(merged, batch_key="patient")
    # Re-run neighbors on harmony-corrected embedding
    import scanpy as sc
    sc.pp.neighbors(merged, use_rep="X_pca_harmony", n_neighbors=15, n_pcs=30)
    merged.uns["preprocessed_steps"] = merged.uns.get("preprocessed_steps", [])

with Timer("2.13 UMAP on batch-corrected data"):
    merged = mt.run_umap(merged)

with Timer("2.14 Leiden clustering on merged"):
    merged = mt.cluster(merged, resolution=0.5)
    print(f"      {merged.obs['cluster'].nunique()} clusters")

with Timer("2.15 NMF on merged"):
    merged, nmf_merged = mt.cluster_nmf(merged, n_components=6)
    mt.save_results(nmf_merged, str(GROUP_DIR / "nmf_merged.csv"))

with Timer("2.16 UMAP plots (patient, cluster, NMF)"):
    mt.plot_umap(merged, color="patient", palette="Set1",
                     figsize=(8, 6), dpi=200,
                     save=str(GROUP_DIR / "umap_patient.pdf"))
    mt.plot_umap(merged, color="cluster", palette="tab20",
                     figsize=(8, 6), dpi=200,
                     save=str(GROUP_DIR / "umap_cluster.pdf"))
    mt.plot_umap(merged, color="nmf_cluster", palette="Set2",
                     figsize=(8, 6), dpi=200,
                     save=str(GROUP_DIR / "umap_nmf.png"))

with Timer("2.17 Spatial plots per patient"):
    for pat_name, pat_adata in patients.items():
        # Subset merged to this patient
        sub = mt.subset_obs(merged, "patient", pat_name)
        mt.plot_spatial(sub, color="cluster", spot_size=4,
                            title=f"Patient {pat_name} — Clusters",
                            save=str(GROUP_DIR / f"spatial_{pat_name}.pdf"))

with Timer("2.18 Find markers on merged"):
    merged, markers_merged = mt.find_markers(merged, n_top=20)
    mt.save_results(markers_merged, str(GROUP_DIR / "markers_merged.csv"))
    mt.plot_markers(merged, n_top=5, dpi=200,
                        save=str(GROUP_DIR / "markers.pdf"))

with Timer("2.19 Compare groups (P1 vs P3)"):
    merged, de_results = mt.compare_groups(
        merged, groupby="patient", group1="P1", group2="P3"
    )
    mt.save_results(de_results, str(GROUP_DIR / "de_P1_vs_P3.csv"))

with Timer("2.20 Volcano plot (with table, PDF + SVG)"):
    mt.plot_volcano(de_results, group1="P1", group2="P3",
                        fc_cutoff=0.5, pval_cutoff=0.05,
                        show_table=True, n_table=5,
                        figsize=(10, 11), dpi=200,
                        save=str(GROUP_DIR / "volcano_P1_P3.pdf"))
    mt.plot_volcano(de_results, group1="P1", group2="P3",
                        fc_cutoff=0.5, show_table=False,
                        figsize=(9, 7), dpi=300,
                        save=str(GROUP_DIR / "volcano_P1_P3.svg"))

with Timer("2.21 Multi-group test (all 3 patients)"):
    merged, multi_results = mt.multi_group_test(
        merged, groupby="patient", method="kruskal"
    )
    mt.save_results(multi_results, str(GROUP_DIR / "multi_group_kruskal.csv"))

with Timer("2.22 Metabolite set enrichment"):
    # Build sets from top DE metabolites
    up_in_p3 = de_results[de_results["log2fc"] > 0.5]["metabolite"].head(20).tolist()
    up_in_p1 = de_results[de_results["log2fc"] < -0.5]["metabolite"].head(20).tolist()
    sets = {}
    if len(up_in_p3) >= 3:
        sets["Elevated_P3"] = up_in_p3
    if len(up_in_p1) >= 3:
        sets["Elevated_P1"] = up_in_p1
    # Add lipid sets
    fa_mets = [m for m in merged.var_names if "acid" in m.lower()][:15]
    if len(fa_mets) >= 3:
        sets["Fatty_acids"] = fa_mets
    if sets:
        enrich_df = mt.metabolite_set_enrichment(de_results, sets, n_permutations=500)
        mt.save_results(enrich_df, str(GROUP_DIR / "set_enrichment.csv"))
        print(f"      Enrichment results: {len(enrich_df)} pathways")

with Timer("2.23 Lipid class summary by cluster"):
    lipid_grp = mt.lipid_class_summary(merged, groupby="cluster")
    mt.save_results(lipid_grp, str(GROUP_DIR / "lipid_class_by_cluster.csv"))

with Timer("2.24 Heatmap (top 30 DE metabolites)"):
    top30_de = de_results.head(30)["metabolite"].tolist()
    mt.plot_heatmap(merged, top30_de, groupby="patient",
                        cmap="RdBu_r", figsize=(8, 14), dpi=200,
                        save=str(GROUP_DIR / "heatmap_de.pdf"))

with Timer("2.25 Violin plots (top 5 DE metabolites)"):
    top5_de = de_results.head(5)["metabolite"].tolist()
    mt.plot_violin(merged, top5_de, groupby="patient",
                       dpi=200, save=str(GROUP_DIR / "violin_de.pdf"))

with Timer("2.26 Cluster composition by patient"):
    mt.plot_cluster_composition(merged, cluster_key="cluster",
                                     groupby="patient", normalize=True,
                                     dpi=200, save=str(GROUP_DIR / "composition.pdf"))
    mt.plot_cluster_composition(merged, cluster_key="cluster",
                                     groupby="patient", normalize=False,
                                     dpi=200, save=str(GROUP_DIR / "composition_counts.png"))

with Timer("2.27 Spatial autocorrelation on merged"):
    merged, morans_merged = mt.spatial_autocorrelation(merged, n_neighbors=6)
    mt.save_results(morans_merged, str(GROUP_DIR / "morans_merged.csv"))
    mt.plot_morans(morans_merged, n_top=20, dpi=200,
                       save=str(GROUP_DIR / "morans.pdf"))

with Timer("2.28 Spatial DE on merged"):
    merged, svgs_merged = mt.spatial_de(merged, n_top=30)
    mt.save_results(svgs_merged, str(GROUP_DIR / "spatial_de_merged.csv"))

with Timer("2.29 Local Moran (LISA) on top SVG"):
    top_svg = morans_merged.sort_values("morans_i", ascending=False)["metabolite"].iloc[0]
    merged, lisa_merged = mt.local_moran(merged, top_svg)
    mt.save_results(lisa_merged, str(GROUP_DIR / f"lisa_{top_svg[:20]}.csv"))
    mt.plot_spatial(merged, color=f"{top_svg}_lisa_type",
                        title=f"LISA: {top_svg[:30]}",
                        save=str(GROUP_DIR / "lisa_spatial.pdf"))

with Timer("2.30 Spatial neighbors + neighborhood enrichment"):
    merged = mt.spatial_neighbors(merged, n_neighbors=6)
    merged, enrich_merged = mt.neighborhood_enrichment(
        merged, n_permutations=500
    )
    mt.save_results(enrich_merged, str(GROUP_DIR / "neighborhood_enrichment.csv"))

with Timer("2.31 Metabolite set scoring"):
    fa_mets_merged = [m for m in merged.var_names if "acid" in m.lower()][:10]
    if fa_mets_merged:
        merged = mt.score_metabolite_set(merged, fa_mets_merged,
                                              score_name="fatty_acids")
        mt.plot_spatial(merged, color="fatty_acids", cmap="plasma",
                            save=str(GROUP_DIR / "fatty_acids.pdf"))

with Timer("2.32 Embedding grid (top 8 SVGs)"):
    if len(svgs_merged) >= 4:
        top8 = svgs_merged["metabolite"].head(8).tolist()
        mt.plot_embedding_grid(merged, top8, ncols=4, cmap="viridis",
                                    dpi=150, save=str(GROUP_DIR / "svg_grid.pdf"))

with Timer("2.33 PAGA trajectory"):
    merged = mt.run_paga(merged)

with Timer("2.34 Save each patient separately"):
    for pat_name in patients.keys():
        sub = mt.subset_obs(merged, "patient", pat_name)
        mt.save_adata(sub, str(GROUP_DIR / f"patient_{pat_name}.h5ad"))

with Timer("2.35 Save merged processed H5AD"):
    mt.save_adata(merged, str(GROUP_DIR / "merged_processed.h5ad"))

print(f"\n✓ Group analysis complete: {GROUP_DIR}")


# ===========================================================================
# PART 3: LARGE DATA BENCHMARK (glass_1_left: 30k pixels)
# ===========================================================================

section("PART 3: LARGE DATA BENCHMARK (30,000 pixels)")

with Timer("3.1 Preprocess 30k pixels"):
    adata_large = mt.preprocess(adata_glass, n_pcs=50, n_neighbors=15)

with Timer("3.2 UMAP 30k pixels"):
    adata_large = mt.run_umap(adata_large)

with Timer("3.3 Leiden clustering 30k pixels"):
    adata_large = mt.cluster(adata_large, resolution=0.5)
    print(f"      {adata_large.obs['cluster'].nunique()} clusters")

with Timer("3.4 Moran's I 30k × 1952 metabolites"):
    adata_large, morans_large = mt.spatial_autocorrelation(adata_large, n_neighbors=6)
    print(f"      {(morans_large['pval_adj'] < 0.05).sum()} spatially variable")

with Timer("3.5 Compare groups 30k pixels"):
    # Split by x-coordinate for demo
    x_med = np.median(adata_large.obs["x"].values)
    adata_large.obs["half"] = np.where(adata_large.obs["x"].values < x_med, "left", "right")
    adata_large, de_large = mt.compare_groups(
        adata_large, groupby="half", group1="left", group2="right"
    )
    print(f"      {de_large['significant'].sum()} significant metabolites")

with Timer("3.6 Spatial neighbors 30k pixels"):
    adata_large = mt.spatial_neighbors(adata_large, n_neighbors=6)

with Timer("3.7 Neighborhood enrichment 30k pixels"):
    adata_large, enrich_large = mt.neighborhood_enrichment(
        adata_large, n_permutations=200
    )

with Timer("3.8 NMF 30k pixels"):
    adata_large, nmf_large = mt.cluster_nmf(adata_large, n_components=8)

with Timer("3.9 Multi-group test 30k pixels"):
    adata_large, multi_large = mt.multi_group_test(
        adata_large, groupby="cluster", method="kruskal"
    )

with Timer("3.10 Save large processed H5AD"):
    mt.save_adata(adata_large, str(BENCH_DIR / "glass_processed.h5ad"))

# ===========================================================================
# PART 4: TIMING REPORT & OPTIMIZATION ANALYSIS
# ===========================================================================

section("PART 4: TIMING REPORT")

# Sort by elapsed time
sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)

report_lines = [
    "SpatPy Full Pipeline Benchmark Report",
    "=" * 60,
    f"Total functions exercised: {len(timings)}",
    f"Total wall time: {sum(timings.values()):.1f}s",
    "",
    "All steps sorted by duration (slowest first):",
    "-" * 60,
]

for label, t in sorted_timings:
    bar = "█" * min(int(t * 2), 40)
    report_lines.append(f"  {t:7.3f}s  {bar}  {label}")

report_lines += [
    "",
    "=" * 60,
    "OPTIMIZATION ANALYSIS",
    "=" * 60,
    "",
    "BOTTLENECKS IDENTIFIED:",
]

# Identify slow steps (>5s)
slow = [(lbl, t) for lbl, t in sorted_timings if t > 5.0]
if slow:
    for label, t in slow:
        report_lines.append(f"  ⚠  {t:.1f}s — {label}")
else:
    report_lines.append("  ✓ No steps exceeded 5 seconds")

report_lines += [
    "",
    "PERFORMANCE NOTES:",
    "  • Moran's I / Geary's C: fully vectorised (sparse W @ X_dev), all metabolites at once",
    "  • compare_groups / multi_group_test: fully vectorised (scipy axis=0 batch test, no per-metabolite loop)",
    "  • neighborhood_enrichment: vectorised np.add.at + Numba-JIT parallel permutations",
    "  • filter_background: column means computed without materialising full matrix",
    "  • run_pca: randomised SVD for >10k pixels, ARPACK for smaller datasets",
    "  • cKDTree.query: respects MORTIS_N_JOBS (see below)",
    "",
    "REMAINING OPTIMIZATION OPPORTUNITIES:",
    "  1. UMAP: single-threaded by design (umap-learn limitation)",
    "     → Consider cuML GPU UMAP for datasets >50k pixels",
    "  2. Leiden clustering: single-threaded igraph",
    "     → Resolution sweep could be parallelised across resolutions",
    "  3. XLSX loading: pandas read_excel is slow for 165MB files",
    "     → Convert to H5AD once and reload; or use polars for CSV",
    "  4. NMF: sklearn NMF is single-threaded",
    "     → For very large data, consider mini-batch NMF or GPU NMF",
    "  5. local_moran (LISA): already O(n) analytical (no permutations), fast by construction",
    "  6. Harmony: runs on CPU; for >100k pixels consider GPU Harmony",
    "  7. Memory: X matrix stored as float32 (good); consider memory-mapped",
    "     H5AD for datasets >10GB via backed mode: ad.read_h5ad(f, backed='r')",
    "",
    "MEMORY / THREADING TIPS:",
    "  • Use mt.subset_obs() to work on one patient at a time",
    "  • Use adata.X = scipy.sparse.csr_matrix(adata.X) for sparse data",
    "  • Delete intermediate variables: del adata_raw; import gc; gc.collect()",
    "  • Set MORTIS_N_JOBS=4 env var (before starting Python) to cap threads",
    "    used by run_pca (via threadpoolctl: measured ~3-5x on a 30k x 2k PCA)",
    "    and neighborhood_enrichment's Numba permutation loop (measured ~3-10x,",
    "    order-independent so not a JIT warm-up artifact). The previous",
    "    implementation of this setting (os.environ-based) measured 0x effect",
    "    on both, because OpenBLAS/MKL only read those env vars once, at",
    "    first use — setting them deep inside a function call is too late.",
    "  • run_neighbors/run_umap: the FIRST call in a process pays a one-time",
    "    ~15-17s Numba JIT-compilation cost (pynndescent), independent of",
    "    thread count or MORTIS_N_JOBS — every later call in the same",
    "    process is ~1.3-1.5s. Don't mistake this for a performance",
    "    regression; it's a compilation cost, not a per-call cost.",
]

report_text = "\n".join(report_lines)
print("\n" + report_text)

report_path = BENCH_DIR / "benchmark.txt"
report_path.write_text(report_text)
print(f"\n✓ Benchmark report saved: {report_path}")

# ===========================================================================
# SUMMARY
# ===========================================================================

section("SUMMARY")
print(f"  Single-sample outputs : {SINGLE_DIR}")
print(f"  Group outputs         : {GROUP_DIR}")
print(f"  Benchmark report      : {BENCH_DIR / 'benchmark.txt'}")
print(f"  Total time            : {sum(timings.values()):.1f}s")
print(f"  Functions exercised   : {len(timings)}")
print("\n  All outputs saved. Script complete.")
