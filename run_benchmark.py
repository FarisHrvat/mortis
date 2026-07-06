#!/usr/bin/env python3
"""
MORTIS Full Pipeline Benchmark & Clinical Demonstration Script
==============================================================
Processes multiple spatial metabolomics CSV files from "Responder"
and "Non Responder" cohorts, executing EVERY analysis and plotting
function in the Mortis toolkit with GPU hardware dispatch.

Inputs:
  - Responder/ (Folder containing CSVs)
  - Non Responder/ (Folder containing CSVs)

Outputs:
  - results_benchmark/ (Contains all plots, CSVs, H5ADs, and Audit Trails)
"""

import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # Headless mode for massive batch plotting
import numpy as np
import scanpy as sc

warnings.filterwarnings("ignore")

# ── Import MORTIS ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import mortis as mt  # noqa: E402
from mortis import audit  # noqa: E402

# ── Output directories ─────────────────────────────────────────────────────
OUT_DIR = Path("results_benchmark")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Timing infrastructure ──────────────────────────────────────────────────
timings = {}

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
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

# ===========================================================================
# PART 1: DATA LOADING & MERGING
# ===========================================================================
section("PART 1: DATA INGESTION & COHORT MERGING")

adatas = []
cohorts = ["Responder", "Non Responder"]

with Timer("1.1 Load all CSVs from cohort folders"):
    for condition in cohorts:
        folder = Path(condition)
        if not folder.exists():
            print(f"    ⚠ Warning: Folder '{condition}' not found!")
            continue

        for file in folder.glob("*.csv"):
            adata = mt.read_metabolomics_data(str(file))
            adata.obs["condition"] = condition
            adata.obs["sample"] = file.stem
            # Flag as tissue since data is already background-filtered
            adata.obs["is_tissue"] = True
            adata.obs["is_background"] = False
            adatas.append(adata)
            print(f"      Loaded: {file.name} ({adata.shape[0]} pixels)")

if not adatas:
    print("\n[ERROR] No CSV files found in 'Responder' or 'Non Responder' folders. Exiting.")
    sys.exit(1)

with Timer("1.2 Merge cohorts into integrated dataset"):
    merged = mt.merge_samples(
        adatas,
        sample_labels=[a.obs["sample"].iloc[0] for a in adatas],
        sample_col="sample",
        join="outer"
    )
    exclude_cols = ['x', 'y', 'Patient_ID', 'Group', 'Spatial_Domain', 'condition', 'sample', 'is_tissue', 'is_background']
    clean_vars = [
        col for col in merged.var_names
        if col not in exclude_cols
        and not str(col).isdigit()
        and 'ID' not in str(col)
        and not col.startswith('Unnamed')
        ]
    merged = merged[:, clean_vars].copy()
    import numpy as np
    from scipy.sparse import issparse

        # 3. THE DENSE ZERO-CENTER FIX (Matches IPYNB memory state)
    if issparse(merged.X):
        merged.X = merged.X.toarray()
    merged.X = np.nan_to_num(merged.X, nan=0.0)

    print(f"      Integrated dataset: {merged.shape[0]} pixels × {merged.shape[1]} metabolites")

# ===========================================================================
# PART 2: HARDWARE-ACCELERATED PREPROCESSING
# ===========================================================================

with Timer("2.1 Preprocess (Scale, GPU PCA, CPU kNN)"):
    # ONE LINE: Perfectly matching your IPYNB settings
    merged = mt.preprocess(
        merged,
        n_pcs=50,
        n_neighbors=15,
        metric="euclidean",
        do_tic=False,     # Because your CSVs are already TIC normalized
        do_log1p=False,   # Matches IPYNB
        scale_data=True,  # Matches IPYNB
        max_value=10.0,   # Matches IPYNB
        use_hardware=True,
        svd_solver = "arpack"
    )
with Timer("2.3 UMAP"):
    merged = mt.run_umap(merged, use_hardware=True)

# ===========================================================================
# PART 3: CLUSTERING & DIFFERENTIAL EXPRESSION
# ===========================================================================
section("PART 3: MACHINE LEARNING, CLUSTERING & DE")
print (merged.uns["neighbors"]["params"])
print("Package PCA variance explained:",
      merged.uns["pca"]["variance_ratio"][:5])

with Timer("3.1 Leiden Spatial Clustering"):
    import scanpy as sc

    # Test both flavors on the same graph
    sc.tl.leiden(merged, resolution=0.1, flavor="leidenalg", key_added="leiden_leidenalg")
    sc.tl.leiden(merged, resolution=0.1, flavor="igraph",    key_added="leiden_igraph")

    print("leidenalg:", merged.obs["leiden_leidenalg"].nunique(), "clusters")
    print("igraph:   ", merged.obs["leiden_igraph"].nunique(),    "clusters")

    merged = mt.cluster(merged, flavor="leidenalg", resolution=[0.1])


with Timer("3.2 Hardware-Accelerated NMF Microenvironments"):
    merged, nmf_df = mt.cluster_nmf(merged, n_components=8, use_hardware=True)
    mt.save_results(nmf_df, str(OUT_DIR / "nmf_components.csv"))

with Timer("3.3 Clinical Differential Expression (Responder vs Non Responder)"):
    merged, de_res = mt.compare_groups(
        merged, groupby="condition", group1="Responder", group2="Non Responder"
    )
    mt.save_results(de_res, str(OUT_DIR / "de_responder_vs_nonresponder.csv"))

with Timer("3.4 Cluster Marker Discovery"):
    merged, markers = mt.find_markers(merged, cluster_key="cluster", n_top=20)
    mt.save_results(markers, str(OUT_DIR / "cluster_markers.csv"))

# ===========================================================================
# PART 4: SPATIAL STATISTICS & KILLER FEATURES
# ===========================================================================
section("PART 4: ADVANCED SPATIAL METABOLOMICS FEATURES")

with Timer("4.1 Spatial Autocorrelation (Moran's I)"):
    merged, morans = mt.spatial_autocorrelation(merged, batch_key="sample")
    mt.save_results(morans, str(OUT_DIR / "morans_i.csv"))

with Timer("4.2 Local Moran's I (LISA) for top metabolite"):
    top_met = morans.iloc[0]["metabolite"]
    merged, lisa = mt.local_moran(merged, top_met, batch_key="sample")
    mt.save_results(lisa, str(OUT_DIR / f"lisa_{top_met[:20]}.csv"))

with Timer("4.3 Spatial Connectivities Graph"):
    merged = mt.spatial_neighbors(merged, batch_key="sample")

with Timer("4.4 Numba JIT-Compiled Neighborhood Enrichment"):
    merged, enrich = mt.neighborhood_enrichment(merged, n_permutations=1000)
    mt.save_results(enrich, str(OUT_DIR / "neighborhood_enrichment.csv"))

with Timer("4.5 Spatially-Weighted NMF (GPU Accelerated)"):
    merged, snmf_df = mt.spatially_weighted_nmf(merged, n_components=6, use_hardware=True)
    mt.save_results(snmf_df, str(OUT_DIR / "spatially_weighted_nmf.csv"))

with Timer("4.6 Spatial Gradient Profiling"):
    # Profiling how metabolism changes as we move away from Cluster 0
    grad_df = mt.spatial_gradient(merged, target_col="cluster", target_val="0", max_dist=500.0)
    mt.save_results(grad_df, str(OUT_DIR / "spatial_gradient_profile.csv"))

with Timer("4.7 Metabolite Interactome (Colocalization Network)"):
    interactome_edges = mt.metabolite_colocalization(merged, top_n=50, corr_threshold=0.35)
    mt.save_results(interactome_edges, str(OUT_DIR / "interactome_edges.csv"))

with Timer("4.8 Lipid Class Profiling & Enrichment"):
    lipid_df = mt.lipid_class_summary(merged, groupby="condition")
    mt.save_results(lipid_df, str(OUT_DIR / "lipid_class_summary.csv"))

    # Dummy pathway dictionary to exercise the enrichment function
    top_up = de_res[de_res["log2fc"] > 0.5]["metabolite"].head(15).tolist()
    pathways = {"Responder_Signature": top_up} if len(top_up) >= 3 else {}

    if pathways:
        enrich_res = mt.metabolite_set_enrichment(de_res, pathways, n_permutations=500)
        mt.save_results(enrich_res, str(OUT_DIR / "metabolite_set_enrichment.csv"))

# ===========================================================================
# PART 5: MASS PLOTTING (SILENT / MODIFIABLE)
# ===========================================================================
section("PART 5: GENERATING PUBLICATION PLOTS")

with Timer("5.1 UMAP Plots (Condition, Cluster, NMF)"):
    mt.plot_umap(merged, color="condition", save=str(OUT_DIR / "umap_condition.pdf"), show=False)
    mt.plot_umap(merged, color="cluster", palette="tab20", save=str(OUT_DIR / "umap_cluster.pdf"), show=False)
    mt.plot_umap(merged, color="nmf_cluster", palette="Set2", save=str(OUT_DIR / "umap_nmf.pdf"), show=False)

with Timer("5.2 Spatial Maps (Condition, Cluster, LISA)"):
    # Plotted just for the first sample to keep the map readable
    sub = mt.subset_obs(merged, "sample", merged.obs["sample"].iloc[0])
    mt.plot_spatial(sub, color="condition", s=5, save=str(OUT_DIR / "spatial_condition.pdf"), show=False)
    mt.plot_spatial(sub, color="cluster", s=5, save=str(OUT_DIR / "spatial_cluster.pdf"), show=False)
    mt.plot_spatial(sub, color=f"{top_met}_lisa_type", s=5, save=str(OUT_DIR / "spatial_lisa.pdf"), show=False)

with Timer("5.3 Statistical Plots (Volcano, Moran's I Bar)"):
    mt.plot_volcano(de_res, group1="Responder", group2="Non Responder", fc_cutoff=0.5,
                    show_table=True, save=str(OUT_DIR / "volcano_responder_vs_nonresponder.pdf"), show=False)
    mt.plot_morans(morans, n_top=15, save=str(OUT_DIR / "morans_i_top15.pdf"), show=False)

with Timer("5.4 Marker Profiling (Dotplot, Heatmap, Violin)"):
    top_markers = markers["metabolite"].head(20).tolist()
    mt.plot_markers(merged, n_top=5, save=str(OUT_DIR / "marker_dotplot.pdf"), show=False)
    mt.plot_heatmap(merged, top_markers, groupby="condition", save=str(OUT_DIR / "marker_heatmap.pdf"), show=False)
    mt.plot_violin(merged, top_markers[:3], groupby="cluster", save=str(OUT_DIR / "marker_violin.pdf"), show=False)

with Timer("5.5 Structural Composition & Embedded Grids"):
    mt.plot_cluster_composition(merged, groupby="condition", normalize=True, save=str(OUT_DIR / "cluster_composition.pdf"), show=False)
    if len(morans) >= 4:
        mt.plot_embedding_grid(sub, morans["metabolite"].head(4).tolist(), ncols=2, save=str(OUT_DIR / "spatial_top_mets_grid.pdf"), show=False)

with Timer("5.6 Killer Feature Plots (Gradient & Interactome)"):
    mt.plot_spatial_gradient(grad_df, top_n=4, save=str(OUT_DIR / "spatial_gradient.pdf"), show=False)
    if not interactome_edges.empty:
        mt.plot_colocalization_network(interactome_edges, save=str(OUT_DIR / "interactome_network.pdf"), show=False)

# ===========================================================================
# PART 6: CRYPTOGRAPHIC AUDIT & EXPORT
# ===========================================================================
section("PART 6: SECURE AUDIT & FINAL EXPORT")

with Timer("6.1 Generate Cryptographic Audit Receipt"):
    params_logged = {
        "n_samples": len(adatas),
        "cohorts": cohorts,
        "n_pixels": merged.shape[0],
        "n_metabolites": merged.shape[1],
        "hardware_acceleration_requested": True
    }
    audit.generate_audit_receipt("full_benchmark_pipeline", params_logged, output_dir=str(OUT_DIR))

with Timer("6.2 Save integrated H5AD matrix"):
    mt.save_adata(merged, str(OUT_DIR / "merged_processed_dataset.h5ad"))


# ===========================================================================
# PART 7: TIMING REPORT
# ===========================================================================
section("MORTIS BENCHMARK TIMING REPORT")

sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)

print(f"Total functions executed: {len(timings)}")
print(f"Total wall time: {sum(timings.values()):.2f} seconds\n")
print("Execution Times (Slowest to Fastest):")
print("-" * 70)

for label, t in sorted_timings:
    bar = "█" * min(int(t * 2), 40)
    print(f"  {t:7.3f}s  {bar}  {label}")

print("\n✓ Benchmark complete. All files, plots, and cryptographic signatures are in 'results_benchmark/'")
