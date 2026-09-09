"""
MORTIS
======
Downstream analysis for spatial metabolomics (imaging mass spectrometry),
on top of AnnData and scanpy.

The short version of what it is for: an imaging run gives you tens of
thousands of pixels per section, but a study has a handful of patients. Tests
that treat pixels as replicates will report differences that are not there.
Everything here that makes a claim about a group does it at the sample level.

Typical workflow
----------------
>>> import mortis as mt
>>>
>>> # 1. Load and clean
>>> adatas = mt.load_from_folder("./data")
>>> clean, stats = mt.filter_background(adatas, cutoff=1.5, mode="sample")
>>> adata = mt.merge_samples(clean, sample_col="section")
>>> adata = mt.preprocess(adata)
>>>
>>> # 2. Look at it
>>> adata = mt.cluster(adata)
>>> mt.plot_spatial(adata, color="cluster")
>>>
>>> # 3. Test at the level of the patient, not the pixel
>>> pb = mt.pseudobulk(adata, sample_key="section", carry_obs=["response"])
>>> abundance = mt.differential_abundance(pb, "response", "R", "NR")
>>>
>>> # 4. Then ask whether the arrangement changed, not just the amount
>>> org = mt.spatial_organization(adata, sample_key="section",
...                               carry_obs=["response"])
>>> layout = mt.differential_spatial_organization(org, "response", "R", "NR")
>>>
>>> # 5. Write down what was run, so the numbers can be checked later
>>> mt.export_manifest(adata, "run/manifest.json")
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("mortis-spatial")
except _PackageNotFoundError:  # pragma: no cover - not installed, e.g. running from source
    __version__ = "0.0.0+unknown"

# Analysis
from .analysis import (
    batch_mixing_score,
    cluster,
    cluster_diversity,
    cluster_nmf,
    cluster_validation,
    co_occurrence,
    compare_clusterings,
    compare_groups,
    diversity_index,
    find_markers,
    getis_ord_gi,
    lipid_class_summary,
    local_moran,
    merge_samples,
    metabolite_colocalization,
    metabolite_set_enrichment,
    multi_group_test,
    neighborhood_enrichment,
    rename_clusters,
    run_paga,
    save_adata,
    save_results,
    score_metabolite_set,
    spatial_autocorrelation,
    spatial_de,
    spatial_domains,
    spatial_domains_kmeans,
    spatial_gradient,
    spatial_neighbors,
    spatially_weighted_nmf,
    split_by_obs,
    subset_obs,
    unmix_pixels,
)

# Annotation & enrichment
from .annotate import (
    CHEMICAL_CLASSES,
    class_enrichment,
    classification_report,
    classify_compounds,
    pathway_ora,
)

# Cross-cohort / longitudinal comparison
from .compare import (
    compare_signatures,
    cross_cohort_profile,
    track_flow,
)

# Exceptions
from .exceptions import (
    FileFormatError,
    InsufficientSamplesError,
    InvalidParameterError,
    MissingROIError,
    MissingSpatialError,
    MortisError,
    NoClustersError,
    NoEmbeddingError,
    NotPreprocessedError,
    PseudoreplicationWarning,
)

# Filtering
from .filter import (
    filter_by_score,
    filter_drugs,
    list_drug_matches,
)

# Image (histology overlay)
from .image import (
    align_image,
    extract_image_features,
    load_image,
    plot_image_overlay,
)

# Interactive
from .interactive import (
    draw_ROIs,
    draw_ROIs_for_folder,
)

# I/O
from .io import (
    check_rois,
    load_annotation_scores,
    load_from_folder,
    make_writable,
    read_metabolomics_data,
    save_spatial_data,
)

# Spatial organization (differential spatial pattern)
from .organization import (
    compare_abundance_and_organization,
    differential_spatial_organization,
    spatial_organization,
)

# Compound ID mapping & KEGG pathways (network)
from .pathway import (
    annotate_pathways,
    clear_cache,
    fetch_kegg_pathway_sets,
    map_compound_ids,
)

# Plotting
from .plotting import (
    plot_cluster_composition,
    plot_colocalization_network,
    plot_embedding_grid,
    plot_heatmap,
    plot_markers,
    plot_morans,
    plot_qc,
    plot_spatial,
    plot_spatial_gradient,
    plot_spatial_metabolite,
    plot_umap,
    plot_violin,
    plot_volcano,
)

# Preprocessing
from .preprocessing import (
    correct_batches,
    filter_background,
    log1p_transform,
    median_normalize,
    preprocess,
    run_harmony,
    run_neighbors,
    run_pca,
    run_umap,
    scale,
    tic_normalize,
)

# Reproducibility manifests
from .reproducibility import (
    MANIFEST_VERSION,
    data_fingerprint,
    export_manifest,
    provenance,
    record_step,
    result_fingerprint,
    verify_manifest,
)

# Statistics (sample-level / patient-level)
from .stats import (
    cliffs_delta,
    differential_abundance,
    paired_differential_abundance,
    pseudobulk,
)

# Publication figures & export
from .viz import (
    PALETTE,
    diverging_cmap,
    ion_cmap,
    plot_abundance_vs_organization,
    plot_class_enrichment,
    plot_delta_volcano,
    plot_effect_forest,
    plot_effect_size,
    plot_ion_images,
    plot_organization_heatmap,
    plot_pathway_dotplot,
    plot_signature_comparison,
    reset_style,
    save_figure,
    set_publication_style,
)

__all__ = [
    # Exceptions
    "MortisError",
    "MissingROIError",
    "MissingSpatialError",
    "NotPreprocessedError",
    "NoClustersError",
    "NoEmbeddingError",
    "InsufficientSamplesError",
    "InvalidParameterError",
    "FileFormatError",
    "PseudoreplicationWarning",

    # I/O
    "read_metabolomics_data",
    "make_writable",
    "load_from_folder",
    "load_annotation_scores",
    "check_rois",
    "save_spatial_data",

    # Interactive
    "draw_ROIs",
    "draw_ROIs_for_folder",

    # Preprocessing
    "filter_background",
    "tic_normalize",
    "median_normalize",
    "log1p_transform",
    "scale",
    "run_pca",
    "run_neighbors",
    "run_umap",
    "correct_batches",
    "run_harmony",
    "preprocess",

    # Filtering
    "filter_by_score",
    "filter_drugs",
    "list_drug_matches",

    # Image (histology overlay)
    "load_image",
    "align_image",
    "extract_image_features",
    "plot_image_overlay",

    # Statistics (sample-level)
    "pseudobulk",
    "differential_abundance",
    "paired_differential_abundance",
    "cliffs_delta",

    # Annotation & enrichment
    "classify_compounds",
    "classification_report",
    "class_enrichment",
    "pathway_ora",
    "CHEMICAL_CLASSES",

    # Reproducibility
    "export_manifest",
    "verify_manifest",
    "record_step",
    "provenance",
    "data_fingerprint",
    "result_fingerprint",
    "MANIFEST_VERSION",

    # Compound ID mapping & KEGG pathways
    "map_compound_ids",
    "fetch_kegg_pathway_sets",
    "annotate_pathways",
    "clear_cache",

    # Publication figures & export
    "set_publication_style",
    "reset_style",
    "save_figure",
    "plot_effect_size",
    "plot_effect_forest",
    "plot_delta_volcano",
    "plot_abundance_vs_organization",
    "plot_signature_comparison",
    "plot_ion_images",
    "plot_organization_heatmap",
    "plot_class_enrichment",
    "plot_pathway_dotplot",
    "PALETTE",
    "diverging_cmap",
    "ion_cmap",

    # Cross-cohort comparison
    "cross_cohort_profile",
    "track_flow",
    "compare_signatures",

    # Spatial organization
    "spatial_organization",
    "differential_spatial_organization",
    "compare_abundance_and_organization",

    # Analysis
    "cluster",
    "cluster_nmf",
    "spatial_domains",
    "spatial_domains_kmeans",
    "rename_clusters",
    "find_markers",
    "compare_groups",
    "multi_group_test",
    "spatial_autocorrelation",
    "spatial_de",
    "local_moran",
    "getis_ord_gi",
    "spatial_neighbors",
    "neighborhood_enrichment",
    "co_occurrence",
    "score_metabolite_set",
    "metabolite_set_enrichment",
    "lipid_class_summary",
    "diversity_index",
    "cluster_diversity",
    "unmix_pixels",
    "cluster_validation",
    "compare_clusterings",
    "batch_mixing_score",
    "run_paga",
    "subset_obs",
    "merge_samples",
    "split_by_obs",
    "save_results",
    "save_adata",

    # Spatial structure
    "spatially_weighted_nmf",
    "spatial_gradient",
    "metabolite_colocalization",

    # Plotting
    "plot_qc",
    "plot_spatial",
    "plot_spatial_metabolite",
    "plot_embedding_grid",
    "plot_umap",
    "plot_markers",
    "plot_volcano",
    "plot_heatmap",
    "plot_violin",
    "plot_cluster_composition",
    "plot_morans",

    # Spatial structure plots
    "plot_spatial_gradient",
    "plot_colocalization_network",
]
