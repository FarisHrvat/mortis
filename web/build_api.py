#!/usr/bin/env python3
"""
Generate the API index the docs page reads.

Walks the installed package and writes ``web/api.json``: every public function,
grouped by pipeline stage, with its signature, its parameters and the first
paragraph of its docstring.

Reading the installed package rather than parsing source means the reference
cannot drift from the code. The page shows whatever is actually importable, and
CI regenerates this on every deploy, so a renamed argument shows up in the docs
the same day it lands.

The one thing that is maintained by hand is ``GROUPS`` below, the stage each
function belongs to, because "which part of the pipeline is this" is editorial
and not something a signature knows. Anything exported but missing from a group
is reported loudly rather than dropped, so new functions cannot quietly go
undocumented.

    python web/build_api.py
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # importing mortis pulls in matplotlib; keep it headless

import mortis as mt  # noqa: E402

#: (stage, one-line description, function names). Order is pipeline order,
#: which is also roughly the order a new user meets them in.
GROUPS: List[tuple] = [
    ("Load", "Reading instrument exports and ROIs into AnnData.",
     ["read_metabolomics_data", "load_from_folder", "load_annotation_scores", "check_rois",
      "save_spatial_data", "make_writable", "draw_ROIs", "draw_ROIs_for_folder"]),
    ("Clean", "Removing background, low-confidence annotations and drug compounds.",
     ["filter_background", "filter_by_score", "filter_drugs", "list_drug_matches"]),
    ("Normalise", "Per-pixel normalisation, transformation and embedding.",
     ["tic_normalize", "median_normalize", "log1p_transform", "scale", "run_pca",
      "run_neighbors", "run_umap", "preprocess", "correct_batches", "run_harmony",
      "batch_mixing_score"]),
    ("Segment", "Clustering pixels into chemical clusters and spatial domains.",
     ["cluster", "cluster_nmf", "spatial_domains", "spatial_domains_kmeans", "rename_clusters",
      "cluster_validation", "compare_clusterings", "spatially_weighted_nmf", "unmix_pixels",
      "run_paga"]),
    ("Test", "Sample-level statistics. Pixels are not replicates.",
     ["pseudobulk", "differential_abundance", "paired_differential_abundance", "cliffs_delta",
      "find_markers", "compare_groups", "multi_group_test"]),
    ("Space", "Where things are, not just how much.",
     ["spatial_autocorrelation", "spatial_de", "local_moran", "getis_ord_gi", "spatial_neighbors",
      "neighborhood_enrichment", "co_occurrence", "spatial_gradient", "metabolite_colocalization",
      "diversity_index", "cluster_diversity"]),
    ("Organise", "Differential spatial organization: same amount, different arrangement.",
     ["spatial_organization", "differential_spatial_organization",
      "compare_abundance_and_organization"]),
    ("Compare", "Across cohorts and across time.",
     ["cross_cohort_profile", "track_flow", "compare_signatures"]),
    ("Annotate", "Chemical class, enrichment and KEGG pathways.",
     ["classify_compounds", "classification_report", "class_enrichment", "pathway_ora",
      "map_compound_ids", "fetch_kegg_pathway_sets", "annotate_pathways", "clear_cache",
      "score_metabolite_set", "metabolite_set_enrichment", "lipid_class_summary"]),
    ("Verify", "Manifests a reviewer can check a re-run against.",
     ["export_manifest", "verify_manifest", "record_step", "provenance",
      "data_fingerprint", "result_fingerprint"]),
    ("Plot", "Publication figures with editable-vector PDF export.",
     ["set_publication_style", "reset_style", "save_figure", "plot_ion_images", "plot_effect_size",
      "plot_delta_volcano", "plot_abundance_vs_organization", "plot_signature_comparison",
      "plot_organization_heatmap", "plot_class_enrichment", "plot_pathway_dotplot",
      "diverging_cmap", "ion_cmap", "plot_spatial", "plot_spatial_metabolite",
      "plot_embedding_grid", "plot_umap", "plot_markers", "plot_volcano", "plot_heatmap",
      "plot_violin", "plot_cluster_composition", "plot_morans", "plot_qc",
      "plot_spatial_gradient", "plot_colocalization_network", "load_image", "align_image",
      "extract_image_features", "plot_image_overlay"]),
    ("Manage", "Subsetting, merging and saving.",
     ["subset_obs", "merge_samples", "split_by_obs", "save_results", "save_adata"]),
]

#: Exported names that are not functions and so have no place in the index.
NOT_FUNCTIONS = {"PALETTE", "CHEMICAL_CLASSES", "MANIFEST_VERSION"}


def first_paragraph(doc: str | None) -> str:
    """The opening paragraph of a docstring, flattened to one line."""
    if not doc:
        return ""
    out: List[str] = []
    for line in inspect.cleandoc(doc).splitlines():
        if not line.strip():
            if out:
                break
            continue
        # Stop at the first numpydoc section header rather than swallowing it.
        if line.strip().startswith(("Parameters", "Returns", "Raises", "Examples", ".. ", "----")):
            break
        out.append(line.strip())
    return " ".join(out)


def parameters(fn) -> List[Dict[str, str]]:
    try:
        signature = inspect.signature(fn)
    except (ValueError, TypeError):
        return []
    rows = []
    for name, param in signature.parameters.items():
        if name in ("self", "args", "kwargs"):
            continue
        annotation = ""
        if param.annotation is not inspect.Parameter.empty:
            annotation = (
                param.annotation if isinstance(param.annotation, str)
                else getattr(param.annotation, "__name__", str(param.annotation))
            )
        rows.append({
            "name": name,
            # Optional[X] reads as noise in a table; the default column already
            # says whether something can be left out.
            "type": annotation.replace("Optional[", "").replace("]", ""),
            "default": "" if param.default is inspect.Parameter.empty else repr(param.default),
        })
    return rows


def main() -> int:
    documented: set = set()
    index: List[Dict[str, Any]] = []

    for title, blurb, names in GROUPS:
        items = []
        for name in names:
            fn = getattr(mt, name, None)
            if fn is None:
                print(f"  ! {name} is in a group but not exported, skipped")
                continue
            documented.add(name)
            try:
                signature = f"{name}{inspect.signature(fn)}"
            except (ValueError, TypeError):
                signature = name
            items.append({
                "name": name,
                "doc": first_paragraph(fn.__doc__),
                "sig": signature,
                "params": parameters(fn),
            })
        index.append({"title": title, "blurb": blurb, "items": items})

    exported = {
        n for n in mt.__all__
        if n not in NOT_FUNCTIONS and not n.endswith(("Error", "Warning"))
    }
    missing = sorted(exported - documented)
    if missing:
        # Loud, and a non-zero exit, so CI fails rather than shipping a
        # reference with holes in it.
        print(f"  ! {len(missing)} exported name(s) are in no group: {missing}")

    out = Path(__file__).parent / "api.json"
    # Docstrings are full of em-dashes and Greek letters; without an explicit
    # encoding this fails outright on Windows.
    out.write_text(json.dumps(index, indent=1), encoding="utf-8")
    total = sum(len(g["items"]) for g in index)
    print(f"wrote {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out} "
          f",  {total} functions across {len(index)} stages")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

