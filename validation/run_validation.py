#!/usr/bin/env python3
"""
End-to-end validation of MORTIS on public data.

Downloads a published, openly available imaging mass spectrometry study from
METASPACE, reconstructs pixel matrices from the ion images, and runs the whole
analysis — cleaning, pseudobulk, patient-level testing, spatial organization,
cross-group comparison, figures and a verification manifest — using nothing but
MORTIS and its declared dependencies.

The study
---------
``2026_Frank_solanum_recal`` on METASPACE: MALDI imaging of grafted Solanum
stems, with sections taken above and below the graft junction. That gives a
real two-group comparison (``top`` vs ``bottom``) on real tissue, with no
simulation anywhere in the pipeline.

Why this study
--------------
It needs to be public, it needs enough sections for sample-level statistics to
mean anything, and the grouping has to be biological rather than technical.
This one has 39 sections across two anatomical positions, which clears the
six-per-arm floor MORTIS documents for spatial-organization testing by a wide
margin.

Run it
------
    python validation/run_validation.py --out validation/results

Everything is cached under ``--cache``, so a second run costs no downloads and
reproduces bit-for-bit. The manifest written at the end lets anyone check their
run against ours without either party sharing data.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

GRAPHQL = "https://metaspace2020.org/graphql"
PROJECT = "2026_Frank_solanum_recal"

#: METASPACE encodes ion images as 8-bit RGBA PNGs: the colour channels carry
#: intensity scaled against `maxIntensity`, and alpha marks which pixels were
#: actually acquired. 8 bits is coarse for absolute quantification but ample
#: for the rank-based statistics MORTIS uses.
_ALPHA_ON = 8


def gql(query: str, timeout: float = 90.0) -> dict:
    request = urllib.request.Request(
        GRAPHQL,
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    if "errors" in body:
        raise RuntimeError(f"METASPACE rejected the query: {body['errors'][0]['message']}")
    return body["data"]


def cached(cache: Path, name: str, fetch) -> bytes:
    """Fetch once, then serve from disk. Keeps reruns free and deterministic."""
    path = cache / name
    if path.exists():
        return path.read_bytes()
    blob = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return blob


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------

def find_sections(cache: Path) -> pd.DataFrame:
    """
    Every section in the study, with the group label it belongs to.

    The group is taken from the **dataset name**, not the ``condition``
    metadata field. Those two disagree for several sections — for instance
    ``tomato-tomato-bottom1_sl4a`` carries ``condition = "top"``. The name
    states the anatomical position explicitly and is what the submitters used
    consistently, so it wins; the disagreement is counted and reported rather
    than quietly resolved.
    """
    raw = cached(cache, "datasets.json", lambda: json.dumps(gql(
        '{ allDatasets(filter:{polarity:POSITIVE}, limit:500){'
        ' id name condition organism organismPart projects{name} } }'
    )).encode())
    everything = json.loads(raw)["allDatasets"]

    rows = []
    for d in everything:
        if not any(p["name"] == PROJECT for p in (d.get("projects") or [])):
            continue
        name = d["name"]
        match = re.search(r"(top|bottom)", name, re.IGNORECASE)
        if not match:
            continue
        position = match.group(1).lower()
        # Names are scion-rootstock: "tomato-pepper_top1" is a tomato scion on
        # a pepper rootstock, so the section ABOVE the junction is tomato
        # tissue and the section BELOW it is pepper. Which means "top vs
        # bottom" on a hetero-graft is a species contrast wearing a position
        # label — the single most important thing to get right about this study.
        parts = re.split(r"[-_]", name.lower())
        species = [p for p in parts if p in ("tomato", "pepper")]
        scion = species[0] if species else None
        rootstock = species[1] if len(species) > 1 else scion
        rows.append({
            "dataset_id": d["id"],
            "name": name,
            "position": position,
            "metadata_condition": d.get("condition"),
            "graft": f"{scion}-{rootstock}" if scion else "unknown",
            "homograft": scion == rootstock,
            "tissue_species": scion if position == "top" else rootstock,
        })

    table = pd.DataFrame(rows)
    table["metadata_agrees"] = table["position"] == table["metadata_condition"]
    return table


def common_ions(cache: Path, dataset_ids: List[str], fdr: float, want: int) -> pd.DataFrame:
    """Ions annotated at the given FDR in *every* selected section."""
    per_dataset = {}
    for ds in dataset_ids:
        raw = cached(cache, f"ann_{ds}.json", lambda ds=ds: json.dumps(gql(
            f'{{ allAnnotations(filter:{{fdrLevel:{fdr}}}, datasetFilter:{{ids:"{ds}"}},'
            f' limit:400){{ ion mz fdrLevel possibleCompounds{{name}}'
            f' isotopeImages{{ url maxIntensity }} }} }}'
        )).encode())
        rows = {}
        for a in json.loads(raw)["allAnnotations"]:
            images = a.get("isotopeImages") or []
            if not images or not images[0].get("url"):
                continue
            compounds = a.get("possibleCompounds") or []
            rows[a["ion"]] = {
                "ion": a["ion"],
                "mz": a["mz"],
                "compound": compounds[0]["name"] if compounds else a["ion"],
                "url": images[0]["url"],
                "max_intensity": images[0].get("maxIntensity") or 1.0,
            }
        per_dataset[ds] = rows

    shared = set.intersection(*(set(r) for r in per_dataset.values())) if per_dataset else set()

    # Rank by how much an ion VARIES between sections, not by how bright it is.
    # Intensity-ranked selection returns the housekeeping metabolome: the ions
    # that are brightest everywhere are, almost by definition, the ones that do
    # not distinguish anything. Requiring shared annotation already biases
    # toward the common core, so the selection step should at least not
    # compound it. Faint ions are still excluded as noise.
    floor = np.percentile(
        [np.mean([per_dataset[ds][i]["max_intensity"] for ds in dataset_ids]) for i in shared], 25
    ) if shared else 0.0
    spread = {}
    for ion in shared:
        values = np.array([per_dataset[ds][ion]["max_intensity"] for ds in dataset_ids], float)
        if values.mean() < floor:
            continue
        spread[ion] = float(values.std() / (values.mean() + 1e-9))   # coefficient of variation
    chosen = sorted(spread, key=lambda i: -spread[i])[:want]
    return pd.DataFrame([per_dataset[dataset_ids[0]][ion] for ion in chosen]), per_dataset


def ion_image(cache: Path, url: str, max_intensity: float) -> Tuple[np.ndarray, np.ndarray]:
    """Decode one ion image to (intensity grid, acquired-pixel mask)."""
    from PIL import Image

    key = "img_" + url.rsplit("/", 1)[-1] + ".png"
    blob = cached(cache, key, lambda: urllib.request.urlopen(url, timeout=90).read())
    rgba = np.array(Image.open(io.BytesIO(blob)).convert("RGBA"))
    intensity = rgba[..., 0].astype(np.float32) / 255.0 * float(max_intensity)
    return intensity, rgba[..., 3] > _ALPHA_ON


def build_section(cache: Path, ds: str, ions: pd.DataFrame, per_dataset: dict) -> Optional[ad.AnnData]:
    """One AnnData per section: acquired pixels x shared ions, with coordinates."""
    columns, mask, shape = [], None, None
    for _, ion in ions.iterrows():
        entry = per_dataset[ds][ion["ion"]]
        grid, acquired = ion_image(cache, entry["url"], entry["max_intensity"])
        if shape is None:
            shape, mask = grid.shape, acquired
        elif grid.shape != shape:
            return None                       # ragged export; skip the section
        else:
            mask = mask & acquired
        columns.append(grid)

    if mask is None or mask.sum() < 200:
        return None

    ys, xs = np.nonzero(mask)
    X = np.column_stack([c[ys, xs] for c in columns]).astype(np.float32)
    section = ad.AnnData(X=X)
    section.obsm["spatial"] = np.column_stack([xs, ys]).astype(np.float64)
    section.var_names = ions["compound"].astype(str).values
    section.var_names_make_unique()
    return section


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="validation/results")
    parser.add_argument("--cache", default="validation/cache")
    parser.add_argument("--sections", type=int, default=16, help="sections per group")
    parser.add_argument("--ions", type=int, default=40)
    parser.add_argument("--fdr", type=float, default=0.1)
    args = parser.parse_args()

    out, cache = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import mortis as mt

    print("=" * 74)
    print("MORTIS validation — public METASPACE study", PROJECT)
    print("=" * 74)

    sections = find_sections(cache)
    disagreements = int((~sections["metadata_agrees"]).sum())
    print(f"\n{len(sections)} sections in the study "
          f"({dict(Counter(sections['position']))})")
    print(f"  {disagreements} section(s) whose 'condition' metadata disagrees with the "
          f"name; grouping follows the name.")

    print(f"  graft types: {dict(Counter(sections['graft']))}")

    # Two contrasts, deliberately:
    #   species  — tomato vs pepper tissue. Different plants, so a method that
    #              works must find something. A positive control.
    #   position — top vs bottom within HOMO-grafts only, where both sections
    #              are the same species and position is the only thing varying.
    #              The honest version of the question the labels invite.
    balanced = pd.concat([
        sections[sections["tissue_species"] == sp].head(args.sections)
        for sp in ("tomato", "pepper")
    ]).drop_duplicates("dataset_id").reset_index(drop=True)
    ids = balanced["dataset_id"].tolist()
    print(f"  using {len(balanced)} sections "
          f"(species {dict(Counter(balanced['tissue_species']))}, "
          f"position {dict(Counter(balanced['position']))})")

    print(f"\nresolving ions annotated at FDR <= {args.fdr} in every section...")
    ions, per_dataset = common_ions(cache, ids, args.fdr, args.ions)
    print(f"  {len(ions)} shared ions selected")

    print("\ndownloading ion images and building sections...")
    built, labels, names = [], [], []
    for _, row in balanced.iterrows():
        section = build_section(cache, row["dataset_id"], ions, per_dataset)
        if section is None:
            print(f"  skipped {row['name']} (inconsistent image geometry)")
            continue
        section.obs["section"] = row["name"]
        section.obs["position"] = row["position"]
        section.obs["graft"] = row["graft"]
        section.obs["tissue_species"] = row["tissue_species"]
        built.append(section)
        labels.append(row["position"])
        names.append(row["name"])
        print(f"  {row['name']:<36s} {section.n_obs:>7,d} px x {section.n_vars} ions  [{row['position']}]")

    if len(set(labels)) < 2:
        print("\nNot enough sections survived to compare two groups.")
        return 1

    adata = mt.merge_samples(built, sample_labels=names, sample_col="section")
    for column in ("position", "graft", "tissue_species"):
        adata.obs[column] = pd.Categorical(
            [s.obs[column].iloc[0] for s, n in zip(built, names) for _ in range(s.n_obs)]
        )
    print(f"\ncohort: {adata.n_obs:,} pixels x {adata.n_vars} ions, "
          f"{adata.obs['section'].nunique()} sections")

    mt.record_step(adata, "loaded from METASPACE", {
        "project": PROJECT, "fdr": args.fdr, "n_sections": adata.obs["section"].nunique(),
    })

    print("\nnormalising...")
    adata = mt.tic_normalize(adata)
    adata = mt.log1p_transform(adata)

    print("\ntesting at section level...")
    pb = mt.pseudobulk(adata, sample_key="section")
    org = mt.spatial_organization(adata, sample_key="section",
                                  metrics=("morans_i", "entropy", "gini"))

    print("\n--- positive control: tomato vs pepper tissue ---")
    ab = mt.differential_abundance(pb, "tissue_species", "tomato", "pepper", bootstrap=2000)
    do = mt.differential_spatial_organization(org, "tissue_species", "tomato", "pepper")
    both = mt.compare_abundance_and_organization(ab, do)

    # Position, but only where species is held constant. On hetero-grafts
    # "top vs bottom" is a species contrast in disguise, so including them
    # would answer a different question than the one being asked.
    homo = pb.obs["graft"].astype(str).isin([g for g in pb.obs["graft"].astype(str).unique()
                                             if g.split("-")[0] == g.split("-")[-1]])
    ab_pos = do_pos = None
    if homo.sum() >= 6 and pb[homo].obs["position"].nunique() == 2:
        print("\n--- position within homo-grafts (species held constant) ---")
        ab_pos = mt.differential_abundance(pb[homo].copy(), "position", "top", "bottom")
        do_pos = mt.differential_spatial_organization(
            org[org.obs["graft"].astype(str).isin(pb[homo].obs["graft"].astype(str))].copy(),
            "position", "top", "bottom")
    else:
        print(f"\n--- position within homo-grafts: only {int(homo.sum())} section(s), "
              "below the six-per-arm floor; not tested ---")

    print("\nchemical classes...")
    adata = mt.classify_compounds(adata)
    classes = mt.class_enrichment(ab, adata)

    # ── figures ──
    print("\nfigures...")
    mt.set_publication_style(base_size=10)
    figures = {
        "effect_size": mt.plot_effect_size(ab, top_n=14, group_labels=("tomato", "pepper")),
        "volcano": mt.plot_delta_volcano(ab),
        "two_axis": mt.plot_abundance_vs_organization(both),
        "organization": mt.plot_organization_heatmap(org, group_key="tissue_species", result=do, top_n=12),
    }
    if len(classes):
        figures["classes"] = mt.plot_class_enrichment(classes)
    top_ion = ab["metabolite"].iloc[0]
    figures["ion_images"] = mt.plot_ion_images(
        adata, top_ion, sample_key="section", group_key="tissue_species", n_cols=4, panel_size=1.3
    )
    for name, fig in figures.items():
        mt.save_figure(fig, out / name, formats=("pdf", "png"),
                       provenance={"study": PROJECT, "figure": name}, close=True)

    # ── tables and manifest ──
    ab.to_csv(out / "differential_abundance.csv", index=False)
    do.to_csv(out / "differential_organization.csv", index=False)
    both.to_csv(out / "two_axis.csv", index=False)
    classes.to_csv(out / "class_enrichment.csv", index=False)
    balanced.to_csv(out / "sections.csv", index=False)

    mt.export_manifest(
        out / "manifest.json", adata=pb,
        results={"abundance": ab, "organization": do, "two_axis": both},
        analysis=f"MORTIS validation on {PROJECT}",
        notes=("Public METASPACE study, top vs bottom stem sections. Grouping taken "
               "from dataset names; see sections.csv for the metadata disagreement."),
        extra={"n_sections": len(built), "n_ions": adata.n_vars, "fdr": args.fdr},
    )

    # ── what we found ──
    counts = both["classification"].value_counts()
    summary = {
        "study": PROJECT,
        "sections_used": len(built),
        "sections_total_in_study": len(sections),
        "metadata_disagreements": disagreements,
        "pixels": int(adata.n_obs),
        "ions": int(adata.n_vars),
        "contrast": "tomato vs pepper tissue (positive control)",
        "abundance_fdr_hits": int((ab["pval_adj"] < 0.05).sum()),
        "organization_fdr_hits": int((do["pval_adj"] < 0.05).sum()),
        "position_within_homografts": (
            None if ab_pos is None else {
                "abundance_fdr_hits": int((ab_pos["pval_adj"] < 0.05).sum()),
                "organization_fdr_hits": int((do_pos["pval_adj"] < 0.05).sum()),
                "n_sections": int(homo.sum()),
            }
        ),
        "classification": {k: int(v) for k, v in counts.items()},
        "top_abundance": ab.head(6)[["metabolite", "delta", "pval_adj"]].to_dict("records"),
        "top_organization": do.head(6)[["metabolite", "delta", "pval_adj"]].to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"  sections            {len(built)}  ({dict(Counter(labels))})")
    print(f"  pixels x ions       {adata.n_obs:,} x {adata.n_vars}")
    print(f"  abundance, FDR<.05  {summary['abundance_fdr_hits']}/{adata.n_vars}")
    print(f"  organization        {summary['organization_fdr_hits']}/{adata.n_vars}")
    print(f"  two-axis            {summary['classification']}")
    print(f"\n  written to {out}/")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
