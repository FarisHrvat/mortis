"""
MORTIS command line
===================
Run a cohort analysis from a config file, without writing any Python.

Why bother
----------
Plenty of people who need this analysis do not want to write a script for it,
and plenty of the machines it should run on, a cluster node, a container,
have no interest in an interactive session. Both want the same thing: one
command, one config file, results on disk.

The config is also the point. A YAML file that produced a result is a far
better methods section than a paragraph reconstructed from memory, and it can
be committed next to the paper. Every run writes its config back out alongside
the results, so what actually ran is never in question.

    mortis run analysis.yaml
    mortis template > analysis.yaml     # a commented starting point
    mortis verify results/manifest.json --data results/pseudobulk.h5ad

Stages can be skipped and resumed, because on a real cohort the expensive part
is loading and normalising, and you will want to re-run the statistics more
than once.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

TEMPLATE = """\
# MORTIS analysis config. Keep it next to your results.

input:
  # A folder of exports, or a single .h5ad / .csv / .xlsx file.
  path: ./data
  # The independent unit. Use the patient, not the section, when one patient
  # gave more than one section.
  sample_key: patient
  # The physical section. Spatial organization is summarised over this.
  section_key: section
  # The comparison you care about, and the two groups being compared.
  group_key: response
  groups: [Responder, Non Responder]

preprocess:
  filter_background: true
  background_cutoff: 1.5
  min_annotation_score: 0.0     # 0 disables; 0.5 is "high confidence"
  remove_drugs: false           # DrugBank compounds, for pharmacology studies
  normalize: tic                # tic | median | none
  log1p: true

analysis:
  abundance: true
  organization: true            # needs >= 6 sections per group to say anything
  organization_metrics: [morans_i, entropy, gini]
  classify_compounds: true
  pathways: false               # needs network access on first run
  bootstrap: 0                  # e.g. 2000 for a CI on Cliff's delta

output:
  dir: ./results
  figures: true
  formats: [pdf, png]
  theme: print                  # print | light | dark
  manifest: true                # the file a reviewer can check your re-run against

runtime:
  seed: 0
  n_jobs: null                  # null = every core the machine will admit to
"""


def _load_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "Reading a config needs PyYAML. Install it with:\n"
            "    pip install 'mortis-spatial[cli]'"
        )
    if not path.exists():
        raise SystemExit(
            f"No config at '{path}'.\n"
            "Write one with:  mortis template > analysis.yaml"
        )
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"'{path}' does not look like a MORTIS config.")
    return config


def _get(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Fetch a dotted key, so a half-filled config still runs on defaults."""
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


def _load_input(mt, source: Path, sample_key: str) -> Any:
    """A folder of sections, or a single file."""
    if source.is_dir():
        adatas = mt.load_from_folder(str(source))
        if not adatas:
            raise SystemExit(f"No readable data files in '{source}'.")
        if len(adatas) == 1:
            return adatas[0]
        labels = [
            a.uns.get("source_file", f"sample_{i}").rsplit(".", 1)[0]
            for i, a in enumerate(adatas)
        ]
        return mt.merge_samples(adatas, sample_labels=labels, sample_col=sample_key)
    return mt.read_metabolomics_data(str(source))


def cmd_run(args: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")           # a cluster node has no display, and does not want one
    import mortis as mt

    config = _load_config(Path(args.config))

    source = Path(_get(config, "input.path", "."))
    sample_key = _get(config, "input.sample_key", "sample")
    section_key = _get(config, "input.section_key", sample_key)
    group_key = _get(config, "input.group_key")
    groups = _get(config, "input.groups", [])
    out = Path(_get(config, "output.dir", "./results"))
    out.mkdir(parents=True, exist_ok=True)

    if group_key and len(groups) != 2:
        raise SystemExit(
            f"input.groups needs exactly two group names, got {groups!r}."
        )

    print(f"[MORTIS] Loading {source}")
    adata = _load_input(mt, source, sample_key)
    print(f"[MORTIS] {adata.n_obs:,} pixels x {adata.n_vars:,} features")

    for key in (sample_key, section_key, group_key):
        if key and key not in adata.obs.columns:
            raise SystemExit(
                f"Column '{key}' is not in the data. Available: "
                f"{sorted(adata.obs.columns)}"
            )

    # --- preprocess ---
    if _get(config, "preprocess.min_annotation_score", 0) > 0:
        adata = mt.filter_by_score(adata, min_score=_get(config, "preprocess.min_annotation_score"))
    if _get(config, "preprocess.remove_drugs", False):
        adata = mt.filter_drugs(adata)

    normalize = _get(config, "preprocess.normalize", "tic")
    if normalize == "tic":
        adata = mt.tic_normalize(adata)
    elif normalize == "median":
        adata = mt.median_normalize(adata)
    if _get(config, "preprocess.log1p", True):
        adata = mt.log1p_transform(adata)

    mt.record_step(adata, "cli preprocess", {
        "normalize": normalize, "log1p": _get(config, "preprocess.log1p", True),
    })

    results: Dict[str, Any] = {}
    figures: Dict[str, Any] = {}
    theme = _get(config, "output.theme", "print")
    mt.set_publication_style(theme=theme)

    pb = mt.pseudobulk(adata, sample_key=sample_key)

    # --- abundance ---
    if group_key and _get(config, "analysis.abundance", True):
        ab = mt.differential_abundance(
            pb, group_key, groups[0], groups[1],
            bootstrap=int(_get(config, "analysis.bootstrap", 0) or 0),
            random_state=int(_get(config, "runtime.seed", 0)),
        )
        ab.to_csv(out / "differential_abundance.csv", index=False)
        results["abundance"] = ab
        if _get(config, "output.figures", True):
            figures["effect_size"] = mt.plot_effect_size(ab, group_labels=tuple(groups))
            figures["volcano"] = mt.plot_delta_volcano(ab)

    # --- organization ---
    org = None
    if group_key and _get(config, "analysis.organization", True):
        if "spatial" not in adata.obsm:
            print("[MORTIS] No spatial coordinates, skipping organization.")
        else:
            org = mt.spatial_organization(
                adata, sample_key=section_key,
                metrics=tuple(_get(config, "analysis.organization_metrics",
                                   ["morans_i", "entropy", "gini"])),
            )
            do = mt.differential_spatial_organization(org, group_key, groups[0], groups[1])
            do.to_csv(out / "differential_organization.csv", index=False)
            results["organization"] = do
            if "abundance" in results:
                both = mt.compare_abundance_and_organization(results["abundance"], do)
                both.to_csv(out / "two_axis.csv", index=False)
                results["two_axis"] = both
                if _get(config, "output.figures", True):
                    figures["two_axis"] = mt.plot_abundance_vs_organization(both)

    # --- annotation ---
    if _get(config, "analysis.classify_compounds", True):
        adata = mt.classify_compounds(adata)
        mt.classification_report(adata).to_csv(out / "chemical_classes.csv", index=False)
        if "abundance" in results:
            classes = mt.class_enrichment(results["abundance"], adata)
            classes.to_csv(out / "class_enrichment.csv", index=False)
            if len(classes) and _get(config, "output.figures", True):
                figures["class_enrichment"] = mt.plot_class_enrichment(classes)

    if _get(config, "analysis.pathways", False) and "abundance" in results:
        try:
            ids, paths = mt.annotate_pathways(adata, results["abundance"])
            ids.to_csv(out / "compound_ids.csv", index=False)
            paths.to_csv(out / "pathway_enrichment.csv", index=False)
            if len(paths) and _get(config, "output.figures", True):
                figures["pathways"] = mt.plot_pathway_dotplot(paths)
        except Exception as exc:                      # network, or nothing mapped
            print(f"[MORTIS] Pathway step skipped: {exc}")

    # --- figures ---
    for name, fig in figures.items():
        mt.save_figure(
            fig, out / name,
            formats=tuple(_get(config, "output.formats", ["pdf"])),
            provenance={"config": str(args.config), "stage": name},
            close=True,
        )

    # --- receipts ---
    pb.write_h5ad(out / "pseudobulk.h5ad")
    if _get(config, "output.manifest", True):
        mt.export_manifest(
            out / "manifest.json", adata=pb, results=results,
            analysis=_get(config, "input.path", "unnamed"),
            notes="Produced by `mortis run`; config.yaml alongside is the exact input.",
        )

    # The config that ran, written back out. Not the config you meant to run.
    try:
        import yaml
        (out / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
    except ImportError:
        pass

    print(f"\n[MORTIS] Done. {len(results)} result table(s), "
          f"{len(figures)} figure(s) -> {out}/")
    return 0


def cmd_template(args: argparse.Namespace) -> int:
    sys.stdout.write(TEMPLATE)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import anndata as ad

    import mortis as mt

    adata = ad.read_h5ad(args.data) if args.data else None
    results = {}
    for spec in args.result or []:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--result expects name=path.csv, got '{spec}'")
        import pandas as pd
        results[name] = pd.read_csv(path)

    report = mt.verify_manifest(args.manifest, adata=adata, results=results or None)
    report.to_csv(sys.stdout, index=False)
    return 0 if not (report["status"] == "fail").any() else 1


def cmd_info(args: argparse.Namespace) -> int:
    """What is installed, and what the machine will give us. First thing to
    paste into a bug report."""
    import platform

    import matplotlib
    matplotlib.use("Agg")
    import mortis as mt
    from mortis.reproducibility import _versions

    print(f"MORTIS {mt.__version__}")
    print(f"  python    {platform.python_version()} ({platform.machine()})")
    print(f"  platform  {platform.platform()}")
    try:
        import numba
        print(f"  threads   {numba.config.NUMBA_NUM_THREADS} available to numba")
    except ImportError:
        pass
    print("  packages")
    for name, version in _versions().items():
        print(f"    {name:<16s} {version}")
    print(f"  public API: {len(mt.__all__)} names")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mortis",
        description="Cohort-scale analysis for spatial metabolomics.",
        epilog="Start with:  mortis template > analysis.yaml",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an analysis from a config file")
    run.add_argument("config", help="path to the YAML config")
    run.set_defaults(func=cmd_run)

    template = sub.add_parser("template", help="print a commented config to stdout")
    template.set_defaults(func=cmd_template)

    verify = sub.add_parser("verify", help="check a re-run against a manifest")
    verify.add_argument("manifest")
    verify.add_argument("--data", help="the .h5ad the re-run produced")
    verify.add_argument("--result", action="append",
                        help="name=path.csv, repeatable")
    verify.set_defaults(func=cmd_verify)

    info = sub.add_parser("info", help="versions, threads and environment")
    info.set_defaults(func=cmd_info)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

