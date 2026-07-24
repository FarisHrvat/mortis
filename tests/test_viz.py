"""
Tests for publication figures and export.

The export claims are checked against the written file, not against rcParams.
"Text stays editable" and "LaTeX was used" are properties of the PDF bytes; a
test that only asserts a setting was applied proves nothing about the artefact
a co-author actually opens.
"""

from __future__ import annotations

import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import mortis as mt  # noqa: E402
from mortis.exceptions import InvalidParameterError  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_style():
    yield
    mt.reset_style()
    plt.close("all")


@pytest.fixture
def result():
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame({
        "metabolite": [f"m{i:03d}" for i in range(n)],
        "delta": rng.uniform(-1, 1, n),
        "pval_adj": rng.uniform(0, 1, n),
        "n_group1": 6,
        "n_group2": 6,
    })


@pytest.fixture
def merged():
    rng = np.random.default_rng(1)
    n = 40
    classification = np.array(
        ["organization only"] * 5 + ["both"] * 5 + ["abundance only"] * 5 + ["neither"] * 25
    )
    return pd.DataFrame({
        "metabolite": [f"m{i:03d}" for i in range(n)],
        "delta_abundance": rng.uniform(-1, 1, n),
        "delta_organization": rng.uniform(-1, 1, n),
        "pval_adj_abundance": rng.uniform(0, 1, n),
        "pval_adj_organization": rng.uniform(0, 1, n),
        "classification": classification,
    })


@pytest.fixture
def comparison():
    rng = np.random.default_rng(2)
    n = 40
    table = pd.DataFrame({
        "metabolite": [f"m{i:03d}" for i in range(n)],
        "delta_cohort_a": rng.uniform(-1, 1, n),
        "delta_cohort_b": rng.uniform(-1, 1, n),
        "agreement": rng.choice(["concordant", "discordant", "weak"], n),
    })
    table.attrs["rho"] = 0.42
    return table


# ---------------------------------------------------------------------------
# Export guarantees
# ---------------------------------------------------------------------------

class TestExport:

    def test_pdf_text_is_editable_not_outlined(self, tmp_path, result):
        """
        The headline claim, checked structurally.

        pdf.fonttype=42 makes matplotlib write a Type0 font with a
        CIDFontType2 descendant: /FontFile2 carries the embedded TrueType
        outlines and /ToUnicode maps glyphs back to characters, which together
        are what let an editor treat the text as text. The default Type 3 path
        produces neither.
        """
        mt.set_publication_style()
        fig = mt.plot_effect_size(result, top_n=10)
        out = mt.save_figure(fig, tmp_path / "fig")["pdf"]

        raw = out.read_bytes()
        assert b"/FontFile2" in raw, "no embedded TrueType outlines - text is not editable"
        assert b"/ToUnicode" in raw, "no ToUnicode CMap - glyphs cannot be mapped back to text"
        assert b"/CIDFontType2" in raw
        assert b"/Type3" not in raw, "Type 3 fonts present - text becomes uneditable outlines"

    @pytest.mark.skipif(shutil.which("pdftotext") is None, reason="requires pdftotext")
    def test_pdf_text_is_actually_extractable(self, tmp_path, result):
        """
        The same claim, checked behaviourally: a PDF reader must be able to get
        the axis labels back out as characters. This is what a co-author
        selecting text in Illustrator, or a reviewer searching the PDF, relies
        on.
        """
        import subprocess

        mt.set_publication_style()
        fig = mt.plot_effect_size(result, top_n=6, group_labels=("responder", "nonresponder"))
        out = mt.save_figure(fig, tmp_path / "fig")["pdf"]

        extracted = subprocess.run(
            ["pdftotext", str(out), "-"], capture_output=True, text=True, check=True
        ).stdout
        assert "responder" in extracted, f"axis text not recoverable from PDF: {extracted!r}"
        assert "m0" in extracted, "metabolite labels not recoverable from PDF"

    def test_all_formats_written(self, tmp_path, result):
        mt.set_publication_style()
        fig = mt.plot_effect_size(result, top_n=5)
        written = mt.save_figure(fig, tmp_path / "fig", formats=("pdf", "svg", "png"))
        assert set(written) == {"pdf", "svg", "png"}
        for path in written.values():
            assert path.exists() and path.stat().st_size > 0

    def test_svg_keeps_text_as_text(self, tmp_path, result):
        mt.set_publication_style()
        fig = mt.plot_effect_size(result, top_n=5)
        svg = mt.save_figure(fig, tmp_path / "fig", formats=("svg",))["svg"]
        assert "<text" in svg.read_text(), "SVG text was converted to paths"

    def test_provenance_is_embedded_and_hashed(self, tmp_path, result):
        mt.set_publication_style()
        fig = mt.plot_effect_size(result, top_n=5)
        out = mt.save_figure(
            fig, tmp_path / "fig", provenance={"cohort": "vedolizumab", "seed": 7}
        )["pdf"]
        raw = out.read_bytes()
        assert b"mortis_hash=" in raw
        assert b"vedolizumab" in raw

    def test_provenance_hash_is_stable_and_sensitive(self, tmp_path, result):
        mt.set_publication_style()

        def hash_for(provenance, name):
            fig = mt.plot_effect_size(result, top_n=5)
            out = mt.save_figure(fig, tmp_path / name, provenance=provenance)["pdf"]
            plt.close(fig)
            raw = out.read_bytes()
            start = raw.index(b"mortis_hash=") + len(b"mortis_hash=")
            return raw[start:start + 12]

        assert hash_for({"seed": 1}, "a") == hash_for({"seed": 1}, "b")
        assert hash_for({"seed": 1}, "c") != hash_for({"seed": 2}, "d")

    def test_extension_on_path_is_replaced(self, tmp_path, result):
        mt.set_publication_style()
        fig = mt.plot_effect_size(result, top_n=5)
        written = mt.save_figure(fig, tmp_path / "fig.pdf", formats=("png",))
        assert written["png"].name == "fig.png"
        assert not (tmp_path / "fig.pdf.png").exists()

    def test_creates_missing_directories(self, tmp_path, result):
        mt.set_publication_style()
        fig = mt.plot_effect_size(result, top_n=5)
        out = mt.save_figure(fig, tmp_path / "nested" / "deep" / "fig")["pdf"]
        assert out.exists()

    def test_rejects_unknown_format(self, tmp_path, result):
        fig = mt.plot_effect_size(result, top_n=5)
        with pytest.raises(InvalidParameterError, match="Unknown format"):
            mt.save_figure(fig, tmp_path / "fig", formats=("tiff",))


# ---------------------------------------------------------------------------
# Style and LaTeX
# ---------------------------------------------------------------------------

class TestStyle:

    def test_sets_editable_font_types(self):
        mt.set_publication_style()
        assert matplotlib.rcParams["pdf.fonttype"] == 42
        assert matplotlib.rcParams["ps.fonttype"] == 42
        assert matplotlib.rcParams["svg.fonttype"] == "none"

    def test_reset_restores_defaults(self):
        before = matplotlib.rcParams["font.size"]
        mt.set_publication_style(base_size=17.0)
        assert matplotlib.rcParams["font.size"] == 17.0
        mt.reset_style()
        assert matplotlib.rcParams["font.size"] == before

    def test_rejects_bad_parameters(self):
        with pytest.raises(InvalidParameterError):
            mt.set_publication_style(base_size=0)
        with pytest.raises(InvalidParameterError):
            mt.set_publication_style(font_family="comic-sans")

    def test_latex_falls_back_with_warning_when_unavailable(self, monkeypatch):
        """Missing LaTeX must degrade to mathtext, never crash mid-figure."""
        monkeypatch.setattr(shutil, "which", lambda _: None)
        with pytest.warns(UserWarning, match="dvipng"):
            mt.set_publication_style(latex=True)
        assert matplotlib.rcParams["text.usetex"] is False

    @pytest.mark.skipif(
        not (shutil.which("latex") and shutil.which("dvipng")),
        reason="requires a LaTeX installation",
    )
    def test_latex_actually_renders(self, tmp_path, result):
        """Only meaningful if a real figure survives the LaTeX pipeline."""
        mt.set_publication_style(latex=True)
        assert matplotlib.rcParams["text.usetex"] is True
        fig = mt.plot_delta_volcano(result, label_top=0)
        out = mt.save_figure(fig, tmp_path / "latex_fig")["pdf"]
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

class TestFigures:

    def test_effect_size_orders_and_fills_by_significance(self, result):
        fig = mt.plot_effect_size(result, top_n=10, fdr_threshold=0.5)
        ax = fig.axes[0]
        assert len(ax.get_yticklabels()) == 10
        bars = [p for p in ax.patches if hasattr(p, "get_width")]
        assert len(bars) >= 10

    def test_effect_size_axis_is_bounded(self, result):
        ax = mt.plot_effect_size(result, top_n=5).axes[0]
        assert ax.get_xlim() == pytest.approx((-1.05, 1.05))

    def test_volcano_handles_zero_adjusted_pvalues(self, result):
        """BH correction can return exactly 0; -log10 must stay finite."""
        result = result.copy()
        result.loc[0, "pval_adj"] = 0.0
        fig = mt.plot_delta_volcano(result)
        ydata = fig.axes[0].collections[0].get_offsets()[:, 1]
        assert np.isfinite(ydata).all()

    def test_two_axis_figure_is_square_and_labelled(self, merged):
        fig = mt.plot_abundance_vs_organization(merged)
        ax = fig.axes[0]
        assert ax.get_aspect() == 1.0
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("organization only" in lbl for lbl in labels)

    def test_signature_comparison_annotates_rho(self, comparison):
        ax = mt.plot_signature_comparison(comparison).axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert any("0.42" in t for t in texts)

    def test_signature_comparison_uses_attrs_rho_when_absent(self, comparison):
        del comparison.attrs["rho"]
        ax = mt.plot_signature_comparison(comparison, rho=-0.77).axes[0]
        assert any("-0.77" in t.get_text() for t in ax.texts)

    def test_large_scatter_is_rasterized_but_axes_are_not(self):
        """Vector axes with a rasterized point cloud - the reviewable compromise."""
        rng = np.random.default_rng(3)
        n = 8000
        big = pd.DataFrame({
            "metabolite": [f"m{i}" for i in range(n)],
            "delta": rng.uniform(-1, 1, n),
            "pval_adj": rng.uniform(0, 1, n),
        })
        ax = mt.plot_delta_volcano(big, label_top=0).axes[0]
        assert ax.collections[0].get_rasterized() is True
        assert ax.spines["left"].get_rasterized() is False

    def test_accepts_external_axes(self, result):
        fig, axes = plt.subplots(1, 2)
        mt.plot_effect_size(result, top_n=5, ax=axes[0])
        mt.plot_delta_volcano(result, ax=axes[1], label_top=0)
        assert len(fig.axes) == 2

    def test_figures_reject_wrong_tables(self, result):
        with pytest.raises(InvalidParameterError, match="missing required column"):
            mt.plot_effect_size(result.drop(columns="delta"))
        with pytest.raises(InvalidParameterError, match="missing required column"):
            mt.plot_abundance_vs_organization(result)

    def test_effect_size_rejects_bad_top_n(self, result):
        with pytest.raises(InvalidParameterError):
            mt.plot_effect_size(result, top_n=0)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_end_to_end_figure_set(tmp_path):
    """Real pipeline output through every figure, exported as vector PDF."""
    import anndata as ad

    rng = np.random.default_rng(4)
    n_vars, n_pixels, side = 25, 400, 20
    coords = np.array([[i % side, i // side] for i in range(n_pixels)], dtype=np.float64)
    blocks, sections, groups = [], [], []
    for i in range(12):
        group = "R" if i < 6 else "NR"
        offset = rng.normal(0, 1.0, n_vars)
        if group == "R":
            offset[:5] += 6.0
        blocks.append(rng.normal(offset, 1.0, (n_pixels, n_vars)))
        sections += [f"S{i:02d}"] * n_pixels
        groups += [group] * n_pixels

    X = np.vstack(blocks).astype(np.float32)
    X -= X.min()
    adata = ad.AnnData(X=X)
    adata.obsm["spatial"] = np.tile(coords, (12, 1))
    adata.obs["section"], adata.obs["response"] = sections, groups
    adata.var_names = [f"m{j:03d}" for j in range(n_vars)]

    mt.set_publication_style()
    pb = mt.pseudobulk(adata, sample_key="section")
    org = mt.spatial_organization(adata, sample_key="section", metrics=("morans_i",))
    ab = mt.differential_abundance(pb, "response", "R", "NR")
    do = mt.differential_spatial_organization(org, "response", "R", "NR")
    both = mt.compare_abundance_and_organization(ab, do)
    _, table = mt.cross_cohort_profile(ab, ab, labels=("run_a", "run_b"))

    figures = {
        "effect": mt.plot_effect_size(ab, top_n=10, group_labels=("R", "NR")),
        "volcano": mt.plot_delta_volcano(ab),
        "two_axis": mt.plot_abundance_vs_organization(both),
        "signature": mt.plot_signature_comparison(table, labels=("run A", "run B")),
    }
    for name, fig in figures.items():
        out = mt.save_figure(
            fig, tmp_path / name, provenance={"figure": name, "seed": 4}, close=True
        )["pdf"]
        assert out.stat().st_size > 0
        assert b"mortis_hash=" in out.read_bytes()


# ---------------------------------------------------------------------------
# Figures for organization, class and pathway results
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def organization_cohort():
    """Two arms where one metabolite is focal in R and diffuse in NR."""
    import anndata as ad

    rng = np.random.default_rng(7)
    side, n_vars, n_sections = 16, 12, 8
    n = side * side
    coords = np.array([[i % side, i // side] for i in range(n)], dtype=np.float64)
    centre = np.array([side / 2, side / 2])
    d = np.linalg.norm(coords - centre, axis=1)

    blocks, sections, groups = [], [], []
    for i in range(n_sections):
        group = "R" if i < n_sections // 2 else "NR"
        section = rng.random((n, n_vars)) + 0.5
        focal = np.exp(-(d ** 2) / 8.0) + rng.normal(0, 0.02, n).clip(0)
        section[:, 0] = focal if group == "R" else rng.random(n) + 0.5
        section[:, 0] *= 1000.0 / section[:, 0].sum()
        blocks.append(section)
        sections += [f"S{i:02d}"] * n
        groups += [group] * n

    adata = ad.AnnData(X=np.vstack(blocks).astype(np.float32))
    adata.obsm["spatial"] = np.tile(coords, (n_sections, 1))
    adata.obs["section"], adata.obs["response"] = sections, groups
    adata.var_names = [f"m{j:03d}" for j in range(n_vars)]
    return adata


class TestIonImages:

    @staticmethod
    def _panels(fig):
        """Panel axes only. Gridded data renders via imshow (ax.images) and
        non-gridded via scatter (ax.collections); the shared colourbar is also
        an axes, so a title is what distinguishes a real panel."""
        return [
            ax for ax in fig.axes
            if ax.get_visible() and (ax.images or ax.collections) and ax.get_title()
        ]

    @staticmethod
    def _clim(ax):
        return (ax.images or ax.collections)[0].get_clim()

    def test_one_panel_per_section(self, organization_cohort):
        fig = mt.plot_ion_images(organization_cohort, "m000", sample_key="section")
        assert len(self._panels(fig)) == 8

    def test_grouping_orders_panels_into_blocks(self, organization_cohort):
        fig = mt.plot_ion_images(
            organization_cohort, "m000", sample_key="section", group_key="response"
        )
        arms = [ax.get_title().split("\n")[1] for ax in self._panels(fig)]
        assert arms == sorted(arms), "panels are interleaved rather than grouped"

    def test_shared_scale_uses_one_clim(self, organization_cohort):
        fig = mt.plot_ion_images(organization_cohort, "m000", sample_key="section")
        clims = {self._clim(ax) for ax in self._panels(fig)}
        assert len(clims) == 1, "shared_scale=True must give every panel the same limits"

    def test_per_panel_scale_differs(self, organization_cohort):
        fig = mt.plot_ion_images(
            organization_cohort, "m000", sample_key="section", shared_scale=False
        )
        clims = {self._clim(ax) for ax in self._panels(fig)}
        assert len(clims) > 1

    def test_percentile_clipping_excludes_outliers(self, organization_cohort):
        """A single hot pixel must not flatten the colour scale."""
        spiked = organization_cohort.copy()
        X = np.asarray(spiked.X).copy()
        X[0, 0] = 1e6
        spiked.X = X
        fig = mt.plot_ion_images(spiked, "m000", sample_key="section")
        _, vmax = self._clim(self._panels(fig)[0])
        assert vmax < 1e5, "colour scale was dominated by one outlier pixel"

    def test_rejects_bad_input(self, organization_cohort):
        with pytest.raises(InvalidParameterError, match="not found in adata.var_names"):
            mt.plot_ion_images(organization_cohort, "nope", sample_key="section")
        with pytest.raises(InvalidParameterError, match="section_missing"):
            mt.plot_ion_images(organization_cohort, "m000", sample_key="section_missing")


class TestOrganizationHeatmap:

    @pytest.fixture(scope="class")
    def org(self, organization_cohort):
        return mt.spatial_organization(
            organization_cohort, sample_key="section", metrics=("morans_i", "entropy")
        )

    def test_shape_and_labels(self, org):
        fig = mt.plot_organization_heatmap(org, top_n=6)
        ax = fig.axes[0]
        assert ax.images[0].get_array().shape == (org.n_obs, 6)

    def test_group_separator_drawn(self, org):
        fig = mt.plot_organization_heatmap(org, group_key="response", top_n=5)
        assert any(line.get_linewidth() > 1.0 for line in fig.axes[0].lines)

    def test_result_selects_the_claimed_metabolites(self, org):
        result = mt.differential_spatial_organization(org, "response", "R", "NR")
        fig = mt.plot_organization_heatmap(org, result=result, top_n=3)
        shown = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        assert result["metabolite"].iloc[0] in shown

    def test_rejects_missing_metric(self, org):
        with pytest.raises(InvalidParameterError, match="not in org.layers"):
            mt.plot_organization_heatmap(org, metric="gini")


class TestClassAndPathwayFigures:

    def test_class_enrichment_annotates_counts(self):
        report = pd.DataFrame({
            "chemical_class": ["Polyamine", "Amino acid", "Sterol"],
            "median_delta": [0.9, -0.6, 0.1],
            "n_compounds": [12, 30, 4],
            "pval_adj": [0.001, 0.02, 0.6],
        })
        ax = mt.plot_class_enrichment(report).axes[0]
        labels = [t.get_text() for t in ax.texts]
        assert "n=12" in labels and "n=30" in labels and "n=4" in labels

    def test_class_enrichment_rejects_empty(self):
        with pytest.raises(InvalidParameterError, match="empty"):
            mt.plot_class_enrichment(pd.DataFrame(
                columns=["chemical_class", "median_delta", "n_compounds", "pval_adj"]
            ))

    def test_pathway_dotplot_splits_directions(self):
        report = pd.DataFrame({
            "pathway": ["Arginine metabolism", "Glycolysis"] * 2,
            "direction": ["up", "up", "down", "down"],
            "n_hits": [8, 4, 6, 3],
            "pval_adj": [0.001, 0.04, 0.002, 0.3],
        })
        fig = mt.plot_pathway_dotplot(report)
        # Panel titles are set with loc="left", so get_title() (centre) is empty.
        titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_visible()]
        assert "up" in titles and "down" in titles

    def test_pathway_dotplot_single_direction(self):
        report = pd.DataFrame({
            "pathway": ["A", "B"], "direction": ["any", "any"],
            "n_hits": [5, 2], "pval_adj": [0.01, 0.2],
        })
        fig = mt.plot_pathway_dotplot(report, direction="any")
        # The colourbar is an axes too; count only ones carrying a panel title.
        panels = [ax for ax in fig.axes if ax.get_visible() and ax.get_title(loc="left")]
        assert len(panels) == 1

    def test_pathway_dotplot_rejects_empty(self):
        with pytest.raises(InvalidParameterError, match="empty"):
            mt.plot_pathway_dotplot(pd.DataFrame(
                columns=["pathway", "direction", "n_hits", "pval_adj"]
            ))


def test_new_figures_export_as_editable_pdf(tmp_path, organization_cohort):
    mt.set_publication_style()
    org = mt.spatial_organization(
        organization_cohort, sample_key="section", metrics=("morans_i",)
    )
    figures = {
        "ion_images": mt.plot_ion_images(
            organization_cohort, "m000", sample_key="section", group_key="response"
        ),
        "org_heatmap": mt.plot_organization_heatmap(org, group_key="response", top_n=6),
    }
    for name, fig in figures.items():
        out = mt.save_figure(fig, tmp_path / name, provenance={"figure": name}, close=True)["pdf"]
        raw = out.read_bytes()
        assert b"/FontFile2" in raw and b"/Type3" not in raw


class TestLabelCollisions:
    """
    Cliff's delta from a handful of samples per group is quantised, so the
    strongest findings routinely land on the exact same coordinate. Four
    organization-only compounds all at (0, 1) stacked into unreadable overlap
    before labels were spread.
    """

    def test_coincident_labels_do_not_overlap(self):
        merged = pd.DataFrame({
            "metabolite": ["Spermidine", "Putrescine", "Spermine", "Agmatine"],
            "delta_abundance": [0.0, 0.0, 0.0, 0.0],
            "delta_organization": [1.0, 1.0, 1.0, 1.0],
            "pval_adj_abundance": [0.9] * 4,
            "pval_adj_organization": [0.001] * 4,
            "classification": ["organization only"] * 4,
        })
        ax = mt.plot_abundance_vs_organization(merged, label_top=4).axes[0]
        annotations = [t for t in ax.texts if t.get_text() in set(merged["metabolite"])]
        assert len(annotations) == 4

        offsets = [t.get_position() for t in annotations]
        vertical = sorted(o[1] for o in offsets)
        assert len(set(vertical)) == 4, "labels share a vertical offset and will overlap"

    def test_distinct_points_keep_the_default_offset(self):
        merged = pd.DataFrame({
            "metabolite": ["A", "B"],
            "delta_abundance": [0.0, 0.8],
            "delta_organization": [1.0, -0.9],
            "pval_adj_abundance": [0.9, 0.001],
            "pval_adj_organization": [0.001, 0.001],
            "classification": ["organization only", "organization only"],
        })
        ax = mt.plot_abundance_vs_organization(merged, label_top=2).axes[0]
        labelled = [t for t in ax.texts if t.get_text() in {"A", "B"}]
        assert len({t.get_position()[1] for t in labelled}) == 1


class TestThemedFigures:
    """
    A figure with a baked-in white rectangle looks pasted onto a dark slide or
    web page rather than placed in it. The light/dark themes render on a
    transparent ground with the ink recoloured; print keeps the opaque white
    page a journal expects.
    """

    def test_print_theme_is_opaque_white(self):
        mt.set_publication_style(theme="print")
        assert matplotlib.rcParams["savefig.transparent"] is False
        assert matplotlib.rcParams["figure.facecolor"] == "white"
        assert matplotlib.rcParams["text.color"] == "black"

    @pytest.mark.parametrize("theme", ["light", "dark"])
    def test_screen_themes_are_transparent(self, theme):
        mt.set_publication_style(theme=theme)
        assert matplotlib.rcParams["savefig.transparent"] is True
        assert matplotlib.rcParams["figure.facecolor"] == "none"
        assert matplotlib.rcParams["axes.facecolor"] == "none"

    def test_dark_theme_ink_is_light(self):
        mt.set_publication_style(theme="dark")
        for key in ("text.color", "axes.labelcolor", "axes.edgecolor", "xtick.color"):
            assert matplotlib.rcParams[key] == "#d8d5cf"

    def test_data_colours_are_theme_independent(self, result):
        """A figure must stay recognisable across themes; only the ink moves."""
        def bar_colours(theme):
            mt.set_publication_style(theme=theme)
            ax = mt.plot_effect_size(result, top_n=6).axes[0]
            return [p.get_edgecolor() for p in ax.patches]

        assert bar_colours("print") == bar_colours("dark")

    def test_transparent_svg_has_no_opaque_page(self, tmp_path, result):
        mt.set_publication_style(theme="dark")
        fig = mt.plot_effect_size(result, top_n=5)
        svg = mt.save_figure(fig, tmp_path / "fig", formats=("svg",))["svg"].read_text()
        assert "#ffffff" not in svg.lower(), "an opaque white page was written into the SVG"

    def test_rejects_unknown_theme(self):
        with pytest.raises(InvalidParameterError, match="theme must be"):
            mt.set_publication_style(theme="solarized")

    def test_dotted_filenames_survive(self, tmp_path, result):
        """Regression: with_suffix('') ate everything after the last dot."""
        mt.set_publication_style()
        fig = mt.plot_effect_size(result, top_n=4)
        written = mt.save_figure(fig, tmp_path / "two_axis.dark", formats=("svg",))
        assert written["svg"].name == "two_axis.dark.svg"

    def test_known_extension_is_still_replaced(self, tmp_path, result):
        mt.set_publication_style()
        fig = mt.plot_effect_size(result, top_n=4)
        written = mt.save_figure(fig, tmp_path / "fig.pdf", formats=("svg",))
        assert written["svg"].name == "fig.svg"

    def test_dark_diverging_cmap_centres_on_the_ground(self):
        """
        RdBu passes through white at zero, so on a dark page every near-zero
        heatmap cell lights up as a white block and the values closest to
        "nothing here" become the loudest thing in the figure.
        """
        mt.set_publication_style(theme="dark")
        cmap = mt.diverging_cmap()
        assert cmap.name == "mortis_dark_diverging"
        centre = cmap(0.5)[:3]
        assert max(centre) < 0.2, f"midpoint {centre} is not dark"

    def test_print_theme_keeps_the_conventional_map(self):
        mt.set_publication_style(theme="print")
        assert mt.diverging_cmap().name == "RdBu_r"
        assert min(mt.diverging_cmap()(0.5)[:3]) > 0.9, "print midpoint should be near-white"

    def test_heatmap_uses_the_theme_map_by_default(self, organization_cohort):
        mt.set_publication_style(theme="dark")
        org = mt.spatial_organization(
            organization_cohort, sample_key="section", metrics=("morans_i",)
        )
        image = mt.plot_organization_heatmap(org, top_n=5).axes[0].images[0]
        assert image.get_cmap().name == "mortis_dark_diverging"

    def test_explicit_cmap_still_wins(self, organization_cohort):
        mt.set_publication_style(theme="dark")
        org = mt.spatial_organization(
            organization_cohort, sample_key="section", metrics=("morans_i",)
        )
        image = mt.plot_organization_heatmap(org, top_n=5, cmap="viridis").axes[0].images[0]
        assert image.get_cmap().name == "viridis"
