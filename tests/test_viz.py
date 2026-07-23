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
