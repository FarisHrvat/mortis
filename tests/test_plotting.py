"""
Tests for mortis.plotting, all plot functions produce Figure objects,
accept style parameters, and save to disk correctly.
"""

from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mortis.analysis import cluster, compare_groups, find_markers, spatial_de
from mortis.plotting import (
    plot_cluster_composition,
    plot_embedding_grid,
    plot_heatmap,
    plot_markers,
    plot_morans,
    plot_qc,
    plot_spatial,
    plot_spatial_metabolite,
    plot_umap,
    plot_violin,
    plot_volcano,
)
from mortis.preprocessing import preprocess, run_umap

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def full_adata():
    """Preprocessed, clustered AnnData with UMAP and condition labels."""
    rng = np.random.default_rng(42)
    n = 100
    X = rng.random((n, 30)).astype(np.float32)
    X[:50, :15] += 1.5
    X[50:, 15:] += 1.5
    obs = pd.DataFrame({
        "x": np.tile(np.arange(10), 10),
        "y": np.repeat(np.arange(10), 10),
        "condition": ["A"] * 50 + ["B"] * 50,
    })
    obs.index = [f"{r['x']}_{r['y']}" for _, r in obs.iterrows()]
    var = pd.DataFrame(index=[f"met_{i}" for i in range(30)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    adata.uns["preprocessed_steps"] = []
    adata = preprocess(adata, n_pcs=10, scale_data=True)
    adata = run_umap(adata)
    adata = cluster(adata, resolution=0.5)
    return adata


@pytest.fixture
def qc_stats():
    rng = np.random.default_rng(0)
    fc = rng.exponential(scale=1.5, size=200).astype(np.float32)
    mask = fc >= 1.5
    return {
        "fold_change": fc,
        "cutoff": 1.5,
        "mean_tissue": fc * 2,
        "mean_bg": fc * 0.5,
        "keep_mask": mask,
        "n_kept": int(mask.sum()),
        "n_removed": int((~mask).sum()),
    }


# ---------------------------------------------------------------------------
# plot_qc
# ---------------------------------------------------------------------------

class TestPlotQc:
    def test_returns_figure(self, qc_stats, full_adata):
        fig = plot_qc(qc_stats, full_adata, sample_name="Test")
        assert isinstance(fig, plt.Figure)

    def test_custom_figsize(self, qc_stats, full_adata):
        fig = plot_qc(qc_stats, full_adata, figsize=(16, 12))
        assert fig.get_size_inches()[0] == pytest.approx(16.0)

    def test_custom_dpi(self, qc_stats, full_adata):
        fig = plot_qc(qc_stats, full_adata, dpi=72)
        assert fig.get_dpi() == pytest.approx(72.0)

    def test_saves_pdf(self, qc_stats, full_adata, tmp_path):
        out = str(tmp_path / "qc.pdf")
        plot_qc(qc_stats, full_adata, save=out)
        assert Path(out).exists()

    def test_saves_png(self, qc_stats, full_adata, tmp_path):
        out = str(tmp_path / "qc.png")
        plot_qc(qc_stats, full_adata, save=out)
        assert Path(out).exists()

    def test_saves_svg(self, qc_stats, full_adata, tmp_path):
        out = str(tmp_path / "qc.svg")
        plot_qc(qc_stats, full_adata, save=out)
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# plot_spatial
# ---------------------------------------------------------------------------

class TestPlotSpatial:
    def test_categorical_color(self, full_adata):
        fig = plot_spatial(full_adata, color="cluster")
        assert isinstance(fig, plt.Figure)

    def test_continuous_color_obs(self, full_adata):
        full_adata.obs["score"] = np.random.rand(full_adata.n_obs)
        fig = plot_spatial(full_adata, color="score")
        assert isinstance(fig, plt.Figure)

    def test_metabolite_color(self, full_adata):
        fig = plot_spatial(full_adata, color="met_0")
        assert isinstance(fig, plt.Figure)

    def test_invalid_color_raises(self, full_adata):
        with pytest.raises(KeyError):
            plot_spatial(full_adata, color="nonexistent_column")

    def test_saves_tiff(self, full_adata, tmp_path):
        out = str(tmp_path / "spatial.tiff")
        plot_spatial(full_adata, color="cluster", save=out)
        assert Path(out).exists()

    def test_custom_style_params(self, full_adata):
        fig = plot_spatial(full_adata, color="cluster",
                           figsize=(10, 8), dpi=100, fontsize=14,
                           show_grid=True, show_axes_border=False)
        assert isinstance(fig, plt.Figure)
        assert fig.get_size_inches()[0] == pytest.approx(10.0)

    def test_s_kwarg_sets_marker_size(self, full_adata):
        # matplotlib-native spelling
        fig = plot_spatial(full_adata, color="cluster", s=9)
        assert isinstance(fig, plt.Figure)

    def test_spot_size_kwarg_sets_marker_size(self, full_adata):
        # descriptive alias, regression test: this used to crash with
        # "PathCollection.set() got an unexpected keyword argument 'spot_size'"
        # because spot_size fell through to ax.scatter(**kwargs) unpopped.
        fig = plot_spatial(full_adata, color="cluster", spot_size=9)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_spatial_metabolite
# ---------------------------------------------------------------------------

class TestPlotSpatialMetabolite:
    def test_returns_figure(self, full_adata):
        fig = plot_spatial_metabolite(full_adata, "met_0")
        assert isinstance(fig, plt.Figure)

    def test_invalid_metabolite_raises(self, full_adata):
        with pytest.raises(KeyError, match="not found"):
            plot_spatial_metabolite(full_adata, "nonexistent_met")


# ---------------------------------------------------------------------------
# plot_embedding_grid
# ---------------------------------------------------------------------------

class TestPlotEmbeddingGrid:
    def test_returns_figure(self, full_adata):
        fig = plot_embedding_grid(full_adata, ["met_0", "met_1", "met_2"], ncols=3)
        assert isinstance(fig, plt.Figure)

    def test_skips_missing_metabolites(self, full_adata):
        with pytest.warns(UserWarning, match="not in this object"):
            plot_embedding_grid(full_adata, ["met_0", "fake_met"], ncols=2)

    def test_saves_pdf(self, full_adata, tmp_path):
        out = str(tmp_path / "grid.pdf")
        plot_embedding_grid(full_adata, ["met_0", "met_1"], save=out)
        assert Path(out).exists()

    def test_spot_size_kwarg(self, full_adata):
        fig = plot_embedding_grid(full_adata, ["met_0", "met_1"], spot_size=6)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_umap
# ---------------------------------------------------------------------------

class TestPlotUmap:
    def test_categorical(self, full_adata):
        fig = plot_umap(full_adata, color="cluster")
        assert isinstance(fig, plt.Figure)

    def test_continuous(self, full_adata):
        fig = plot_umap(full_adata, color="met_0")
        assert isinstance(fig, plt.Figure)

    def test_raises_without_umap(self):
        obs = pd.DataFrame({"x": [0], "y": [0]})
        adata = ad.AnnData(X=np.ones((1, 3), dtype=np.float32), obs=obs)
        from mortis.exceptions import NoEmbeddingError
        with pytest.raises(NoEmbeddingError):
            plot_umap(adata)

    def test_saves_svg(self, full_adata, tmp_path):
        out = str(tmp_path / "umap.svg")
        plot_umap(full_adata, color="cluster", save=out)
        assert Path(out).exists()

    def test_spot_size_kwarg(self, full_adata):
        fig = plot_umap(full_adata, color="cluster", spot_size=6)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_markers
# ---------------------------------------------------------------------------

class TestPlotMarkers:
    def test_returns_figure(self, full_adata):
        adata, _ = find_markers(full_adata.copy(), n_top=3)
        fig = plot_markers(adata, n_top=3)
        assert isinstance(fig, plt.Figure)

    def test_raises_without_markers(self, full_adata):
        from mortis.exceptions import NoClustersError
        with pytest.raises(NoClustersError):
            plot_markers(full_adata)


# ---------------------------------------------------------------------------
# plot_volcano
# ---------------------------------------------------------------------------

class TestPlotVolcano:
    def test_returns_figure(self, full_adata):
        _, results = compare_groups(full_adata, "condition", "A", "B")
        fig = plot_volcano(results, group1="A", group2="B")
        assert isinstance(fig, plt.Figure)

    def test_with_table(self, full_adata):
        _, results = compare_groups(full_adata, "condition", "A", "B")
        fig = plot_volcano(results, show_table=True, n_table=3)
        assert isinstance(fig, plt.Figure)

    def test_saves_pdf(self, full_adata, tmp_path):
        _, results = compare_groups(full_adata, "condition", "A", "B")
        out = str(tmp_path / "volcano.pdf")
        plot_volcano(results, save=out)
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# plot_heatmap
# ---------------------------------------------------------------------------

class TestPlotHeatmap:
    def test_returns_figure(self, full_adata):
        mets = [f"met_{i}" for i in range(10)]
        fig = plot_heatmap(full_adata, mets, groupby="cluster")
        assert isinstance(fig, plt.Figure)

    def test_invalid_groupby_raises(self, full_adata):
        from mortis.exceptions import InvalidParameterError
        with pytest.raises(InvalidParameterError):
            plot_heatmap(full_adata, ["met_0"], groupby="nonexistent")

    def test_saves_png(self, full_adata, tmp_path):
        out = str(tmp_path / "heatmap.png")
        plot_heatmap(full_adata, ["met_0", "met_1"], save=out)
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# plot_violin
# ---------------------------------------------------------------------------

class TestPlotViolin:
    def test_returns_figure(self, full_adata):
        fig = plot_violin(full_adata, ["met_0", "met_1"], groupby="condition")
        assert isinstance(fig, plt.Figure)

    def test_single_metabolite(self, full_adata):
        fig = plot_violin(full_adata, ["met_0"])
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_morans
# ---------------------------------------------------------------------------

class TestPlotMorans:
    def test_returns_figure(self, full_adata):
        _, morans = spatial_de(full_adata.copy(), n_top=10)
        if len(morans) == 0:
            # If no significant SVGs, use the full morans df
            from mortis.analysis import spatial_autocorrelation
            _, morans = spatial_autocorrelation(full_adata.copy(), n_neighbors=4)
        fig = plot_morans(morans, n_top=min(10, len(morans)))
        assert isinstance(fig, plt.Figure)

    def test_saves_pdf(self, full_adata, tmp_path):
        from mortis.analysis import spatial_autocorrelation
        _, morans = spatial_autocorrelation(full_adata.copy(), n_neighbors=4)
        out = str(tmp_path / "morans.pdf")
        plot_morans(morans, n_top=5, save=out)
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# plot_cluster_composition
# ---------------------------------------------------------------------------

class TestPlotClusterComposition:
    def test_returns_figure(self, full_adata):
        fig = plot_cluster_composition(full_adata, cluster_key="cluster",
                                       groupby="condition")
        assert isinstance(fig, plt.Figure)

    def test_unnormalized(self, full_adata):
        fig = plot_cluster_composition(full_adata, normalize=False)
        assert isinstance(fig, plt.Figure)

    def test_raises_without_clusters(self, full_adata):
        from mortis.exceptions import NoClustersError
        with pytest.raises(NoClustersError):
            plot_cluster_composition(full_adata, cluster_key="nonexistent")

    def test_saves_svg(self, full_adata, tmp_path):
        out = str(tmp_path / "composition.svg")
        plot_cluster_composition(full_adata, save=out)
        assert Path(out).exists()

