"""
Tests for mortis.image. TIFF loading, alignment, feature extraction,
and overlay plotting.
"""

from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mortis.exceptions import InvalidParameterError, MissingSpatialError
from mortis.image import (
    align_image,
    extract_image_features,
    load_image,
    plot_image_overlay,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


@pytest.fixture
def spatial_adata():
    n = 20
    obs = pd.DataFrame({
        "x": np.tile(np.arange(5), 4).astype(float),
        "y": np.repeat(np.arange(4), 5).astype(float),
    })
    obs.index = [f"{r['x']}_{r['y']}" for _, r in obs.iterrows()]
    adata = ad.AnnData(
        X=np.random.rand(n, 5).astype(np.float32),
        obs=obs,
    )
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    return adata


@pytest.fixture
def grayscale_tiff(tmp_path):
    """Write a small grayscale TIFF."""
    import tifffile
    img = (np.random.rand(50, 50) * 65535).astype(np.uint16)
    p = tmp_path / "gray.tif"
    tifffile.imwrite(str(p), img)
    return str(p)


@pytest.fixture
def rgb_tiff(tmp_path):
    """Write a small RGB TIFF."""
    import tifffile
    img = (np.random.rand(50, 50, 3) * 255).astype(np.uint8)
    p = tmp_path / "rgb.tif"
    tifffile.imwrite(str(p), img)
    return str(p)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_loads_grayscale(self, spatial_adata, grayscale_tiff):
        result = load_image(spatial_adata, grayscale_tiff)
        assert "histology" in result.uns
        assert result.uns["histology"]["data"].dtype == np.float32
        assert result.uns["histology"]["data"].max() <= 1.0

    def test_loads_rgb(self, spatial_adata, rgb_tiff):
        result = load_image(spatial_adata, rgb_tiff)
        assert result.uns["histology"]["shape"][2] == 3

    def test_custom_key(self, spatial_adata, grayscale_tiff):
        result = load_image(spatial_adata, grayscale_tiff, image_key="he_stain")
        assert "he_stain" in result.uns

    def test_missing_file_raises(self, spatial_adata):
        with pytest.raises(FileNotFoundError):
            load_image(spatial_adata, "/nonexistent/image.tif")

    def test_copy_flag(self, spatial_adata, grayscale_tiff):
        original_uns = set(spatial_adata.uns.keys())
        load_image(spatial_adata, grayscale_tiff, copy=True)
        assert set(spatial_adata.uns.keys()) == original_uns


# ---------------------------------------------------------------------------
# align_image
# ---------------------------------------------------------------------------

class TestAlignImage:
    def test_alignment_stored(self, spatial_adata, grayscale_tiff):
        adata = load_image(spatial_adata, grayscale_tiff)
        result = align_image(adata, scale_x=2.0, scale_y=2.0, offset_x=5.0)
        assert result.uns["histology"]["scale_x"] == 2.0
        assert result.uns["histology"]["offset_x"] == 5.0
        assert result.uns["histology"]["aligned"] is True

    def test_raises_without_image(self, spatial_adata):
        with pytest.raises(InvalidParameterError, match="not found in adata.uns"):
            align_image(spatial_adata, image_key="nonexistent")


# ---------------------------------------------------------------------------
# extract_image_features
# ---------------------------------------------------------------------------

class TestExtractImageFeatures:
    def test_grayscale_feature_added(self, spatial_adata, grayscale_tiff):
        adata = load_image(spatial_adata, grayscale_tiff)
        result = extract_image_features(adata, radius=2)
        assert "histology_intensity" in result.obs.columns
        assert result.obs["histology_intensity"].shape[0] == spatial_adata.n_obs

    def test_rgb_features_added(self, spatial_adata, rgb_tiff):
        adata = load_image(spatial_adata, rgb_tiff)
        result = extract_image_features(adata, radius=2)
        for ch in ("R", "G", "B"):
            assert f"histology_{ch}" in result.obs.columns

    def test_raises_without_spatial(self, grayscale_tiff):
        obs = pd.DataFrame({"x": [0], "y": [0]})
        adata = ad.AnnData(X=np.ones((1, 2), dtype=np.float32), obs=obs)
        adata = load_image(adata, grayscale_tiff)
        with pytest.raises(MissingSpatialError):
            extract_image_features(adata)

    def test_raises_without_image(self, spatial_adata):
        with pytest.raises(InvalidParameterError):
            extract_image_features(spatial_adata)


# ---------------------------------------------------------------------------
# plot_image_overlay
# ---------------------------------------------------------------------------

class TestPlotImageOverlay:
    def test_returns_figure(self, spatial_adata, grayscale_tiff):
        adata = load_image(spatial_adata, grayscale_tiff)
        fig = plot_image_overlay(adata)
        assert isinstance(fig, plt.Figure)

    def test_with_color(self, spatial_adata, rgb_tiff):
        adata = load_image(spatial_adata, rgb_tiff)
        adata.obs["score"] = np.random.rand(spatial_adata.n_obs)
        fig = plot_image_overlay(adata, color="score")
        assert isinstance(fig, plt.Figure)

    def test_saves_png(self, spatial_adata, grayscale_tiff, tmp_path):
        adata = load_image(spatial_adata, grayscale_tiff)
        out = str(tmp_path / "overlay.png")
        plot_image_overlay(adata, save=out)
        assert Path(out).exists()

    def test_raises_without_image(self, spatial_adata):
        with pytest.raises(InvalidParameterError):
            plot_image_overlay(spatial_adata)

