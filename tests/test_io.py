"""
Tests for mortis.io, loading, pairing, validation, and saving.
"""


import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mortis.exceptions import FileFormatError, MissingROIError
from mortis.io import (
    check_rois,
    load_from_folder,
    read_metabolomics_data,
    save_spatial_data,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n_pixels: int = 10, n_metabolites: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "x": np.arange(n_pixels),
        "y": np.zeros(n_pixels, dtype=int),
    })
    for i in range(n_metabolites):
        df[f"met_{i}"] = rng.random(n_pixels).astype(np.float32)
    return df


@pytest.fixture
def tmp_csv(tmp_path):
    df = _make_df()
    p = tmp_path / "tissue.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def tmp_xlsx(tmp_path):
    df = _make_df()
    p = tmp_path / "tissue.xlsx"
    df.to_excel(p, index=False)
    return p


@pytest.fixture
def tmp_h5ad(tmp_path):
    df = _make_df()
    obs = df[["x", "y"]].copy()
    obs.index = obs["x"].astype(str) + "_" + obs["y"].astype(str)
    X = df.drop(columns=["x", "y"]).to_numpy(dtype=np.float32)
    var = pd.DataFrame(index=[f"met_{i}" for i in range(5)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    p = tmp_path / "sample.h5ad"
    adata.write_h5ad(p)
    return p


@pytest.fixture
def paired_folder(tmp_path):
    """Folder with a paired tissue + background xlsx."""
    df_t = _make_df(n_pixels=20)
    df_b = _make_df(n_pixels=10)
    df_t.to_excel(tmp_path / "sample_tissue.xlsx", index=False)
    df_b.to_excel(tmp_path / "sample_background.xlsx", index=False)
    return tmp_path


# ---------------------------------------------------------------------------
# read_metabolomics_data
# ---------------------------------------------------------------------------

class TestReadMetabolomicsData:
    def test_csv_loads_correctly(self, tmp_csv):
        adata = read_metabolomics_data(str(tmp_csv))
        assert adata.n_obs == 10
        assert adata.n_vars == 5
        assert "spatial" in adata.obsm
        assert adata.obsm["spatial"].shape == (10, 2)

    def test_xlsx_loads_correctly(self, tmp_xlsx):
        adata = read_metabolomics_data(str(tmp_xlsx))
        assert adata.n_obs == 10
        assert adata.n_vars == 5

    def test_h5ad_loads_correctly(self, tmp_h5ad):
        adata = read_metabolomics_data(str(tmp_h5ad))
        assert adata.n_obs == 10
        assert "spatial" in adata.obsm

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            read_metabolomics_data(str(tmp_path / "nonexistent.csv"))

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.write_text("dummy")
        with pytest.raises(FileFormatError, match="Unsupported"):
            read_metabolomics_data(str(p))

    def test_missing_xy_columns_raises(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        p = tmp_path / "bad.csv"
        df.to_csv(p, index=False)
        with pytest.raises(FileFormatError, match="missing required column"):
            read_metabolomics_data(str(p))

    def test_no_metabolite_columns_raises(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = tmp_path / "no_mets.csv"
        df.to_csv(p, index=False)
        with pytest.raises(FileFormatError, match="no metabolite"):
            read_metabolomics_data(str(p))

    def test_x_dtype_is_float32(self, tmp_csv):
        adata = read_metabolomics_data(str(tmp_csv))
        assert adata.X.dtype == np.float32


# ---------------------------------------------------------------------------
# load_from_folder
# ---------------------------------------------------------------------------

class TestLoadFromFolder:
    def test_paired_files_merged(self, paired_folder):
        adatas = load_from_folder(str(paired_folder))
        assert len(adatas) == 1
        adata = adatas[0]
        assert "is_tissue" in adata.obs.columns
        assert "is_background" in adata.obs.columns
        assert adata.obs["is_tissue"].sum() == 20
        assert adata.obs["is_background"].sum() == 10

    def test_h5ad_loaded(self, tmp_h5ad):
        adatas = load_from_folder(str(tmp_h5ad.parent))
        assert len(adatas) == 1

    def test_empty_folder_returns_empty_list(self, tmp_path):
        result = load_from_folder(str(tmp_path))
        assert result == []

    def test_nonexistent_folder_raises(self):
        with pytest.raises(FileNotFoundError, match="Folder not found"):
            load_from_folder("/nonexistent/path/xyz")


# ---------------------------------------------------------------------------
# check_rois
# ---------------------------------------------------------------------------

class TestCheckRois:
    def test_passes_when_rois_present(self):
        obs = pd.DataFrame({
            "x": [0, 1], "y": [0, 0],
            "is_tissue": [True, False],
            "is_background": [False, True],
        })
        adata = ad.AnnData(X=np.ones((2, 3), dtype=np.float32), obs=obs)
        check_rois([adata])  # should not raise

    def test_raises_when_rois_missing(self):
        obs = pd.DataFrame({"x": [0], "y": [0]})
        adata = ad.AnnData(X=np.ones((1, 3), dtype=np.float32), obs=obs)
        with pytest.raises(MissingROIError, match="missing ROI labels"):
            check_rois([adata])

    def test_raises_with_index_info(self):
        obs_ok = pd.DataFrame({
            "x": [0], "y": [0],
            "is_tissue": [True], "is_background": [False],
        })
        obs_bad = pd.DataFrame({"x": [1], "y": [0]})
        a_ok = ad.AnnData(X=np.ones((1, 2), dtype=np.float32), obs=obs_ok)
        a_bad = ad.AnnData(X=np.ones((1, 2), dtype=np.float32), obs=obs_bad)
        with pytest.raises(MissingROIError, match="indices: \\[1\\]"):
            check_rois([a_ok, a_bad])


# ---------------------------------------------------------------------------
# save_spatial_data
# ---------------------------------------------------------------------------

class TestSaveSpatialData:
    def test_saves_h5ad_files(self, tmp_path):
        obs = pd.DataFrame({"x": [0, 1], "y": [0, 0]})
        obs.index = ["0_0", "1_0"]
        adata = ad.AnnData(X=np.ones((2, 3), dtype=np.float32), obs=obs)
        adata.obsm["spatial"] = np.array([[0, 0], [1, 0]], dtype=np.float32)
        paths = save_spatial_data([adata], output_dir=str(tmp_path), prefix="test")
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".h5ad"

    def test_creates_output_dir(self, tmp_path):
        new_dir = tmp_path / "new_subdir"
        obs = pd.DataFrame({"x": [0], "y": [0]})
        obs.index = ["0_0"]
        adata = ad.AnnData(X=np.ones((1, 2), dtype=np.float32), obs=obs)
        adata.obsm["spatial"] = np.array([[0, 0]], dtype=np.float32)
        save_spatial_data([adata], output_dir=str(new_dir))
        assert new_dir.exists()

