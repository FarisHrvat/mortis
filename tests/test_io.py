"""
Tests for mortis.io, loading, pairing, validation, and saving.
"""


import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mortis.exceptions import FileFormatError, InvalidParameterError, MissingROIError
from mortis.io import (
    check_rois,
    from_dataframe,
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
        p = tmp_path / "data.mzml"
        p.write_text("dummy")
        with pytest.raises(FileFormatError, match="Unsupported file extension"):
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



class TestDelimiterSniffing:
    """
    Facilities do not agree on a delimiter. SCiLS writes tabs, a European
    Excel writes semicolons and a comma decimal mark, METASPACE writes commas.
    The reader works it out instead of making the user re-export.
    """

    def _frame(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(np.abs(rng.normal(50, 10, (12, 3))).round(3),
                             columns=["Taurine", "Creatine", "Choline"])
        frame.insert(0, "y", np.tile(np.arange(4), 3))
        frame.insert(0, "x", np.repeat(np.arange(3), 4))
        return frame

    @pytest.mark.parametrize("sep,name", [
        (",", "comma.csv"), (";", "semi.csv"), ("\t", "tab.txt"),
        ("|", "pipe.txt"), ("\t", "tabbed.tsv"),
    ])
    def test_reads_whatever_the_delimiter_is(self, tmp_path, sep, name):
        path = tmp_path / name
        self._frame().to_csv(path, index=False, sep=sep)
        adata = read_metabolomics_data(str(path))
        assert adata.shape == (12, 3)
        assert list(adata.var_names) == ["Taurine", "Creatine", "Choline"]

    def test_european_decimal_comma_parses_as_numbers(self, tmp_path):
        path = tmp_path / "euro.csv"
        self._frame().to_csv(path, index=False, sep=";", decimal=",")
        adata = read_metabolomics_data(str(path))
        assert adata.X.dtype == np.float32
        assert adata.X.sum() > 0          # not parsed as text and zeroed

    def test_row_and_column_are_accepted_as_coordinates(self, tmp_path):
        path = tmp_path / "rc.csv"
        self._frame().rename(columns={"x": "Row", "y": "Column"}).to_csv(path, index=False)
        adata = read_metabolomics_data(str(path))
        assert "spatial" in adata.obsm

    def test_unreadable_delimiter_says_what_was_tried(self, tmp_path):
        path = tmp_path / "odd.csv"
        path.write_text("x~y~Taurine\n1~2~3\n", encoding="utf-8")
        with pytest.raises(FileFormatError, match="comma, semicolon, tab and pipe"):
            read_metabolomics_data(str(path))


class TestFromDataFrame:
    """The escape hatch for a format MORTIS does not read."""

    def test_builds_an_object_from_a_frame(self):
        frame = pd.DataFrame({"x": [0, 1], "y": [0, 0], "Taurine": [1.0, 2.0]})
        adata = from_dataframe(frame)
        assert adata.shape == (2, 1)
        assert "spatial" in adata.obsm

    def test_custom_coordinate_columns(self):
        frame = pd.DataFrame({"Row": [0, 1], "Col": [0, 0], "Taurine": [1.0, 2.0]})
        adata = from_dataframe(frame, x="Row", y="Col")
        assert adata.n_obs == 2

    def test_missing_coordinates_names_the_columns_present(self):
        frame = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(InvalidParameterError, match="Coordinate column"):
            from_dataframe(frame)

    def test_no_compounds_is_refused(self):
        frame = pd.DataFrame({"x": [0], "y": [0]})
        with pytest.raises(InvalidParameterError, match="no intensities"):
            from_dataframe(frame)
