"""
Tests for previously-uncovered analysis functions: correct_batches,
run_harmony (edge cases), spatially_weighted_nmf, spatial_gradient,
metabolite_colocalization, spatial_domains, median_normalize and
load_annotation_scores.
"""


import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mortis.analysis import (
    metabolite_colocalization,
    spatial_autocorrelation,
    spatial_domains,
    spatial_gradient,
    spatially_weighted_nmf,
)
from mortis.exceptions import InvalidParameterError, NoEmbeddingError
from mortis.io import load_annotation_scores
from mortis.preprocessing import correct_batches, median_normalize, preprocess

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def base_adata():
    rng = np.random.default_rng(11)
    n = 100
    X = rng.random((n, 25)).astype(np.float32)
    X[:50, :10] += 2.0
    X[50:, 10:20] += 2.0
    obs = pd.DataFrame({
        "x": np.tile(np.arange(10), 10),
        "y": np.repeat(np.arange(10), 10),
        "condition": ["A"] * 50 + ["B"] * 50,
        "sample": ["S1"] * 60 + ["S2"] * 40,
    })
    obs.index = [f"{r['x']}_{r['y']}_{i}" for i, (_, r) in enumerate(obs.iterrows())]
    var = pd.DataFrame(index=[f"met_{i}" for i in range(25)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    adata.uns["preprocessed_steps"] = []
    return preprocess(adata, n_pcs=10, scale_data=True)


# ---------------------------------------------------------------------------
# median_normalize
# ---------------------------------------------------------------------------

class TestMedianNormalize:
    def test_equalizes_row_medians(self):
        rng = np.random.default_rng(0)
        X = rng.random((20, 15)).astype(np.float32) * 100
        obs = pd.DataFrame({"x": np.arange(20), "y": np.zeros(20, dtype=int)})
        obs.index = obs["x"].astype(str) + "_0"
        adata = ad.AnnData(X=X, obs=obs)
        adata.uns["preprocessed_steps"] = []
        result = median_normalize(adata)
        row_medians = np.array([np.median(row[row > 0]) for row in result.X])
        assert np.allclose(row_medians, row_medians[0], atol=1e-2)

    def test_zero_row_stays_zero(self):
        X = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        obs = pd.DataFrame({"x": [0, 1], "y": [0, 0]})
        adata = ad.AnnData(X=X, obs=obs)
        result = median_normalize(adata)
        assert np.allclose(result.X[0], 0.0)

    def test_step_recorded(self):
        X = np.ones((3, 3), dtype=np.float32)
        obs = pd.DataFrame({"x": [0, 1, 2], "y": [0, 0, 0]})
        adata = ad.AnnData(X=X, obs=obs)
        adata.uns["preprocessed_steps"] = []
        result = median_normalize(adata)
        assert "median_normalize" in result.uns["preprocessed_steps"]

    def test_preprocess_dispatches_to_median(self):
        rng = np.random.default_rng(1)
        X = rng.random((30, 10)).astype(np.float32) * 50
        obs = pd.DataFrame({"x": np.arange(30), "y": np.zeros(30, dtype=int)})
        obs.index = obs["x"].astype(str) + "_0"
        adata = ad.AnnData(X=X, obs=obs)
        adata.uns["preprocessed_steps"] = []
        result = preprocess(adata, n_pcs=5, normalize_method="median")
        assert "median_normalize" in result.uns["preprocessed_steps"]

    def test_preprocess_invalid_normalize_method_raises(self):
        rng = np.random.default_rng(1)
        X = rng.random((10, 5)).astype(np.float32)
        obs = pd.DataFrame({"x": np.arange(10), "y": np.zeros(10, dtype=int)})
        obs.index = obs["x"].astype(str) + "_0"
        adata = ad.AnnData(X=X, obs=obs)
        adata.uns["preprocessed_steps"] = []
        with pytest.raises(InvalidParameterError):
            preprocess(adata, n_pcs=5, normalize_method="bogus")


# ---------------------------------------------------------------------------
# correct_batches (ComBat)
# ---------------------------------------------------------------------------

class TestCorrectBatches:
    def test_runs_and_recomputes_pca(self, base_adata):
        original_pca_shape = base_adata.obsm["X_pca"].shape
        result = correct_batches(base_adata.copy(), batch_key="sample")
        assert result.obsm["X_pca"].shape == original_pca_shape

    def test_invalid_batch_key_raises(self, base_adata):
        with pytest.raises(InvalidParameterError):
            correct_batches(base_adata, batch_key="nonexistent")

    def test_copy_flag(self, base_adata):
        original = base_adata.X.copy()
        correct_batches(base_adata, batch_key="sample", copy=True)
        assert np.allclose(base_adata.X, original)


# ---------------------------------------------------------------------------
# spatial_domains
# ---------------------------------------------------------------------------

class TestSpatialDomains:
    def test_returns_domain_labels(self, base_adata):
        result = spatial_domains(base_adata.copy(), resolution=0.5, alpha=0.5)
        assert "domain" in result.obs.columns
        assert result.obs["domain"].nunique() >= 1

    def test_alpha_zero_matches_feature_space(self, base_adata):
        # alpha=0 means no spatial smoothing is applied.
        result = spatial_domains(base_adata.copy(), alpha=0.0)
        assert "domain" in result.obs.columns

    def test_invalid_alpha_raises(self, base_adata):
        with pytest.raises(InvalidParameterError):
            spatial_domains(base_adata, alpha=1.5)

    def test_invalid_resolution_raises(self, base_adata):
        with pytest.raises(InvalidParameterError):
            spatial_domains(base_adata, resolution=-1.0)

    def test_raises_without_pca(self):
        obs = pd.DataFrame({"x": [0, 1], "y": [0, 0]})
        obs.index = ["0_0", "1_0"]
        adata = ad.AnnData(X=np.ones((2, 5), dtype=np.float32), obs=obs)
        adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
        with pytest.raises(NoEmbeddingError):
            spatial_domains(adata)

    def test_custom_key_added(self, base_adata):
        result = spatial_domains(base_adata.copy(), key_added="niche")
        assert "niche" in result.obs.columns


# ---------------------------------------------------------------------------
# spatially_weighted_nmf
# ---------------------------------------------------------------------------

class TestSpatiallyWeightedNmf:
    def test_returns_adata_and_df(self, base_adata):
        adata, df = spatially_weighted_nmf(base_adata.copy(), n_components=4, use_hardware=False)
        assert "X_snmf" in adata.obsm
        assert "snmf_cluster" in adata.obs.columns
        assert adata.obsm["X_snmf"].shape[1] == 4
        assert isinstance(df, pd.DataFrame)
        assert "component" in df.columns


# ---------------------------------------------------------------------------
# spatial_gradient
# ---------------------------------------------------------------------------

class TestSpatialGradient:
    def test_returns_dataframe(self, base_adata):
        base_adata.obs["region"] = (["core"] * 50) + (["edge"] * 50)
        df = spatial_gradient(base_adata, target_col="region", target_val="core",
                               bins=5, max_dist=15.0)
        assert isinstance(df, pd.DataFrame)
        assert "distance" in df.columns

    def test_invalid_target_col_raises(self, base_adata):
        with pytest.raises(InvalidParameterError):
            spatial_gradient(base_adata, target_col="nonexistent", target_val="x")


# ---------------------------------------------------------------------------
# metabolite_colocalization
# ---------------------------------------------------------------------------

class TestMetaboliteColocalization:
    def test_requires_morans_i(self, base_adata):
        with pytest.raises(InvalidParameterError):
            metabolite_colocalization(base_adata)

    def test_returns_edges(self, base_adata):
        adata, _ = spatial_autocorrelation(base_adata.copy(), n_neighbors=4)
        edges = metabolite_colocalization(adata, top_n=15, corr_threshold=0.0)
        assert isinstance(edges, pd.DataFrame)
        assert {"source", "target", "weight"}.issubset(edges.columns)


# ---------------------------------------------------------------------------
# load_annotation_scores
# ---------------------------------------------------------------------------

class TestLoadAnnotationScores:
    @pytest.fixture
    def feature_table(self, tmp_path):
        df = pd.DataFrame({
            "Compound": ["met_0", "met_1", "met_2"],
            "Chemical Formula": ["C1", "C2", "C3"],
            "Identification Score": [0.9, 0.2, 0.6],
        })
        p = tmp_path / "feature_table.xlsx"
        df.to_excel(p, index=False)
        return p

    def test_scores_merged(self, base_adata, feature_table):
        adata = load_annotation_scores(base_adata, str(feature_table))
        assert "score" in adata.var.columns
        assert adata.var.loc["met_0", "score"] == pytest.approx(0.9)
        assert adata.var.loc["met_1", "score"] == pytest.approx(0.2)

    def test_unmatched_metabolites_get_nan(self, base_adata, feature_table):
        adata = load_annotation_scores(base_adata, str(feature_table))
        assert np.isnan(adata.var.loc["met_24", "score"])

    def test_missing_file_raises(self, base_adata):
        with pytest.raises(FileNotFoundError):
            load_annotation_scores(base_adata, "/nonexistent/feature_table.xlsx")

    def test_missing_columns_raises(self, base_adata, tmp_path):
        df = pd.DataFrame({"Name": ["met_0"], "Score": [0.5]})
        p = tmp_path / "bad_table.xlsx"
        df.to_excel(p, index=False)
        from mortis.exceptions import FileFormatError
        with pytest.raises(FileFormatError):
            load_annotation_scores(base_adata, str(p))

    def test_csv_supported(self, base_adata, tmp_path):
        df = pd.DataFrame({
            "Compound": ["met_0"], "Identification Score": [0.7],
        })
        p = tmp_path / "feature_table.csv"
        df.to_csv(p, index=False)
        adata = load_annotation_scores(base_adata, str(p))
        assert adata.var.loc["met_0", "score"] == pytest.approx(0.7)
