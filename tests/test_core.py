"""
Tests for mortis.analysis — clustering, markers, differential expression,
spatial statistics, and metabolite set scoring.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mortis.analysis import (
    cluster,
    compare_groups,
    find_markers,
    rename_clusters,
    score_metabolite_set,
    spatial_autocorrelation,
    spatial_de,
)
from mortis.exceptions import (
    InsufficientSamplesError,
    InvalidParameterError,
    NoClustersError,
    NoEmbeddingError,
    NotPreprocessedError,
)
from mortis.preprocessing import preprocess

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def preprocessed_adata():
    """
    Preprocessed AnnData with 100 pixels, 40 metabolites, and two groups.
    Metabolites 0-19 are elevated in group A; 20-39 in group B.
    """
    rng = np.random.default_rng(7)
    n = 100
    X = rng.random((n, 40)).astype(np.float32)
    # Make groups separable
    X[:50, :20] += 2.0   # group A elevated in first 20 metabolites
    X[50:, 20:] += 2.0   # group B elevated in last 20 metabolites

    obs = pd.DataFrame({
        "x": np.tile(np.arange(10), 10),
        "y": np.repeat(np.arange(10), 10),
        "condition": ["A"] * 50 + ["B"] * 50,
    })
    obs.index = [f"{r['x']}_{r['y']}" for _, r in obs.iterrows()]
    var = pd.DataFrame(index=[f"met_{i}" for i in range(40)])

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    adata.uns["preprocessed_steps"] = []

    adata = preprocess(adata, n_pcs=10, scale_data=True)
    return adata


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------

class TestCluster:
    def test_cluster_labels_created(self, preprocessed_adata):
        result = cluster(preprocessed_adata.copy(), resolution=0.5)
        assert "cluster" in result.obs.columns
        assert result.obs["cluster"].nunique() >= 1

    def test_custom_key_added(self, preprocessed_adata):
        result = cluster(preprocessed_adata.copy(), key_added="leiden_custom")
        assert "leiden_custom" in result.obs.columns

    def test_invalid_resolution_raises(self, preprocessed_adata):
        with pytest.raises(InvalidParameterError, match="resolution must be > 0"):
            cluster(preprocessed_adata, resolution=-0.1)

    def test_raises_without_neighbors(self):
        obs = pd.DataFrame({"x": [0], "y": [0]})
        adata = ad.AnnData(X=np.ones((1, 3), dtype=np.float32), obs=obs)
        with pytest.raises(NoEmbeddingError):
            cluster(adata)

    def test_copy_flag(self, preprocessed_adata):
        original_obs_cols = set(preprocessed_adata.obs.columns)
        cluster(preprocessed_adata, copy=True)
        assert set(preprocessed_adata.obs.columns) == original_obs_cols


# ---------------------------------------------------------------------------
# rename_clusters
# ---------------------------------------------------------------------------

class TestRenameClusters:
    def test_renaming_applied(self, preprocessed_adata):
        adata = cluster(preprocessed_adata.copy(), resolution=0.5)
        original_ids = adata.obs["cluster"].unique().tolist()
        mapping = {str(original_ids[0]): "Tumour"}
        result = rename_clusters(adata, mapping=mapping)
        assert "Tumour" in result.obs["cluster"].values

    def test_original_preserved(self, preprocessed_adata):
        adata = cluster(preprocessed_adata.copy(), resolution=0.5)
        rename_clusters(adata, mapping={"0": "Renamed"})
        assert "cluster_original" in adata.obs.columns

    def test_unmapped_clusters_unchanged(self, preprocessed_adata):
        adata = cluster(preprocessed_adata.copy(), resolution=0.5)
        all_ids = adata.obs["cluster"].unique().tolist()
        if len(all_ids) > 1:
            mapping = {str(all_ids[0]): "Named"}
            rename_clusters(adata, mapping=mapping)
            # Other clusters should still have their original IDs
            remaining = [c for c in adata.obs["cluster"].unique() if c != "Named"]
            assert len(remaining) >= 1

    def test_raises_without_clusters(self, preprocessed_adata):
        with pytest.raises(NoClustersError):
            rename_clusters(preprocessed_adata, mapping={"0": "X"})


# ---------------------------------------------------------------------------
# find_markers
# ---------------------------------------------------------------------------

class TestFindMarkers:
    def test_markers_dataframe_returned(self, preprocessed_adata):
        adata = cluster(preprocessed_adata.copy(), resolution=0.5)
        _, markers = find_markers(adata, n_top=5)
        assert isinstance(markers, pd.DataFrame)
        for col in ("cluster", "metabolite", "score", "pval", "pval_adj", "log2fc"):
            assert col in markers.columns

    def test_n_top_respected(self, preprocessed_adata):
        adata = cluster(preprocessed_adata.copy(), resolution=0.5)
        _, markers = find_markers(adata, n_top=3)
        n_clusters = adata.obs["cluster"].nunique()
        assert len(markers) == n_clusters * 3

    def test_raises_without_clusters(self, preprocessed_adata):
        with pytest.raises(NoClustersError):
            find_markers(preprocessed_adata)

    def test_raises_without_preprocessing(self):
        obs = pd.DataFrame({"x": [0, 1], "y": [0, 0], "cluster": ["0", "1"]})
        adata = ad.AnnData(X=np.ones((2, 3), dtype=np.float32), obs=obs)
        with pytest.raises(NotPreprocessedError):
            find_markers(adata)

    def test_invalid_method_raises(self, preprocessed_adata):
        adata = cluster(preprocessed_adata.copy(), resolution=0.5)
        with pytest.raises(InvalidParameterError, match="method must be"):
            find_markers(adata, method="bad_method")


# ---------------------------------------------------------------------------
# compare_groups
# ---------------------------------------------------------------------------

class TestCompareGroups:
    def test_results_dataframe_returned(self, preprocessed_adata):
        _, results = compare_groups(
            preprocessed_adata, groupby="condition",
            group1="A", group2="B"
        )
        assert isinstance(results, pd.DataFrame)
        for col in ("metabolite", "log2fc", "pval", "pval_adj", "significant"):
            assert col in results.columns

    def test_significant_metabolites_detected(self, preprocessed_adata):
        _, results = compare_groups(
            preprocessed_adata, groupby="condition",
            group1="A", group2="B"
        )
        assert results["significant"].sum() > 0

    def test_sorted_by_pval_adj(self, preprocessed_adata):
        _, results = compare_groups(
            preprocessed_adata, groupby="condition",
            group1="A", group2="B"
        )
        assert (results["pval_adj"].diff().dropna() >= 0).all()

    def test_invalid_groupby_raises(self, preprocessed_adata):
        with pytest.raises(InvalidParameterError, match="not found in adata.obs"):
            compare_groups(preprocessed_adata, "nonexistent", "A", "B")

    def test_invalid_group_raises(self, preprocessed_adata):
        with pytest.raises(InvalidParameterError, match="not found"):
            compare_groups(preprocessed_adata, "condition", "A", "C")

    def test_insufficient_samples_raises(self):
        obs = pd.DataFrame({
            "x": [0, 1, 2], "y": [0, 0, 0],
            "condition": ["A", "A", "B"],
        })
        obs.index = ["0_0", "1_0", "2_0"]
        adata = ad.AnnData(X=np.ones((3, 5), dtype=np.float32), obs=obs)
        adata.uns["preprocessed_steps"] = ["tic_normalize"]
        with pytest.raises(InsufficientSamplesError):
            compare_groups(adata, "condition", "A", "B")


# ---------------------------------------------------------------------------
# spatial_autocorrelation
# ---------------------------------------------------------------------------

class TestSpatialAutocorrelation:
    def test_morans_i_computed(self, preprocessed_adata):
        _, morans = spatial_autocorrelation(preprocessed_adata.copy(), n_neighbors=4)
        assert "morans_i" in morans.columns
        assert len(morans) == preprocessed_adata.n_vars

    def test_morans_i_range(self, preprocessed_adata):
        _, morans = spatial_autocorrelation(preprocessed_adata.copy(), n_neighbors=4)
        assert morans["morans_i"].between(-1.1, 1.1).all()

    def test_var_columns_added(self, preprocessed_adata):
        adata, _ = spatial_autocorrelation(preprocessed_adata.copy(), n_neighbors=4)
        assert "morans_i" in adata.var.columns
        assert "morans_pval" in adata.var.columns

    def test_raises_without_spatial(self):
        obs = pd.DataFrame({"x": [0], "y": [0]})
        adata = ad.AnnData(X=np.ones((1, 3), dtype=np.float32), obs=obs)
        with pytest.raises(Exception):
            spatial_autocorrelation(adata)


# ---------------------------------------------------------------------------
# spatial_de
# ---------------------------------------------------------------------------

class TestSpatialDe:
    def test_returns_dataframe(self, preprocessed_adata):
        _, svgs = spatial_de(preprocessed_adata.copy(), n_top=10)
        assert isinstance(svgs, pd.DataFrame)

    def test_n_top_respected(self, preprocessed_adata):
        _, svgs = spatial_de(preprocessed_adata.copy(), n_top=5)
        assert len(svgs) <= 5


# ---------------------------------------------------------------------------
# score_metabolite_set
# ---------------------------------------------------------------------------

class TestScoreMetaboliteSet:
    def test_score_added_to_obs(self, preprocessed_adata):
        mets = ["met_0", "met_1", "met_2"]
        result = score_metabolite_set(preprocessed_adata.copy(), mets, score_name="test_score")
        assert "test_score" in result.obs.columns
        assert result.obs["test_score"].shape[0] == preprocessed_adata.n_obs

    def test_missing_metabolites_warned(self, preprocessed_adata, capsys):
        mets = ["met_0", "nonexistent_metabolite"]
        score_metabolite_set(preprocessed_adata.copy(), mets)
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_all_missing_raises(self, preprocessed_adata):
        with pytest.raises(InvalidParameterError, match="None of the provided"):
            score_metabolite_set(preprocessed_adata, ["fake_met_1", "fake_met_2"])

    def test_copy_flag(self, preprocessed_adata):
        original_cols = set(preprocessed_adata.obs.columns)
        score_metabolite_set(preprocessed_adata, ["met_0"], copy=True)
        assert set(preprocessed_adata.obs.columns) == original_cols
