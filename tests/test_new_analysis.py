"""
Tests for new analysis functions: cluster_nmf, multi_group_test, local_moran,
metabolite_set_enrichment, lipid_class_summary, subset_obs, merge_samples,
split_by_obs, run_harmony.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mortis.analysis import (
    cluster_nmf,
    compare_groups,
    lipid_class_summary,
    local_moran,
    merge_samples,
    metabolite_set_enrichment,
    multi_group_test,
    split_by_obs,
    subset_obs,
)
from mortis.exceptions import InvalidParameterError
from mortis.preprocessing import preprocess, run_harmony

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def base_adata():
    rng = np.random.default_rng(7)
    n = 120
    X = rng.random((n, 30)).astype(np.float32)
    X[:40, :10] += 2.0
    X[40:80, 10:20] += 2.0
    X[80:, 20:] += 2.0
    obs = pd.DataFrame({
        "x": np.tile(np.arange(12), 10),
        "y": np.repeat(np.arange(10), 12),
        "condition": ["A"] * 40 + ["B"] * 40 + ["C"] * 40,
        "patient": ["P1"] * 60 + ["P2"] * 60,
    })
    obs.index = [f"{r['x']}_{r['y']}_{i}" for i, (_, r) in enumerate(obs.iterrows())]
    var = pd.DataFrame(index=[f"met_{i}" for i in range(30)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    adata.uns["preprocessed_steps"] = []
    return preprocess(adata, n_pcs=10, scale_data=True)


# ---------------------------------------------------------------------------
# cluster_nmf
# ---------------------------------------------------------------------------

class TestClusterNmf:
    def test_returns_adata_and_df(self, base_adata):
        adata, df = cluster_nmf(base_adata.copy(), n_components=5)
        assert "nmf_cluster" in adata.obs.columns
        assert "X_nmf" in adata.obsm
        assert "nmf_components" in adata.uns
        assert isinstance(df, pd.DataFrame)
        assert "component" in df.columns
        assert "metabolite" in df.columns

    def test_n_components_respected(self, base_adata):
        adata, _ = cluster_nmf(base_adata.copy(), n_components=4)
        assert adata.obsm["X_nmf"].shape[1] == 4

    def test_custom_keys(self, base_adata):
        adata, _ = cluster_nmf(base_adata.copy(), n_components=3,
                                key_added="my_nmf", basis_key="X_my_nmf")
        assert "my_nmf" in adata.obs.columns
        assert "X_my_nmf" in adata.obsm

    def test_invalid_n_components_raises(self, base_adata):
        with pytest.raises(InvalidParameterError):
            cluster_nmf(base_adata, n_components=1)

    def test_copy_flag(self, base_adata):
        original_obs = set(base_adata.obs.columns)
        cluster_nmf(base_adata, n_components=3, copy=True)
        assert set(base_adata.obs.columns) == original_obs


# ---------------------------------------------------------------------------
# multi_group_test
# ---------------------------------------------------------------------------

class TestMultiGroupTest:
    def test_returns_adata_and_df(self, base_adata):
        adata, df = multi_group_test(base_adata.copy(), groupby="condition")
        assert isinstance(df, pd.DataFrame)
        for col in ("metabolite", "statistic", "pval", "pval_adj",
                    "significant", "eta_squared"):
            assert col in df.columns

    def test_detects_significant_metabolites(self, base_adata):
        _, df = multi_group_test(base_adata.copy(), groupby="condition")
        assert df["significant"].sum() > 0

    def test_anova_method(self, base_adata):
        _, df = multi_group_test(base_adata.copy(), groupby="condition",
                                 method="anova")
        assert len(df) == base_adata.n_vars

    def test_invalid_method_raises(self, base_adata):
        with pytest.raises(InvalidParameterError):
            multi_group_test(base_adata, groupby="condition", method="bad")

    def test_invalid_groupby_raises(self, base_adata):
        with pytest.raises(InvalidParameterError):
            multi_group_test(base_adata, groupby="nonexistent")

    def test_sorted_by_pval(self, base_adata):
        _, df = multi_group_test(base_adata.copy(), groupby="condition")
        assert (df["pval_adj"].diff().dropna() >= 0).all()

    def test_uns_populated(self, base_adata):
        adata, _ = multi_group_test(base_adata.copy(), groupby="condition")
        assert "multi_group_test" in adata.uns


# ---------------------------------------------------------------------------
# local_moran
# ---------------------------------------------------------------------------

class TestLocalMoran:
    def test_returns_adata_and_df(self, base_adata):
        adata, df = local_moran(base_adata.copy(), "met_0")
        assert "met_0_lisa" in adata.obs.columns
        assert "met_0_lisa_type" in adata.obs.columns
        assert isinstance(df, pd.DataFrame)
        for col in ("x", "y", "value", "local_i", "z_score", "pval", "lisa_type"):
            assert col in df.columns

    def test_lisa_types_valid(self, base_adata):
        _, df = local_moran(base_adata.copy(), "met_0")
        valid = {"HH", "LL", "HL", "LH", "NS"}
        assert set(df["lisa_type"].unique()).issubset(valid)

    def test_invalid_metabolite_raises(self, base_adata):
        with pytest.raises(InvalidParameterError):
            local_moran(base_adata, "nonexistent_met")

    def test_df_length_matches_n_obs(self, base_adata):
        _, df = local_moran(base_adata.copy(), "met_0")
        assert len(df) == base_adata.n_obs


# ---------------------------------------------------------------------------
# metabolite_set_enrichment
# ---------------------------------------------------------------------------

class TestMetaboliteSetEnrichment:
    def test_returns_dataframe(self, base_adata):
        _, results = compare_groups(base_adata.copy(), "condition", "A", "B")
        sets = {
            "Set1": ["met_0", "met_1", "met_2", "met_3"],
            "Set2": ["met_20", "met_21", "met_22", "met_23"],
        }
        df = metabolite_set_enrichment(results, sets)
        assert isinstance(df, pd.DataFrame)
        assert "pathway" in df.columns
        assert "nes" in df.columns
        assert "pval_adj" in df.columns

    def test_min_set_size_respected(self, base_adata):
        _, results = compare_groups(base_adata.copy(), "condition", "A", "B")
        sets = {"TooSmall": ["met_0", "met_1"]}  # only 2 members
        df = metabolite_set_enrichment(results, sets, min_set_size=3)
        assert len(df) == 0

    def test_empty_sets_returns_empty_df(self, base_adata):
        _, results = compare_groups(base_adata.copy(), "condition", "A", "B")
        df = metabolite_set_enrichment(results, {})
        assert len(df) == 0


# ---------------------------------------------------------------------------
# lipid_class_summary
# ---------------------------------------------------------------------------

class TestLipidClassSummary:
    def test_returns_dataframe(self, base_adata):
        df = lipid_class_summary(base_adata)
        assert isinstance(df, pd.DataFrame)
        assert "lipid_class" in df.columns
        assert "n_metabolites" in df.columns

    def test_with_groupby(self, base_adata):
        df = lipid_class_summary(base_adata, groupby="condition")
        assert "A" in df.columns or "B" in df.columns or "C" in df.columns

    def test_all_metabolites_classified(self, base_adata):
        df = lipid_class_summary(base_adata)
        total = df["n_metabolites"].sum()
        assert total == base_adata.n_vars


# ---------------------------------------------------------------------------
# subset_obs
# ---------------------------------------------------------------------------

class TestSubsetSample:
    def test_subsets_correctly(self, base_adata):
        sub = subset_obs(base_adata, "condition", "A")
        assert (sub.obs["condition"] == "A").all()
        assert sub.n_obs == 40

    def test_multiple_values(self, base_adata):
        sub = subset_obs(base_adata, "condition", ["A", "B"])
        assert sub.n_obs == 80

    def test_invalid_col_raises(self, base_adata):
        with pytest.raises(InvalidParameterError):
            subset_obs(base_adata, "nonexistent", "A")

    def test_invalid_value_raises(self, base_adata):
        with pytest.raises(InvalidParameterError):
            subset_obs(base_adata, "condition", "Z")

    def test_returns_copy_by_default(self, base_adata):
        sub = subset_obs(base_adata, "condition", "A")
        sub.obs["new_col"] = "test"
        assert "new_col" not in base_adata.obs.columns


# ---------------------------------------------------------------------------
# merge_samples
# ---------------------------------------------------------------------------

class TestMergeSamples:
    def test_merges_correctly(self, base_adata):
        a1 = subset_obs(base_adata, "condition", "A")
        a2 = subset_obs(base_adata, "condition", "B")
        merged = merge_samples([a1, a2], sample_labels=["S1", "S2"])
        assert merged.n_obs == 80
        assert "sample" in merged.obs.columns
        assert set(merged.obs["sample"].unique()) == {"S1", "S2"}

    def test_auto_labels(self, base_adata):
        a1 = subset_obs(base_adata, "condition", "A")
        a2 = subset_obs(base_adata, "condition", "B")
        merged = merge_samples([a1, a2])
        assert "sample_0" in merged.obs["sample"].values
        assert "sample_1" in merged.obs["sample"].values

    def test_label_length_mismatch_raises(self, base_adata):
        a1 = subset_obs(base_adata, "condition", "A")
        with pytest.raises(InvalidParameterError):
            merge_samples([a1], sample_labels=["S1", "S2"])


# ---------------------------------------------------------------------------
# split_by_obs
# ---------------------------------------------------------------------------

class TestSplitByObs:
    def test_splits_correctly(self, base_adata):
        splits = split_by_obs(base_adata, "condition")
        assert set(splits.keys()) == {"A", "B", "C"}
        assert splits["A"].n_obs == 40
        assert splits["B"].n_obs == 40
        assert splits["C"].n_obs == 40

    def test_total_obs_preserved(self, base_adata):
        splits = split_by_obs(base_adata, "condition")
        total = sum(v.n_obs for v in splits.values())
        assert total == base_adata.n_obs

    def test_invalid_col_raises(self, base_adata):
        with pytest.raises(InvalidParameterError):
            split_by_obs(base_adata, "nonexistent")

    def test_returns_copies(self, base_adata):
        splits = split_by_obs(base_adata, "condition")
        splits["A"].obs["new_col"] = "test"
        assert "new_col" not in base_adata.obs.columns


# ---------------------------------------------------------------------------
# run_harmony
# ---------------------------------------------------------------------------

class TestRunHarmony:
    def test_harmony_embedding_created(self, base_adata):
        adata = run_harmony(base_adata.copy(), batch_key="patient")
        assert "X_pca_harmony" in adata.obsm
        assert adata.obsm["X_pca_harmony"].shape == adata.obsm["X_pca"].shape

    def test_custom_basis_and_output(self, base_adata):
        adata = run_harmony(base_adata.copy(), batch_key="patient",
                            adjusted_basis="X_harmony_custom")
        assert "X_harmony_custom" in adata.obsm

    def test_raises_without_pca(self):
        obs = pd.DataFrame({"x": [0, 1], "y": [0, 0], "patient": ["P1", "P2"]})
        obs.index = ["0_0", "1_0"]
        adata = ad.AnnData(X=np.ones((2, 5), dtype=np.float32), obs=obs)
        from mortis.exceptions import NoEmbeddingError
        with pytest.raises(NoEmbeddingError):
            run_harmony(adata, batch_key="patient")

    def test_raises_with_invalid_batch_key(self, base_adata):
        with pytest.raises(InvalidParameterError):
            run_harmony(base_adata, batch_key="nonexistent")

    def test_copy_flag(self, base_adata):
        original_obsm = set(base_adata.obsm.keys())
        run_harmony(base_adata, batch_key="patient", copy=True)
        assert set(base_adata.obsm.keys()) == original_obsm
