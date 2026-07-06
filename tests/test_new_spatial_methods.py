"""
Tests for the second wave of spatial-statistics/validation additions:
Geary's C (bundled into spatial_autocorrelation), getis_ord_gi,
co_occurrence, unmix_pixels, diversity_index, cluster_diversity,
spatial_domains_kmeans, cluster_validation, compare_clusterings,
batch_mixing_score, and the cosine-metric option on
metabolite_colocalization.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mortis.analysis import (
    batch_mixing_score,
    cluster,
    cluster_diversity,
    cluster_validation,
    co_occurrence,
    compare_clusterings,
    diversity_index,
    getis_ord_gi,
    metabolite_colocalization,
    spatial_autocorrelation,
    spatial_domains_kmeans,
    unmix_pixels,
)
from mortis.exceptions import InvalidParameterError, NoClustersError, NoEmbeddingError
from mortis.preprocessing import preprocess

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clustered_grid_adata():
    """
    A 30x30 grid where the left half has a strong, spatially clustered
    "hotspot" metabolite and the right half is random noise, so tests can
    check that spatial statistics correctly identify the planted structure.
    """
    rng = np.random.default_rng(0)
    xs, ys = np.meshgrid(np.arange(30), np.arange(30))
    x, y = xs.ravel(), ys.ravel()
    n = len(x)

    hotspot = rng.random(n).astype(np.float32) * 0.1
    left_mask = x < 10
    hotspot[left_mask] += 5.0  # strong spatially contiguous block

    noise = rng.random(n).astype(np.float32)  # no spatial structure

    X = np.column_stack([hotspot, noise, rng.random(n).astype(np.float32)])
    obs = pd.DataFrame({"x": x, "y": y, "region": np.where(left_mask, "hot", "cold")})
    obs.index = [f"{a}_{b}" for a, b in zip(x, y)]
    var = pd.DataFrame(index=["hotspot_met", "noise_met", "extra_met"])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    adata.uns["preprocessed_steps"] = []
    return adata


@pytest.fixture
def preprocessed_adata():
    rng = np.random.default_rng(7)
    n = 100
    X = rng.random((n, 20)).astype(np.float32)
    obs = pd.DataFrame({
        "x": np.tile(np.arange(10), 10),
        "y": np.repeat(np.arange(10), 10),
        "sample": ["S1"] * 50 + ["S2"] * 50,
    })
    obs.index = [f"{r['x']}_{r['y']}_{i}" for i, (_, r) in enumerate(obs.iterrows())]
    var = pd.DataFrame(index=[f"met_{i}" for i in range(20)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    adata.uns["preprocessed_steps"] = []
    return preprocess(adata, n_pcs=8, scale_data=True)


# ---------------------------------------------------------------------------
# Geary's C (via spatial_autocorrelation)
# ---------------------------------------------------------------------------

class TestGearyC:
    def test_columns_present(self, clustered_grid_adata):
        adata, df = spatial_autocorrelation(clustered_grid_adata.copy(), n_neighbors=4)
        for col in ("geary_c", "geary_z_score", "geary_pval", "geary_pval_adj"):
            assert col in df.columns
        # z_score is DataFrame-only (matches the existing morans_i convention:
        # adata.var stores the statistic + both p-values, not the z-score).
        for col in ("geary_c", "geary_pval", "geary_pval_adj"):
            assert col in adata.var.columns

    def test_clustered_metabolite_has_low_geary_c(self, clustered_grid_adata):
        # Geary's C < 1 indicates positive spatial autocorrelation (clustering).
        _, df = spatial_autocorrelation(clustered_grid_adata.copy(), n_neighbors=4)
        hotspot_row = df[df["metabolite"] == "hotspot_met"].iloc[0]
        noise_row = df[df["metabolite"] == "noise_met"].iloc[0]
        assert hotspot_row["geary_c"] < noise_row["geary_c"]
        assert hotspot_row["geary_c"] < 0.9

    def test_geary_agrees_in_direction_with_moran(self, clustered_grid_adata):
        _, df = spatial_autocorrelation(clustered_grid_adata.copy(), n_neighbors=4)
        # High Moran's I (clustered) should correspond to low Geary's C.
        hotspot_row = df[df["metabolite"] == "hotspot_met"].iloc[0]
        assert hotspot_row["morans_i"] > 0.3
        assert hotspot_row["geary_c"] < 0.9


# ---------------------------------------------------------------------------
# getis_ord_gi
# ---------------------------------------------------------------------------

class TestGetisOrdGi:
    def test_returns_adata_and_df(self, clustered_grid_adata):
        adata, df = getis_ord_gi(clustered_grid_adata.copy(), "hotspot_met", n_neighbors=4)
        assert "hotspot_met_gi" in adata.obs.columns
        assert "hotspot_met_gi_type" in adata.obs.columns
        for col in ("x", "y", "value", "gi_star", "pval", "hotspot_type"):
            assert col in df.columns

    def test_planted_hotspot_detected(self, clustered_grid_adata):
        _, df = getis_ord_gi(clustered_grid_adata.copy(), "hotspot_met", n_neighbors=4)
        left_mask = df["x"] < 10
        hot_fraction_in_region = (df.loc[left_mask, "hotspot_type"] == "hot").mean()
        hot_fraction_outside = (df.loc[~left_mask, "hotspot_type"] == "hot").mean()
        assert hot_fraction_in_region > hot_fraction_outside

    def test_invalid_metabolite_raises(self, clustered_grid_adata):
        with pytest.raises(InvalidParameterError):
            getis_ord_gi(clustered_grid_adata, "nonexistent_met")

    def test_hotspot_types_valid(self, clustered_grid_adata):
        _, df = getis_ord_gi(clustered_grid_adata.copy(), "hotspot_met", n_neighbors=4)
        assert set(df["hotspot_type"].unique()).issubset({"hot", "cold", "NS"})


# ---------------------------------------------------------------------------
# co_occurrence
# ---------------------------------------------------------------------------

class TestCoOccurrence:
    def test_returns_dataframe(self, clustered_grid_adata):
        adata = clustered_grid_adata.copy()
        adata.obs["cluster"] = adata.obs["region"]  # reuse planted region as "clusters"
        df = co_occurrence(adata, cluster_key="cluster", n_bins=10)
        assert isinstance(df, pd.DataFrame)
        for col in ("cluster_a", "cluster_b", "bin", "distance", "ratio"):
            assert col in df.columns

    def test_self_occurrence_enriched_at_short_range(self, clustered_grid_adata):
        # A region should co-occur with itself more than chance at short distance.
        adata = clustered_grid_adata.copy()
        adata.obs["cluster"] = adata.obs["region"]
        df = co_occurrence(adata, cluster_key="cluster", n_bins=10)
        self_hot = df[(df["cluster_a"] == "hot") & (df["cluster_b"] == "hot")].sort_values("distance")
        assert self_hot.iloc[0]["ratio"] > 1.0

    def test_raises_without_clusters(self, clustered_grid_adata):
        with pytest.raises(NoClustersError):
            co_occurrence(clustered_grid_adata, cluster_key="nonexistent")


# ---------------------------------------------------------------------------
# unmix_pixels
# ---------------------------------------------------------------------------

class TestUnmixPixels:
    def test_recovers_known_fractions(self):
        # Construct pixels as EXACT known mixtures of two references so we
        # can check the recovered fractions against ground truth.
        mets = ["m0", "m1", "m2", "m3"]
        ref_a = {"m0": 1.0, "m1": 0.0, "m2": 1.0, "m3": 0.0}
        ref_b = {"m0": 0.0, "m1": 1.0, "m2": 0.0, "m3": 1.0}
        true_fracs = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.25, 0.75]])
        X = true_fracs @ np.array([[ref_a[m] for m in mets], [ref_b[m] for m in mets]])

        obs = pd.DataFrame({"x": range(4), "y": [0] * 4})
        obs.index = [f"{i}_0" for i in range(4)]
        var = pd.DataFrame(index=mets)
        adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)

        result = unmix_pixels(adata, {"A": ref_a, "B": ref_b})
        assert "X_unmixed" in result.obsm
        assert result.obsm["X_unmixed"].shape == (4, 2)
        recovered = result.obsm["X_unmixed"]
        assert np.allclose(recovered.sum(axis=1), 1.0, atol=1e-4)
        assert np.allclose(recovered, true_fracs, atol=0.05)

    def test_fraction_columns_added(self):
        mets = ["m0", "m1"]
        ref_a = {"m0": 1.0, "m1": 0.0}
        X = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        obs = pd.DataFrame({"x": [0, 1], "y": [0, 0]})
        obs.index = ["0_0", "1_0"]
        adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=mets))
        result = unmix_pixels(adata, {"A": ref_a})
        assert "fraction_A" in result.obs.columns

    def test_empty_reference_raises(self):
        adata = ad.AnnData(
            X=np.ones((2, 2), dtype=np.float32),
            obs=pd.DataFrame({"x": [0, 1], "y": [0, 0]}, index=["0_0", "1_0"]),
            var=pd.DataFrame(index=["m0", "m1"]),
        )
        with pytest.raises(InvalidParameterError):
            unmix_pixels(adata, {})

    def test_no_shared_metabolites_raises(self):
        adata = ad.AnnData(
            X=np.ones((2, 2), dtype=np.float32),
            obs=pd.DataFrame({"x": [0, 1], "y": [0, 0]}, index=["0_0", "1_0"]),
            var=pd.DataFrame(index=["m0", "m1"]),
        )
        with pytest.raises(InvalidParameterError):
            unmix_pixels(adata, {"A": {"totally_different_met": 1.0}})


# ---------------------------------------------------------------------------
# diversity_index / cluster_diversity
# ---------------------------------------------------------------------------

class TestDiversityIndex:
    def test_uniform_composition_maximizes_shannon(self):
        # A pixel with equal intensity across all metabolites has maximum
        # Shannon entropy; a pixel dominated by one metabolite has near-zero.
        X = np.array([[1.0, 1.0, 1.0, 1.0], [10.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        obs = pd.DataFrame({"x": [0, 1], "y": [0, 0]})
        obs.index = ["0_0", "1_0"]
        adata = ad.AnnData(X=X, obs=obs)
        result = diversity_index(adata, method="shannon")
        uniform_score, dominated_score = result.obs["shannon_diversity"].values
        assert uniform_score > dominated_score
        assert dominated_score == pytest.approx(0.0, abs=1e-5)

    def test_simpson_bounded(self):
        X = np.random.default_rng(0).random((10, 5)).astype(np.float32)
        obs = pd.DataFrame({"x": range(10), "y": [0] * 10})
        obs.index = [f"{i}_0" for i in range(10)]
        adata = ad.AnnData(X=X, obs=obs)
        result = diversity_index(adata, method="simpson")
        assert (result.obs["simpson_diversity"] >= 0).all()
        assert (result.obs["simpson_diversity"] < 1).all()

    def test_invalid_method_raises(self):
        adata = ad.AnnData(
            X=np.ones((2, 2), dtype=np.float32),
            obs=pd.DataFrame({"x": [0, 1], "y": [0, 0]}, index=["0_0", "1_0"]),
        )
        with pytest.raises(InvalidParameterError):
            diversity_index(adata, method="bad")


class TestClusterDiversity:
    def test_returns_dataframe(self, clustered_grid_adata):
        adata = clustered_grid_adata.copy()
        adata.obs["cluster"] = np.tile(["c0", "c1", "c2"], len(adata) // 3 + 1)[: len(adata)]
        df = cluster_diversity(adata, cluster_key="cluster", groupby="region")
        assert isinstance(df, pd.DataFrame)
        assert "shannon_diversity" in df.columns

    def test_raises_without_clusters(self, clustered_grid_adata):
        with pytest.raises(NoClustersError):
            cluster_diversity(clustered_grid_adata, cluster_key="nonexistent", groupby="region")


# ---------------------------------------------------------------------------
# spatial_domains_kmeans
# ---------------------------------------------------------------------------

class TestSpatialDomainsKmeans:
    def test_returns_exact_k_domains(self, preprocessed_adata):
        result = spatial_domains_kmeans(preprocessed_adata.copy(), n_domains=4, alpha=0.5)
        assert "domain_kmeans" in result.obs.columns
        assert result.obs["domain_kmeans"].nunique() == 4

    def test_invalid_n_domains_raises(self, preprocessed_adata):
        with pytest.raises(InvalidParameterError):
            spatial_domains_kmeans(preprocessed_adata, n_domains=1)

    def test_raises_without_pca(self):
        obs = pd.DataFrame({"x": [0, 1], "y": [0, 0]})
        obs.index = ["0_0", "1_0"]
        adata = ad.AnnData(X=np.ones((2, 5), dtype=np.float32), obs=obs)
        adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
        with pytest.raises(NoEmbeddingError):
            spatial_domains_kmeans(adata)


# ---------------------------------------------------------------------------
# cluster_validation / compare_clusterings
# ---------------------------------------------------------------------------

class TestClusterValidation:
    def test_returns_float_in_range(self, preprocessed_adata):
        adata = cluster(preprocessed_adata.copy(), resolution=0.5)
        score = cluster_validation(adata, cluster_key="cluster")
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_raises_without_clusters(self, preprocessed_adata):
        with pytest.raises(NoClustersError):
            cluster_validation(preprocessed_adata, cluster_key="nonexistent")


class TestCompareClusterings:
    def test_identical_labels_give_perfect_scores(self):
        labels = ["a", "a", "b", "b", "c"]
        result = compare_clusterings(labels, labels)
        assert result["ari"] == pytest.approx(1.0)
        assert result["ami"] == pytest.approx(1.0)

    def test_permuted_labels_still_perfect(self):
        # ARI/AMI must be invariant to cluster ID relabeling.
        a = ["0", "0", "1", "1", "2"]
        b = ["x", "x", "y", "y", "z"]
        result = compare_clusterings(a, b)
        assert result["ari"] == pytest.approx(1.0)

    def test_mismatched_length_raises(self):
        with pytest.raises(InvalidParameterError):
            compare_clusterings(["a", "b"], ["a", "b", "c"])


# ---------------------------------------------------------------------------
# batch_mixing_score
# ---------------------------------------------------------------------------

class TestBatchMixingScore:
    def test_separated_batches_score_near_one(self):
        # Two batches that occupy completely disjoint regions of embedding
        # space should have LISI close to 1 (no mixing).
        rng = np.random.default_rng(0)
        emb_a = rng.normal(loc=-10, scale=0.1, size=(30, 2))
        emb_b = rng.normal(loc=10, scale=0.1, size=(30, 2))
        X_pca = np.vstack([emb_a, emb_b]).astype(np.float32)
        obs = pd.DataFrame({
            "x": np.arange(60), "y": np.zeros(60, dtype=int),
            "sample": ["S1"] * 30 + ["S2"] * 30,
        })
        obs.index = [f"{i}_0" for i in range(60)]
        adata = ad.AnnData(X=np.ones((60, 3), dtype=np.float32), obs=obs)
        adata.obsm["X_pca"] = X_pca
        result = batch_mixing_score(adata, batch_key="sample", n_neighbors=10)
        assert result.obs["lisi_score"].mean() < 1.5

    def test_mixed_batches_score_higher(self):
        # Interleaved/shared-space batches should mix much better than
        # the separated case above.
        rng = np.random.default_rng(0)
        X_pca = rng.normal(size=(60, 2)).astype(np.float32)
        obs = pd.DataFrame({
            "x": np.arange(60), "y": np.zeros(60, dtype=int),
            "sample": (["S1", "S2"] * 30),
        })
        obs.index = [f"{i}_0" for i in range(60)]
        adata = ad.AnnData(X=np.ones((60, 3), dtype=np.float32), obs=obs)
        adata.obsm["X_pca"] = X_pca
        result = batch_mixing_score(adata, batch_key="sample", n_neighbors=10)
        assert result.obs["lisi_score"].mean() > 1.5

    def test_raises_without_embedding(self):
        obs = pd.DataFrame({"x": [0, 1], "y": [0, 0], "sample": ["S1", "S2"]})
        obs.index = ["0_0", "1_0"]
        adata = ad.AnnData(X=np.ones((2, 3), dtype=np.float32), obs=obs)
        with pytest.raises(NoEmbeddingError):
            batch_mixing_score(adata, batch_key="sample")


# ---------------------------------------------------------------------------
# metabolite_colocalization cosine metric
# ---------------------------------------------------------------------------

class TestColocalizationCosineMetric:
    def test_cosine_metric_runs(self, clustered_grid_adata):
        adata, _ = spatial_autocorrelation(clustered_grid_adata.copy(), n_neighbors=4)
        edges = metabolite_colocalization(adata, top_n=3, corr_threshold=-1.0, metric="cosine")
        assert isinstance(edges, pd.DataFrame)
        assert {"source", "target", "weight"}.issubset(edges.columns)
        # cosine similarity is bounded in [-1, 1]
        assert edges["weight"].between(-1.0001, 1.0001).all()

    def test_invalid_metric_raises(self, clustered_grid_adata):
        adata, _ = spatial_autocorrelation(clustered_grid_adata.copy(), n_neighbors=4)
        with pytest.raises(InvalidParameterError):
            metabolite_colocalization(adata, metric="bad_metric")
