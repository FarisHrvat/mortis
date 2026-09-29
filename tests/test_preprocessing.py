"""
Tests for mortis.preprocess, normalization, filtering, PCA, UMAP, neighbors.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mortis.exceptions import (
    InvalidParameterError,
    MissingROIError,
    NoEmbeddingError,
)
from mortis.preprocessing import (
    _numba_thread_limit,
    filter_background,
    log1p_transform,
    preprocess,
    run_neighbors,
    run_pca,
    run_umap,
    scale,
    tic_normalize,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def paired_adata():
    """
    4 tissue pixels, 2 background pixels, 3 metabolites.
    Metabolite 0: high tissue, low background  → FC ~10  (keep at cutoff 1.5)
    Metabolite 1: low tissue, high background  → FC ~0.1 (remove at cutoff 1.5)
    Metabolite 2: high tissue, medium background → FC ~9 (keep at cutoff 1.5)
    """
    X = np.array([
        [10.0, 0.5, 45.0],   # tissue
        [20.0, 1.5, 55.0],   # tissue
        [12.0, 0.8, 40.0],   # tissue
        [18.0, 1.2, 50.0],   # tissue
        [1.0,  12.0, 5.0],   # background
        [2.0,  14.0, 6.0],   # background
    ], dtype=np.float32)
    obs = pd.DataFrame({
        "x": [0, 1, 2, 3, 4, 5],
        "y": [0, 0, 0, 0, 0, 0],
        "is_tissue": [True, True, True, True, False, False],
        "is_background": [False, False, False, False, True, True],
    })
    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    return adata


@pytest.fixture
def tissue_adata():
    """Tissue-only AnnData for normalization / PCA tests."""
    rng = np.random.default_rng(0)
    X = rng.random((50, 30)).astype(np.float32) * 100
    obs = pd.DataFrame({
        "x": np.arange(50),
        "y": np.zeros(50, dtype=int),
    })
    obs.index = obs["x"].astype(str) + "_0"
    adata = ad.AnnData(X=X, obs=obs)
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    adata.uns["preprocessed_steps"] = []
    return adata


# ---------------------------------------------------------------------------
# filter_background
# ---------------------------------------------------------------------------

class TestFilterBackground:
    def test_sample_mode_keeps_correct_metabolites(self, paired_adata):
        clean, stats = filter_background([paired_adata], cutoff=1.5, mode="sample")
        assert len(clean) == 1
        assert clean[0].n_obs == 4          # only tissue pixels
        assert clean[0].n_vars == 2         # met_0 and met_2 kept
        assert stats[0]["n_kept"] == 2
        assert stats[0]["n_removed"] == 1

    def test_group_mode_keeps_correct_metabolites(self, paired_adata):
        clean, stats = filter_background([paired_adata], cutoff=1.5, mode="group")
        assert clean[0].n_vars == 2

    def test_high_cutoff_removes_all(self, paired_adata):
        clean, stats = filter_background([paired_adata], cutoff=100.0, mode="sample")
        assert clean[0].n_vars == 0

    def test_zero_cutoff_raises(self, paired_adata):
        with pytest.raises(InvalidParameterError, match="has to be positive"):
            filter_background([paired_adata], cutoff=0, mode="sample")

    def test_invalid_mode_raises(self, paired_adata):
        with pytest.raises(InvalidParameterError, match="mode must be"):
            filter_background([paired_adata], cutoff=1.5, mode="invalid")

    def test_missing_roi_raises(self):
        obs = pd.DataFrame({"x": [0], "y": [0]})
        adata = ad.AnnData(X=np.ones((1, 3), dtype=np.float32), obs=obs)
        with pytest.raises(MissingROIError):
            filter_background([adata], cutoff=1.5, mode="sample")

    def test_stats_keys_present(self, paired_adata):
        _, stats = filter_background([paired_adata], cutoff=1.5, mode="sample")
        for key in ("mean_tissue", "mean_bg", "fold_change", "keep_mask", "cutoff", "n_kept", "n_removed"):
            assert key in stats[0]

    def test_original_adata_not_modified(self, paired_adata):
        original_shape = paired_adata.shape
        filter_background([paired_adata], cutoff=1.5, mode="sample")
        assert paired_adata.shape == original_shape


# ---------------------------------------------------------------------------
# tic_normalize
# ---------------------------------------------------------------------------

class TestTicNormalize:
    def test_row_sums_equalized(self, tissue_adata):
        # Default target is the median row sum (not 1.0) so log1p stays
        # informative, see tic_normalize docstring. The invariant to check
        # is that every row is scaled to the *same* total, not to 1.0.
        result = tic_normalize(tissue_adata.copy())
        row_sums = result.X.sum(axis=1)
        assert np.allclose(row_sums, row_sums[0], atol=1e-2)

    def test_explicit_target_sum(self, tissue_adata):
        result = tic_normalize(tissue_adata.copy(), target_sum=1.0)
        row_sums = result.X.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_zero_row_handled(self):
        X = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
        obs = pd.DataFrame({"x": [0, 1], "y": [0, 0]})
        adata = ad.AnnData(X=X, obs=obs)
        result = tic_normalize(adata, target_sum=1.0)
        assert np.allclose(result.X[0], 0.0)  # zero row stays zero
        assert np.isclose(result.X[1].sum(), 1.0)

    def test_copy_flag(self, tissue_adata):
        original_X = tissue_adata.X.copy()
        result = tic_normalize(tissue_adata, copy=True)
        assert not np.allclose(result.X, original_X)
        assert np.allclose(tissue_adata.X, original_X)  # original unchanged

    def test_step_recorded(self, tissue_adata):
        result = tic_normalize(tissue_adata.copy())
        assert "tic_normalize" in result.uns["preprocessed_steps"]


# ---------------------------------------------------------------------------
# log1p_transform
# ---------------------------------------------------------------------------

class TestLog1pTransform:
    def test_values_correct(self, tissue_adata):
        original = tissue_adata.X.copy()
        result = log1p_transform(tissue_adata.copy())
        assert np.allclose(result.X, np.log1p(original), atol=1e-5)

    def test_zero_maps_to_zero(self):
        X = np.array([[0.0, 1.0]], dtype=np.float32)
        obs = pd.DataFrame({"x": [0], "y": [0]})
        adata = ad.AnnData(X=X, obs=obs)
        result = log1p_transform(adata)
        assert np.isclose(result.X[0, 0], 0.0)

    def test_step_recorded(self, tissue_adata):
        result = log1p_transform(tissue_adata.copy())
        assert "log1p" in result.uns["preprocessed_steps"]


# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------

class TestScale:
    def test_mean_near_zero(self, tissue_adata):
        adata = tic_normalize(tissue_adata.copy())
        adata = log1p_transform(adata)
        result = scale(adata.copy())
        col_means = result.X.mean(axis=0)
        assert np.allclose(col_means, 0.0, atol=0.1)

    def test_log1p_layer_preserved(self, tissue_adata):
        adata = tic_normalize(tissue_adata.copy())
        adata = log1p_transform(adata)
        result = scale(adata.copy())
        assert "log1p" in result.layers

    def test_clipping_applied(self, tissue_adata):
        adata = tic_normalize(tissue_adata.copy())
        adata = log1p_transform(adata)
        result = scale(adata.copy(), max_value=5.0)
        assert result.X.max() <= 5.0 + 1e-5
        assert result.X.min() >= -5.0 - 1e-5


# ---------------------------------------------------------------------------
# run_pca
# ---------------------------------------------------------------------------

class TestRunPca:
    def test_pca_embedding_created(self, tissue_adata):
        adata = preprocess(tissue_adata.copy(), n_pcs=10, scale_data=False)
        assert "X_pca" in adata.obsm
        assert adata.obsm["X_pca"].shape[0] == 50

    def test_n_comps_capped(self):
        X = np.random.rand(5, 3).astype(np.float32)
        obs = pd.DataFrame({"x": range(5), "y": [0] * 5})
        obs.index = [f"{i}_0" for i in range(5)]
        adata = ad.AnnData(X=X, obs=obs)
        adata.uns["preprocessed_steps"] = ["tic_normalize", "log1p"]
        result = run_pca(adata, n_comps=100)
        assert result.obsm["X_pca"].shape[1] <= min(5, 3) - 1


# ---------------------------------------------------------------------------
# run_neighbors
# ---------------------------------------------------------------------------

class TestRunNeighbors:
    def test_raises_without_pca(self, tissue_adata):
        with pytest.raises(NoEmbeddingError, match="PCA embedding not found"):
            run_neighbors(tissue_adata)

    def test_neighbors_computed(self, tissue_adata):
        adata = preprocess(tissue_adata.copy(), n_pcs=10, scale_data=False)
        assert "neighbors" in adata.uns


# ---------------------------------------------------------------------------
# run_umap
# ---------------------------------------------------------------------------

class TestRunUmap:
    def test_raises_without_neighbors(self, tissue_adata):
        with pytest.raises(NoEmbeddingError, match="Neighbour graph not found"):
            run_umap(tissue_adata)

    def test_umap_embedding_created(self, tissue_adata):
        adata = preprocess(tissue_adata.copy(), n_pcs=10, scale_data=False)
        adata = run_umap(adata)
        assert "X_umap" in adata.obsm
        assert adata.obsm["X_umap"].shape == (50, 2)


# ---------------------------------------------------------------------------
# preprocess (convenience wrapper)
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_full_pipeline(self, tissue_adata):
        result = preprocess(tissue_adata.copy(), n_pcs=10)
        assert "X_pca" in result.obsm
        assert "neighbors" in result.uns
        steps = result.uns["preprocessed_steps"]
        assert "tic_normalize" in steps
        assert "log1p" in steps

    def test_copy_flag(self, tissue_adata):
        original_X = tissue_adata.X.copy()
        preprocess(tissue_adata, copy=True, n_pcs=10)
        assert np.allclose(tissue_adata.X, original_X)


# ---------------------------------------------------------------------------
# _numba_thread_limit (internal thread-control helper used by run_neighbors
# and run_umap; see MORTIS_N_JOBS in the README for the full rationale)
# ---------------------------------------------------------------------------

class TestNumbaThreadLimit:
    def test_sets_and_restores_thread_count(self):
        import numba
        numba.set_num_threads(numba.config.NUMBA_DEFAULT_NUM_THREADS)
        prior = numba.get_num_threads()
        with _numba_thread_limit():
            # Whatever MORTIS_N_JOBS/_SKLEARN_THREAD_LIMIT resolves to,
            # entering the context must change numba's active thread count
            # to that value (unless it already equals prior, e.g. running
            # with 1 CPU available).
            from mortis.preprocessing import _SKLEARN_THREAD_LIMIT
            assert numba.get_num_threads() == _SKLEARN_THREAD_LIMIT
        assert numba.get_num_threads() == prior

