"""
Tests for extended analysis functions: save_results, save_adata, run_paga,
spatial_neighbors, neighborhood_enrichment, plot_cluster_composition,
plot_embedding_grid, plot_volcano (with table), plot_cluster_composition.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mortis.analysis import (
    cluster,
    neighborhood_enrichment,
    run_paga,
    save_adata,
    save_results,
    spatial_neighbors,
)
from mortis.exceptions import InvalidParameterError, NoClustersError, NoEmbeddingError
from mortis.preprocessing import preprocess

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def preprocessed_adata():
    rng = np.random.default_rng(99)
    n = 80
    X = rng.random((n, 20)).astype(np.float32)
    obs = pd.DataFrame({
        "x": np.tile(np.arange(8), 10),
        "y": np.repeat(np.arange(10), 8),
        "condition": ["A"] * 40 + ["B"] * 40,
    })
    obs.index = [f"{r['x']}_{r['y']}" for _, r in obs.iterrows()]
    var = pd.DataFrame(index=[f"met_{i}" for i in range(20)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = obs[["x", "y"]].to_numpy(dtype=np.float32)
    adata.uns["preprocessed_steps"] = []
    return preprocess(adata, n_pcs=10, scale_data=False)


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

class TestSaveResults:
    def test_saves_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        out = str(tmp_path / "results.csv")
        save_results(df, out)
        assert Path(out).exists()
        loaded = pd.read_csv(out)
        assert len(loaded) == 2

    def test_creates_parent_dir(self, tmp_path):
        df = pd.DataFrame({"x": [1]})
        out = str(tmp_path / "subdir" / "results.csv")
        save_results(df, out)
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# save_adata
# ---------------------------------------------------------------------------

class TestSaveAdata:
    def test_saves_h5ad(self, preprocessed_adata, tmp_path):
        out = str(tmp_path / "test.h5ad")
        save_adata(preprocessed_adata, out)
        assert Path(out).exists()
        loaded = ad.read_h5ad(out)
        assert loaded.shape == preprocessed_adata.shape

    def test_preserves_embeddings(self, preprocessed_adata, tmp_path):
        out = str(tmp_path / "test.h5ad")
        save_adata(preprocessed_adata, out)
        loaded = ad.read_h5ad(out)
        assert "X_pca" in loaded.obsm

    def test_creates_parent_dir(self, preprocessed_adata, tmp_path):
        out = str(tmp_path / "subdir" / "test.h5ad")
        save_adata(preprocessed_adata, out)
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# run_paga
# ---------------------------------------------------------------------------

class TestRunPaga:
    def test_paga_computed(self, preprocessed_adata):
        # Use higher resolution to ensure multiple clusters exist
        adata = cluster(preprocessed_adata.copy(), resolution=2.0)
        if adata.obs["cluster"].nunique() < 2:
            pytest.skip("Not enough clusters for PAGA with this fixture")
        result = run_paga(adata)
        assert "paga" in result.uns

    def test_raises_without_clusters(self, preprocessed_adata):
        with pytest.raises(NoClustersError):
            run_paga(preprocessed_adata)

    def test_raises_without_neighbors(self):
        obs = pd.DataFrame({"x": [0], "y": [0], "cluster": ["0"]})
        adata = ad.AnnData(X=np.ones((1, 3), dtype=np.float32), obs=obs)
        with pytest.raises(NoEmbeddingError):
            run_paga(adata)


# ---------------------------------------------------------------------------
# spatial_neighbors
# ---------------------------------------------------------------------------

class TestSpatialNeighbors:
    def test_connectivity_matrix_created(self, preprocessed_adata):
        result = spatial_neighbors(preprocessed_adata.copy(), n_neighbors=4)
        assert "spatial_connectivities" in result.obsp
        assert "spatial_neighbors" in result.uns

    def test_matrix_is_symmetric(self, preprocessed_adata):
        result = spatial_neighbors(preprocessed_adata.copy(), n_neighbors=4)
        W = result.obsp["spatial_connectivities"].toarray()
        assert np.allclose(W, W.T)

    def test_raises_without_spatial(self):
        obs = pd.DataFrame({"x": [0], "y": [0]})
        adata = ad.AnnData(X=np.ones((1, 3), dtype=np.float32), obs=obs)
        from mortis.exceptions import MissingSpatialError
        with pytest.raises(MissingSpatialError):
            spatial_neighbors(adata)


# ---------------------------------------------------------------------------
# neighborhood_enrichment
# ---------------------------------------------------------------------------

class TestNeighborhoodEnrichment:
    def test_returns_dataframe(self, preprocessed_adata):
        adata = cluster(preprocessed_adata.copy(), resolution=0.5)
        adata = spatial_neighbors(adata, n_neighbors=4)
        _, enrich = neighborhood_enrichment(adata, n_permutations=50)
        assert isinstance(enrich, pd.DataFrame)
        for col in ("cluster_a", "cluster_b", "observed", "expected", "zscore", "pval"):
            assert col in enrich.columns

    def test_raises_without_spatial_graph(self, preprocessed_adata):
        adata = cluster(preprocessed_adata.copy(), resolution=0.5)
        with pytest.raises(InvalidParameterError, match="spatial_neighbors"):
            neighborhood_enrichment(adata)

    def test_raises_without_clusters(self, preprocessed_adata):
        adata = spatial_neighbors(preprocessed_adata.copy(), n_neighbors=4)
        with pytest.raises(NoClustersError):
            neighborhood_enrichment(adata)

    def test_uns_populated(self, preprocessed_adata):
        adata = cluster(preprocessed_adata.copy(), resolution=0.5)
        adata = spatial_neighbors(adata, n_neighbors=4)
        adata, _ = neighborhood_enrichment(adata, n_permutations=20)
        assert "neighborhood_enrichment" in adata.uns

    def test_n_jobs_restores_prior_thread_count(self, preprocessed_adata):
        # Regression test: n_jobs must not leak a changed Numba thread
        # count into the rest of the process after the call returns.
        import numba
        adata = cluster(preprocessed_adata.copy(), resolution=0.5)
        adata = spatial_neighbors(adata, n_neighbors=4)
        numba.set_num_threads(numba.config.NUMBA_DEFAULT_NUM_THREADS)
        prior = numba.get_num_threads()
        neighborhood_enrichment(adata, n_permutations=20, n_jobs=1)
        assert numba.get_num_threads() == prior
