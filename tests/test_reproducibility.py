"""
Regression tests for correctness and reproducibility.

Every test here corresponds to a bug that was actually shipped and actually
produced wrong numbers on real data. They are grouped separately from the
feature tests because this file is the answer to "can I trust the output?",
which is a different question from "does the function run?".

The rule for this file: a test only belongs here if it fails on the code as it
stood before the fix. No aspirational tests.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

import mortis as mt
from mortis.analysis import _build_spatial_weights, _get_X, _offset_coords_by_batch


def _grid_adata(n_side=12, n_vars=20, n_batches=1, seed=0):
    """Small gridded dataset with integer pixel coordinates, one or more batches."""
    rng = np.random.default_rng(seed)
    n_per = n_side * n_side
    coords = np.array([[i % n_side, i // n_side] for i in range(n_per)], dtype=np.float64)
    adata = ad.AnnData(X=rng.random((n_per * n_batches, n_vars)).astype(np.float32) * 100)
    adata.obsm["spatial"] = np.tile(coords, (n_batches, 1))
    adata.obs["sample"] = np.repeat([f"s{i}" for i in range(n_batches)], n_per)
    return adata


# ---------------------------------------------------------------------------
# Spatial coordinates must survive batch offsetting
# ---------------------------------------------------------------------------

class TestBatchOffsetPrecision:
    """
    ``_offset_coords_by_batch`` used to shift batch i by ``i * 1e6`` in float32.
    Past ~8.4e6 the float32 gap between representable values exceeds one pixel,
    so neighbouring pixels round onto each other. On a real 52-section cohort
    this collapsed ~49% of distinct pixel coordinates in 35 of 52 sections, and
    every spatial statistic computed for those sections was quietly wrong.
    """

    @pytest.mark.parametrize("n_batches", [2, 17, 30, 60])
    def test_no_coordinates_collapse(self, n_batches):
        adata = _grid_adata(n_side=10, n_batches=n_batches)
        offset = _offset_coords_by_batch(adata, "sample")
        batches = adata.obs["sample"].values

        for b in np.unique(batches):
            mask = batches == b
            n_before = len(np.unique(adata.obsm["spatial"][mask], axis=0))
            n_after = len(np.unique(offset[mask], axis=0))
            assert n_after == n_before, (
                f"batch {b} of {n_batches}: {n_before} distinct coords collapsed to {n_after}"
            )

    def test_batches_never_overlap(self, n_batches=40):
        """The whole point of the offset: no batch may reach into another."""
        adata = _grid_adata(n_side=8, n_batches=n_batches)
        offset = _offset_coords_by_batch(adata, "sample")
        batches = adata.obs["sample"].values

        ranges = []
        for b in np.unique(batches):
            xs = offset[batches == b, 0]
            ranges.append((xs.min(), xs.max()))
        ranges.sort()
        for (_, hi), (lo, _) in zip(ranges, ranges[1:]):
            assert lo > hi, "two batches occupy overlapping x-ranges"

    def test_offset_survives_large_coordinate_units(self):
        """A hard-coded 1e6 shift breaks for coordinates reported in nanometres."""
        adata = _grid_adata(n_side=6, n_batches=5)
        adata.obsm["spatial"] = adata.obsm["spatial"] * 5e6  # huge units
        offset = _offset_coords_by_batch(adata, "sample")
        batches = adata.obs["sample"].values
        for b in np.unique(batches):
            mask = batches == b
            assert len(np.unique(offset[mask], axis=0)) == len(
                np.unique(adata.obsm["spatial"][mask], axis=0)
            )

    def test_neighbours_never_cross_batches(self):
        adata = _grid_adata(n_side=10, n_batches=35)
        _, idx = _build_spatial_weights(adata, n_neighbors=6, batch_key="sample")
        batches = adata.obs["sample"].values
        neighbour_batches = batches[idx]
        assert (neighbour_batches == batches[:, None]).all(), (
            "a spatial neighbour was drawn from a different section"
        )

    def test_coordinates_are_float64(self):
        adata = _grid_adata(n_batches=3)
        assert _offset_coords_by_batch(adata, "sample").dtype == np.float64


# ---------------------------------------------------------------------------
# No stale cached matrices
# ---------------------------------------------------------------------------

class TestNoStaleMatrixCache:
    """
    ``_get_X`` used to memoise its dense result into ``adata.uns['_perf_cache']``
    with no invalidation. Any step that replaced ``.X`` left later calls reading
    pre-modification data, and the copies were serialised into every h5ad.
    """

    def test_get_X_follows_X(self):
        adata = _grid_adata()
        first = _get_X(adata).copy()
        adata.X = np.zeros_like(np.asarray(adata.X))
        second = _get_X(adata)
        assert not np.allclose(first, second), "_get_X returned pre-modification data"
        assert np.allclose(second, 0.0)

    def test_analysis_reflects_modified_X(self):
        adata = _grid_adata(n_vars=15)
        adata.obs["grp"] = ["a"] * (adata.n_obs // 2) + ["b"] * (adata.n_obs - adata.n_obs // 2)
        _, before = mt.multi_group_test(adata, "grp")

        rng = np.random.default_rng(99)
        adata.X = rng.random(adata.shape).astype(np.float32) * 100
        _, after = mt.multi_group_test(adata, "grp")

        before = before.set_index("metabolite")["statistic"]
        after = after.set_index("metabolite")["statistic"].reindex(before.index)
        assert not np.allclose(before.values, after.values), (
            "multi_group_test returned cached statistics after .X was replaced"
        )

    def test_no_cache_written_to_uns(self):
        adata = _grid_adata(n_vars=15)
        adata.obs["grp"] = ["a"] * (adata.n_obs // 2) + ["b"] * (adata.n_obs - adata.n_obs // 2)
        mt.spatial_autocorrelation(adata)
        mt.multi_group_test(adata, "grp")
        mt.score_metabolite_set(adata, list(adata.var_names[:3]))
        assert "_perf_cache" not in adata.uns

    def test_legacy_cache_is_purged_on_read(self, tmp_path):
        adata = _grid_adata(n_vars=10)
        adata.uns["_perf_cache"] = {"_x_for_spatial_moran": np.zeros((4, 4), dtype=np.float32)}
        path = tmp_path / "legacy.h5ad"
        adata.write_h5ad(path)

        reloaded = mt.read_metabolomics_data(str(path))
        assert "_perf_cache" not in reloaded.uns


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """
    Same seed must give the same numbers — on a rerun, and on a machine with a
    different core count. ``neighborhood_enrichment`` failed the second half:
    ``np.random.seed`` inside an ``@njit(parallel=True)`` function seeds only
    one worker thread, so results tracked the host's CPU count.
    """

    @staticmethod
    def _clustered(seed=0):
        adata = _grid_adata(n_side=14, n_vars=12, seed=seed)
        rng = np.random.default_rng(seed)
        adata.obs["cluster"] = np.array(["c0", "c1", "c2"])[rng.integers(0, 3, adata.n_obs)]
        return mt.spatial_neighbors(adata)

    @pytest.mark.parametrize("n_jobs", [1, 2, 4])
    def test_neighborhood_enrichment_independent_of_thread_count(self, n_jobs):
        reference = mt.neighborhood_enrichment(
            self._clustered(), n_permutations=100, n_jobs=1, random_state=42
        )[1].sort_values(["cluster_a", "cluster_b"])

        result = mt.neighborhood_enrichment(
            self._clustered(), n_permutations=100, n_jobs=n_jobs, random_state=42
        )[1].sort_values(["cluster_a", "cluster_b"])

        np.testing.assert_array_equal(
            reference["zscore"].to_numpy(), result["zscore"].to_numpy(),
            err_msg=f"z-scores differ between 1 and {n_jobs} threads at the same seed",
        )

    def test_neighborhood_enrichment_repeatable(self):
        a = mt.neighborhood_enrichment(self._clustered(), n_permutations=100, random_state=7)[1]
        b = mt.neighborhood_enrichment(self._clustered(), n_permutations=100, random_state=7)[1]
        np.testing.assert_array_equal(a["zscore"].to_numpy(), b["zscore"].to_numpy())

    def test_different_seeds_give_different_nulls(self):
        """Guards against the opposite failure: a seed that does nothing at all."""
        a = mt.neighborhood_enrichment(self._clustered(), n_permutations=100, random_state=1)[1]
        b = mt.neighborhood_enrichment(self._clustered(), n_permutations=100, random_state=2)[1]
        assert not np.array_equal(a["expected"].to_numpy(), b["expected"].to_numpy())

    def test_spatial_autocorrelation_is_deterministic(self):
        adata = _grid_adata(n_side=12, n_vars=15)
        _, first = mt.spatial_autocorrelation(adata.copy())
        _, second = mt.spatial_autocorrelation(adata.copy())
        np.testing.assert_array_equal(
            first["morans_i"].to_numpy(), second["morans_i"].to_numpy()
        )
