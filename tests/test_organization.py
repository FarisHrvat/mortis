"""
Tests for differential spatial organization.

The claim this module makes is specific and falsifiable: it can find a
metabolite whose *total abundance* is identical between two groups but whose
*spatial arrangement* differs. ``TestOrganizationOnlySignal`` constructs
exactly that situation — same mean intensity per section, different pattern —
and asserts that abundance testing misses it while organization testing finds
it. If that test ever fails, the module has no reason to exist.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

import mortis as mt
from mortis.exceptions import InsufficientSamplesError, InvalidParameterError

SIDE = 24
N_PIXELS = SIDE * SIDE
COORDS = np.array([[i % SIDE, i // SIDE] for i in range(N_PIXELS)], dtype=np.float64)


def _clustered_image(rng, total, blob_radius=4.0):
    """Signal concentrated in one central blob, on a low background."""
    centre = np.array([SIDE / 2, SIDE / 2])
    d = np.linalg.norm(COORDS - centre, axis=1)
    image = np.exp(-(d ** 2) / (2 * blob_radius ** 2))
    image = image + rng.normal(0, 0.02, N_PIXELS).clip(0)
    return image * (total / image.sum())


def _diffuse_image(rng, total):
    """The same total signal, spread uniformly at random."""
    image = rng.random(N_PIXELS) + 0.5
    return image * (total / image.sum())


def _organization_cohort(n_per_group=5, n_vars=30, seed=0, organized_indices=(0, 1, 2)):
    """
    Both groups get the *same total intensity* for every metabolite. For the
    metabolites in ``organized_indices``, group 'R' gets a clustered pattern and
    'NR' gets a diffuse one. Every other metabolite is diffuse in both groups.

    So abundance is null by construction, and organization is not.
    """
    rng = np.random.default_rng(seed)
    blocks, sections, groups = [], [], []

    for i in range(n_per_group * 2):
        group = "R" if i < n_per_group else "NR"
        section = np.empty((N_PIXELS, n_vars), dtype=np.float64)
        for j in range(n_vars):
            total = 1000.0 + rng.normal(0, 20)      # same distribution in both arms
            if j in organized_indices and group == "R":
                section[:, j] = _clustered_image(rng, total)
            else:
                section[:, j] = _diffuse_image(rng, total)
        blocks.append(section)
        sections += [f"S{i:02d}"] * N_PIXELS
        groups += [group] * N_PIXELS

    adata = ad.AnnData(X=np.vstack(blocks).astype(np.float32))
    adata.obsm["spatial"] = np.tile(COORDS, (n_per_group * 2, 1))
    adata.obs["section"] = sections
    adata.obs["response"] = groups
    adata.var_names = [f"m{j:03d}" for j in range(n_vars)]
    return adata


# ---------------------------------------------------------------------------
# The claim
# ---------------------------------------------------------------------------

class TestOrganizationOnlySignal:

    @pytest.fixture(scope="class")
    def cohort(self):
        return _organization_cohort(n_per_group=8, seed=1)

    def test_abundance_finds_nothing(self, cohort):
        """By construction the two groups have the same amount of everything."""
        pb = mt.pseudobulk(cohort, sample_key="section")
        res = mt.differential_abundance(pb, "response", "R", "NR")
        assert int((res["pval_adj"] < 0.05).sum()) == 0

    def test_organization_finds_the_planted_metabolites(self, cohort):
        org = mt.spatial_organization(cohort, sample_key="section", metrics=("morans_i",))
        res = mt.differential_spatial_organization(
            org, "response", "R", "NR", metric="morans_i"
        ).set_index("metabolite")

        planted = ["m000", "m001", "m002"]
        assert (res.loc[planted, "delta"] == 1.0).all(), (
            "clustered-in-R metabolites should separate completely on Moran's I"
        )
        assert (res.loc[planted, "pval_adj"] < 0.05).all()

        others = res.drop(index=planted)
        assert (others["pval_adj"] >= 0.05).all(), "unplanted metabolites should not be significant"

    def test_classifier_labels_them_organization_only(self, cohort):
        pb = mt.pseudobulk(cohort, sample_key="section")
        org = mt.spatial_organization(cohort, sample_key="section", metrics=("morans_i",))
        ab = mt.differential_abundance(pb, "response", "R", "NR")
        do = mt.differential_spatial_organization(org, "response", "R", "NR")

        merged = mt.compare_abundance_and_organization(ab, do).set_index("metabolite")
        assert (merged.loc[["m000", "m001", "m002"], "classification"] == "organization only").all()
        assert (merged["classification"] == "organization only").sum() == 3

    @staticmethod
    def _classify(n_per_group, **kwargs):
        cohort = _organization_cohort(n_per_group=n_per_group, seed=1)
        pb = mt.pseudobulk(cohort, sample_key="section")
        org = mt.spatial_organization(cohort, sample_key="section", metrics=("morans_i",))
        merged = mt.compare_abundance_and_organization(
            mt.differential_abundance(pb, "response", "R", "NR"),
            mt.differential_spatial_organization(org, "response", "R", "NR"),
            **kwargs,
        ).set_index("metabolite")
        return set(merged.index[merged["classification"] == "organization only"])

    def test_effect_size_alone_over_calls(self):
        """Why the classifier requires FDR by default: delta alone adds noise."""
        called = self._classify(5, fdr_threshold=None)
        assert {"m000", "m001", "m002"} <= called, "planted differences should be found"
        assert len(called) > 3, "effect size alone should over-call at n=5"

    def test_fdr_filter_is_exact_from_six_per_group(self):
        assert self._classify(6) == {"m000", "m001", "m002"}
        assert self._classify(8) == {"m000", "m001", "m002"}

    def test_below_six_per_group_fdr_returns_nothing(self):
        """
        Not a defect — a property of rank tests at tiny n. Mann-Whitney on 5 vs
        5 bottoms out at p = 0.0079, which cannot survive BH correction across
        30 metabolites even under perfect separation. Pinned here so the
        six-per-group floor documented in compare_abundance_and_organization
        stays honest.
        """
        assert self._classify(5) == set()
        assert self._classify(4) == set()

    def test_organization_only_findings_sort_first(self, cohort):
        pb = mt.pseudobulk(cohort, sample_key="section")
        org = mt.spatial_organization(cohort, sample_key="section", metrics=("morans_i",))
        merged = mt.compare_abundance_and_organization(
            mt.differential_abundance(pb, "response", "R", "NR"),
            mt.differential_spatial_organization(org, "response", "R", "NR"),
        )
        assert set(merged.head(3)["metabolite"]) == {"m000", "m001", "m002"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestOrganizationMetrics:

    @pytest.fixture(scope="class")
    def org(self):
        return mt.spatial_organization(
            _organization_cohort(n_per_group=3, n_vars=12, seed=2), sample_key="section"
        )

    def test_shape_and_layers(self, org):
        assert org.n_obs == 6
        assert org.n_vars == 12
        # Subset, not equality: anndata >= 0.13 also lists .X under a None key,
        # so demanding an exact match breaks on new anndata for no good reason.
        named = {k for k in org.layers.keys() if k is not None}
        assert set(mt.organization.ORGANIZATION_METRICS) <= named

    def test_metrics_are_in_expected_ranges(self, org):
        entropy = org.layers["entropy"]
        gini = org.layers["gini"]
        hotspot = org.layers["hotspot_fraction"]
        moran = org.layers["morans_i"]
        assert np.all((entropy >= 0) & (entropy <= 1.0 + 1e-6))
        assert np.all((gini >= -1e-6) & (gini <= 1.0 + 1e-6))
        assert np.all((hotspot >= 0) & (hotspot <= 1.0))
        assert np.all((moran >= -1.05) & (moran <= 1.05))

    def test_clustered_image_scores_higher_than_diffuse(self, org):
        """Direction check on every metric, using the planted contrast."""
        responder = org.obs["response"].values == "R"
        for metric, expect_higher in (
            ("morans_i", True), ("gini", True), ("hotspot_fraction", True), ("entropy", False),
        ):
            values = org.layers[metric][:, 0]
            r_mean, nr_mean = values[responder].mean(), values[~responder].mean()
            if expect_higher:
                assert r_mean > nr_mean, f"{metric}: clustered should score higher"
            else:
                assert r_mean < nr_mean, f"{metric}: clustered should score lower"

    def test_entropy_is_section_size_invariant(self):
        """
        The normalisation that matters: a bigger section must not score higher
        just for being bigger, or the metric measures biopsy size.
        """
        rng = np.random.default_rng(3)
        entropies = []
        for side in (12, 24, 36):
            n = side * side
            coords = np.array([[i % side, i // side] for i in range(n)], dtype=np.float64)
            adata = ad.AnnData(X=(rng.random((n, 4)) + 0.5).astype(np.float32))
            adata.obsm["spatial"] = coords
            adata.obs["section"] = "s"
            org = mt.spatial_organization(adata, sample_key="section", metrics=("entropy",))
            entropies.append(float(org.layers["entropy"].mean()))
        assert max(entropies) - min(entropies) < 0.02, (
            f"normalised entropy drifted with section size: {entropies}"
        )

    def test_sections_are_processed_independently(self):
        """Duplicating a section must not change that section's own metrics."""
        single = _organization_cohort(n_per_group=1, n_vars=6, seed=4)
        one = mt.spatial_organization(
            single[single.obs["section"] == "S00"].copy(), sample_key="section",
            metrics=("morans_i",),
        )
        many = mt.spatial_organization(single, sample_key="section", metrics=("morans_i",))
        np.testing.assert_allclose(
            one.layers["morans_i"][0],
            many.layers["morans_i"][list(many.obs["section"]).index("S00")],
            rtol=1e-5,
        )

    def test_carries_group_metadata(self, org):
        assert "response" in org.obs.columns
        assert "n_pixels" in org.obs.columns

    def test_records_provenance(self, org):
        assert org.uns["spatial_organization"]["sample_key"] == "section"
        assert org.uns["spatial_organization"]["n_samples"] == 6


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

class TestGuards:

    def test_requires_spatial_coordinates(self):
        adata = ad.AnnData(X=np.ones((10, 3), dtype=np.float32))
        adata.obs["section"] = "s"
        with pytest.raises(mt.MissingSpatialError):
            mt.spatial_organization(adata, sample_key="section")

    def test_rejects_unknown_metric(self):
        adata = _organization_cohort(n_per_group=1, n_vars=4, seed=5)
        with pytest.raises(InvalidParameterError, match="Unknown metric"):
            mt.spatial_organization(adata, sample_key="section", metrics=("fractal_dimension",))

    def test_skips_tiny_sections_with_warning(self):
        adata = _organization_cohort(n_per_group=1, n_vars=4, seed=6)
        keep = np.r_[np.arange(20), np.arange(N_PIXELS, adata.n_obs)]
        adata = adata[keep].copy()
        with pytest.warns(UserWarning, match="Skipped"):
            org = mt.spatial_organization(adata, sample_key="section", metrics=("morans_i",))
        assert org.n_obs == 1

    def test_differential_rejects_missing_metric(self):
        org = mt.spatial_organization(
            _organization_cohort(n_per_group=3, n_vars=4, seed=7),
            sample_key="section", metrics=("morans_i",),
        )
        with pytest.raises(InvalidParameterError, match="not found in org.layers"):
            mt.differential_spatial_organization(org, "response", "R", "NR", metric="gini")

    def test_differential_requires_enough_sections(self):
        org = mt.spatial_organization(
            _organization_cohort(n_per_group=2, n_vars=4, seed=8),
            sample_key="section", metrics=("morans_i",),
        )
        with pytest.raises(InsufficientSamplesError):
            mt.differential_spatial_organization(org, "response", "R", "NR", min_samples=3)

    def test_classifier_rejects_disjoint_inputs(self):
        import pandas as pd

        left = pd.DataFrame({"metabolite": ["a"], "delta": [0.5], "pval_adj": [0.1]})
        right = pd.DataFrame({"metabolite": ["z"], "delta": [0.5], "pval_adj": [0.1]})
        with pytest.raises(InvalidParameterError, match="No metabolites in common"):
            mt.compare_abundance_and_organization(left, right)
