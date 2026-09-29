"""
Tests for sample-level statistics.

The important test in this file is
``TestFalsePositiveControl::test_null_data_yields_almost_no_hits``. It is the
reason the module exists: it simulates a cohort with no group difference and
asserts that the sample-level path does not invent one. Everything else here
supports that claim.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

import mortis as mt
from mortis.exceptions import InsufficientSamplesError, InvalidParameterError


def _cohort(n_per_group=4, n_pixels=200, n_vars=60, effect=0.0, seed=0):
    """
    Cohort with realistic nesting: each patient gets its own random offset, and
    pixels vary within the patient. ``effect`` shifts the first 10 metabolites
    in group 'R' only, so the null case (effect=0) has no group signal at all.
    """
    rng = np.random.default_rng(seed)
    blocks, patients, groups = [], [], []
    for i in range(n_per_group * 2):
        group = "R" if i < n_per_group else "NR"
        offset = rng.normal(0.0, 1.0, n_vars)          # patient-level variation
        if group == "R":
            offset[:10] += effect
        blocks.append(rng.normal(offset, 1.0, (n_pixels, n_vars)))
        patients += [f"P{i:02d}"] * n_pixels
        groups += [group] * n_pixels

    X = np.vstack(blocks).astype(np.float32)
    X -= X.min()                                        # keep intensities non-negative
    adata = ad.AnnData(X=X)
    adata.obs["patient"] = patients
    adata.obs["response"] = groups
    adata.obs["cluster"] = rng.integers(0, 3, adata.n_obs).astype(str)  # varies within patient
    adata.var_names = [f"m{j:03d}" for j in range(n_vars)]
    return adata


# ---------------------------------------------------------------------------
# The headline claim
# ---------------------------------------------------------------------------

class TestFalsePositiveControl:

    def test_null_data_yields_almost_no_hits(self):
        """No true group effect -> the sample-level test must stay quiet."""
        adata = _cohort(n_per_group=5, effect=0.0, seed=1)
        pb = mt.pseudobulk(adata, sample_key="patient")
        res = mt.differential_abundance(pb, "response", "R", "NR")

        n_sig = int((res["pval_adj"] < 0.05).sum())
        assert n_sig <= 1, f"{n_sig}/{adata.n_vars} false positives on null data"

    def test_pixel_level_inflates_and_sample_level_does_not(self):
        """Both paths on identical null data; only the pixel path fabricates hits."""
        adata = _cohort(n_per_group=3, effect=0.0, seed=2)

        with pytest.warns(mt.PseudoreplicationWarning):
            _, pixel_res = mt.compare_groups(adata, "response", "R", "NR")
        pixel_hits = int(pixel_res["significant"].sum())

        pb = mt.pseudobulk(adata, sample_key="patient")
        sample_hits = int((mt.differential_abundance(pb, "response", "R", "NR")["pval_adj"] < 0.05).sum())

        assert pixel_hits > 10 * max(sample_hits, 1), (
            "expected the pixel-level test to over-call relative to the sample-level test"
        )
        assert sample_hits <= 1

    def test_real_effect_is_still_detected(self):
        """Controlling false positives is worthless if it also kills true signal."""
        adata = _cohort(n_per_group=6, effect=6.0, seed=3)
        pb = mt.pseudobulk(adata, sample_key="patient")
        res = mt.differential_abundance(pb, "response", "R", "NR").set_index("metabolite")

        spiked = [f"m{j:03d}" for j in range(10)]
        assert (res.loc[spiked, "delta"] > 0.6).sum() >= 8, "true effects were not recovered"

        untouched = res.drop(index=spiked)
        assert untouched["delta"].abs().median() < 0.5


# ---------------------------------------------------------------------------
# Pseudobulk
# ---------------------------------------------------------------------------

class TestPseudobulk:

    def test_shape_and_pixel_counts(self):
        adata = _cohort(n_per_group=4, n_pixels=150)
        pb = mt.pseudobulk(adata, sample_key="patient")
        assert pb.n_obs == 8
        assert pb.n_vars == adata.n_vars
        assert (pb.obs["n_pixels"] == 150).all()

    def test_carries_constant_obs_and_drops_varying_ones(self):
        adata = _cohort(n_per_group=3)
        pb = mt.pseudobulk(adata, sample_key="patient")
        assert "response" in pb.obs.columns, "sample-constant metadata should carry over"
        assert "cluster" not in pb.obs.columns, "pixel-varying metadata must not be carried"

    def test_explicit_carry_obs(self):
        adata = _cohort(n_per_group=3)
        pb = mt.pseudobulk(adata, sample_key="patient", carry_obs=["response"])
        assert list(pb.obs.columns) == ["patient", "response", "n_pixels"]

    @pytest.mark.parametrize("method", ["mean", "median", "sum"])
    def test_aggregation_methods_agree_with_numpy(self, method):
        adata = _cohort(n_per_group=2, n_pixels=50, n_vars=10)
        pb = mt.pseudobulk(adata, sample_key="patient", method=method)
        fn = {"mean": np.mean, "median": np.median, "sum": np.sum}[method]
        first = adata.obs["patient"].iloc[0]
        expected = fn(np.asarray(adata.X)[adata.obs["patient"].values == first], axis=0)
        np.testing.assert_allclose(np.asarray(pb.X)[0], expected, rtol=1e-5)

    def test_drops_samples_below_min_pixels(self):
        adata = _cohort(n_per_group=2, n_pixels=50)
        adata = adata[np.r_[np.arange(5), np.arange(50, adata.n_obs)]].copy()  # starve one patient
        with pytest.warns(UserWarning, match="Dropped"):
            pb = mt.pseudobulk(adata, sample_key="patient", min_pixels=10)
        assert pb.n_obs == 3

    def test_records_provenance(self):
        adata = _cohort(n_per_group=2)
        pb = mt.pseudobulk(adata, sample_key="patient", method="median")
        assert pb.uns["pseudobulk"]["method"] == "median"
        assert pb.uns["pseudobulk"]["sample_key"] == "patient"

    def test_rejects_bad_input(self):
        adata = _cohort(n_per_group=2)
        with pytest.raises(InvalidParameterError):
            mt.pseudobulk(adata, sample_key="not_a_column")
        with pytest.raises(InvalidParameterError):
            mt.pseudobulk(adata, sample_key="patient", method="geometric")


# ---------------------------------------------------------------------------
# Cliff's delta
# ---------------------------------------------------------------------------

class TestCliffsDelta:

    def test_complete_separation(self):
        a = np.array([[10.0], [11.0], [12.0]])
        b = np.array([[1.0], [2.0], [3.0]])
        assert mt.cliffs_delta(a, b)[0] == pytest.approx(1.0)
        assert mt.cliffs_delta(b, a)[0] == pytest.approx(-1.0)

    def test_identical_groups_give_zero(self):
        a = np.array([[1.0], [2.0], [3.0]])
        assert mt.cliffs_delta(a, a.copy())[0] == pytest.approx(0.0)

    def test_matches_mann_whitney_identity(self):
        """delta = 2U/(n1*n2) - 1 for scipy's U, which counts wins plus half-ties."""
        from scipy import stats as sps

        rng = np.random.default_rng(11)
        a, b = rng.normal(0.5, 1, (9, 40)), rng.normal(0, 1, (7, 40))
        u, _ = sps.mannwhitneyu(a, b, axis=0, alternative="two-sided")
        np.testing.assert_allclose(
            mt.cliffs_delta(a, b), 2 * u / (a.shape[0] * b.shape[0]) - 1, atol=1e-12
        )

    def test_handles_ties(self):
        a = np.array([[1.0], [1.0], [2.0]])
        b = np.array([[1.0], [1.0], [1.0]])
        # one of three a-values exceeds all of b, none fall below.
        assert mt.cliffs_delta(a, b)[0] == pytest.approx(3 / 9)

    def test_chunking_matches_unchunked(self):
        rng = np.random.default_rng(5)
        a, b = rng.normal(0, 1, (6, 900)), rng.normal(0, 1, (6, 900))
        combined = mt.cliffs_delta(a, b)
        per_feature = np.array([mt.cliffs_delta(a[:, [j]], b[:, [j]])[0] for j in range(0, 900, 97)])
        np.testing.assert_allclose(combined[::97], per_feature, atol=1e-12)


# ---------------------------------------------------------------------------
# Differential abundance
# ---------------------------------------------------------------------------

class TestDifferentialAbundance:

    def test_direction_convention(self):
        """Positive delta must mean higher in group1, as documented."""
        pb = mt.pseudobulk(_cohort(n_per_group=5, effect=8.0, seed=4), sample_key="patient")
        res = mt.differential_abundance(pb, "response", "R", "NR").set_index("metabolite")
        assert res.loc["m000", "delta"] > 0, "group1 was spiked, so delta should be positive"

        flipped = mt.differential_abundance(pb, "response", "NR", "R").set_index("metabolite")
        assert flipped.loc["m000", "delta"] == pytest.approx(-res.loc["m000", "delta"])

    def test_sorted_by_absolute_effect(self):
        pb = mt.pseudobulk(_cohort(n_per_group=5, effect=5.0, seed=6), sample_key="patient")
        deltas = mt.differential_abundance(pb, "response", "R", "NR")["delta"].abs().to_numpy()
        assert np.all(np.diff(deltas) <= 1e-12)

    def test_reports_sample_counts_not_pixel_counts(self):
        adata = _cohort(n_per_group=4, n_pixels=500)
        pb = mt.pseudobulk(adata, sample_key="patient")
        res = mt.differential_abundance(pb, "response", "R", "NR")
        assert (res["n_group1"] == 4).all() and (res["n_group2"] == 4).all()

    def test_refuses_pixel_level_input(self):
        """A raw pixel object has thousands of 'samples' but no sample structure."""
        adata = _cohort(n_per_group=4)
        with pytest.raises(InsufficientSamplesError, match="pseudobulk"):
            mt.differential_abundance(adata, "response", "R", "NR", min_samples=10_000)

    def test_bootstrap_interval(self):
        pb = mt.pseudobulk(_cohort(n_per_group=6, effect=8.0, seed=7), sample_key="patient")
        res = mt.differential_abundance(
            pb, "response", "R", "NR", bootstrap=200, random_state=0
        ).set_index("metabolite")
        assert (res["delta_ci_low"] <= res["delta"]).all()
        assert (res["delta"] <= res["delta_ci_high"]).all()
        assert bool(res.loc["m000", "ci_excludes_zero"])

    def test_bootstrap_is_deterministic(self):
        pb = mt.pseudobulk(_cohort(n_per_group=5, effect=4.0, seed=8), sample_key="patient")
        kwargs = dict(bootstrap=100, random_state=3)
        a = mt.differential_abundance(pb, "response", "R", "NR", **kwargs)
        b = mt.differential_abundance(pb, "response", "R", "NR", **kwargs)
        np.testing.assert_array_equal(a["delta_ci_low"].to_numpy(), b["delta_ci_low"].to_numpy())

    def test_rejects_bad_groups(self):
        pb = mt.pseudobulk(_cohort(n_per_group=3), sample_key="patient")
        with pytest.raises(InvalidParameterError):
            mt.differential_abundance(pb, "response", "R", "Missing")
        with pytest.raises(InvalidParameterError):
            mt.differential_abundance(pb, "response", "R", "R")


# ---------------------------------------------------------------------------
# Paired / longitudinal
# ---------------------------------------------------------------------------

def _longitudinal(n_patients=6, n_pixels=80, n_vars=40, effect=0.0, seed=0):
    rng = np.random.default_rng(seed)
    blocks, patients, weeks = [], [], []
    for i in range(n_patients):
        baseline = rng.normal(0.0, 1.0, n_vars)
        for week in ("W0", "W14"):
            shift = effect if week == "W14" else 0.0
            profile = baseline.copy()
            profile[:8] += shift
            blocks.append(rng.normal(profile, 0.5, (n_pixels, n_vars)))
            patients += [f"P{i}"] * n_pixels
            weeks += [week] * n_pixels
    X = np.vstack(blocks).astype(np.float32)
    X -= X.min()
    adata = ad.AnnData(X=X)
    adata.obs["patient"] = patients
    adata.obs["week"] = weeks
    adata.obs["sample_id"] = [f"{p}_{w}" for p, w in zip(patients, weeks)]
    adata.var_names = [f"m{j:03d}" for j in range(n_vars)]
    return adata


class TestPairedDifferentialAbundance:

    def test_detects_within_patient_change(self):
        adata = _longitudinal(n_patients=8, effect=5.0, seed=9)
        pb = mt.pseudobulk(adata, sample_key="sample_id")
        res = mt.paired_differential_abundance(pb, "patient", "week", "W0", "W14").set_index("metabolite")
        spiked = [f"m{j:03d}" for j in range(8)]
        assert (res.loc[spiked, "delta"] > 0.9).all(), "W14 increase should give delta near +1"

    def test_null_change_stays_quiet(self):
        adata = _longitudinal(n_patients=8, effect=0.0, seed=10)
        pb = mt.pseudobulk(adata, sample_key="sample_id")
        res = mt.paired_differential_abundance(pb, "patient", "week", "W0", "W14")
        assert int((res["pval_adj"] < 0.05).sum()) <= 1

    def test_direction_convention(self):
        """Positive delta means higher at time2, the later timepoint."""
        adata = _longitudinal(n_patients=6, effect=5.0, seed=12)
        pb = mt.pseudobulk(adata, sample_key="sample_id")
        forward = mt.paired_differential_abundance(pb, "patient", "week", "W0", "W14").set_index("metabolite")
        reverse = mt.paired_differential_abundance(pb, "patient", "week", "W14", "W0").set_index("metabolite")
        assert forward.loc["m000", "delta"] > 0
        assert reverse.loc["m000", "delta"] == pytest.approx(-forward.loc["m000", "delta"])

    def test_unmatched_subjects_are_dropped_with_warning(self):
        adata = _longitudinal(n_patients=6, effect=3.0, seed=13)
        # Remove P0's W14 section entirely, leaving it unmatched.
        adata = adata[~((adata.obs["patient"] == "P0") & (adata.obs["week"] == "W14"))].copy()
        pb = mt.pseudobulk(adata, sample_key="sample_id")
        with pytest.warns(UserWarning, match="lack both timepoints"):
            res = mt.paired_differential_abundance(pb, "patient", "week", "W0", "W14")
        assert (res["n_pairs"] == 5).all()

    def test_requires_enough_pairs(self):
        adata = _longitudinal(n_patients=2, seed=14)
        pb = mt.pseudobulk(adata, sample_key="sample_id")
        with pytest.raises(InsufficientSamplesError, match="min_pairs"):
            mt.paired_differential_abundance(pb, "patient", "week", "W0", "W14", min_pairs=3)

