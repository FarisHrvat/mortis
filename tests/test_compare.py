"""
Tests for cross-cohort and longitudinal signature comparison.

These functions produce a single headline number that a reader will quote, so
the tests are built around the three regimes that number is meant to
distinguish, persists, reorganises, flips, using signatures constructed to
sit unambiguously in each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mortis as mt
from mortis.exceptions import InsufficientSamplesError, InvalidParameterError


def _result(deltas, names=None):
    """Minimal stand-in for a differential_abundance table."""
    deltas = np.asarray(deltas, dtype=float)
    names = names if names is not None else [f"m{i:03d}" for i in range(len(deltas))]
    return pd.DataFrame({
        "metabolite": names,
        "delta": deltas,
        "pval_adj": np.full(len(deltas), 0.01),
    })


@pytest.fixture
def signature():
    rng = np.random.default_rng(0)
    return rng.uniform(-1, 1, 60)


# ---------------------------------------------------------------------------
# The three regimes
# ---------------------------------------------------------------------------

class TestRegimes:

    def test_identical_signatures_give_rho_one(self, signature):
        rho, table = mt.cross_cohort_profile(_result(signature), _result(signature))
        assert rho == pytest.approx(1.0)
        assert (table["agreement"] != "discordant").all()

    def test_inverted_signature_flips(self, signature):
        rho, table = mt.track_flow(_result(signature), _result(-signature))
        assert rho == pytest.approx(-1.0)
        strong = table[table["agreement"] != "weak"]
        assert (strong["agreement"] == "discordant").all()

    def test_unrelated_signatures_give_rho_near_zero(self):
        rng = np.random.default_rng(1)
        rho, _ = mt.cross_cohort_profile(
            _result(rng.uniform(-1, 1, 400)), _result(rng.uniform(-1, 1, 400))
        )
        assert abs(rho) < 0.2

    def test_partial_agreement_lands_between(self, signature):
        rng = np.random.default_rng(2)
        noisy = signature + rng.normal(0, 0.6, signature.size)
        rho, _ = mt.track_flow(_result(signature), _result(noisy))
        assert 0.3 < rho < 0.95


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------

class TestBehaviour:

    def test_only_shared_metabolites_are_compared(self, signature):
        a = _result(signature)
        b = _result(signature[:40], names=[f"m{i:03d}" for i in range(40)])
        _, table = mt.cross_cohort_profile(a, b)
        assert len(table) == 40

    def test_metabolite_order_does_not_matter(self, signature):
        a = _result(signature)
        shuffled = a.sample(frac=1.0, random_state=3).reset_index(drop=True)
        rho_ordered, _ = mt.cross_cohort_profile(a, a)
        rho_shuffled, _ = mt.cross_cohort_profile(a, shuffled)
        assert rho_ordered == pytest.approx(rho_shuffled)

    def test_symmetric(self, signature):
        rng = np.random.default_rng(4)
        other = _result(signature + rng.normal(0, 0.5, signature.size))
        forward, _ = mt.cross_cohort_profile(_result(signature), other)
        reverse, _ = mt.cross_cohort_profile(other, _result(signature))
        assert forward == pytest.approx(reverse)

    def test_agreement_labels(self):
        a = _result([0.9, -0.9, 0.9, 0.05] * 5)
        b = _result([0.8, -0.8, -0.8, 0.9] * 5)
        _, table = mt.cross_cohort_profile(a, b)
        labels = table.set_index("metabolite")["agreement"]
        assert labels["m000"] == "concordant"
        assert labels["m001"] == "concordant"   # both negative
        assert labels["m002"] == "discordant"
        assert labels["m003"] == "weak"         # |0.05| below the weak cutoff

    def test_bootstrap_interval_brackets_rho(self, signature):
        rng = np.random.default_rng(5)
        noisy = _result(signature + rng.normal(0, 0.4, signature.size))
        rho, _, table = mt.compare_signatures(
            _result(signature), noisy, n_boot=300, random_state=0
        )
        assert table.attrs["rho_ci_low"] <= rho <= table.attrs["rho_ci_high"]

    def test_bootstrap_is_deterministic(self, signature):
        rng = np.random.default_rng(6)
        noisy = _result(signature + rng.normal(0, 0.4, signature.size))
        first = mt.compare_signatures(_result(signature), noisy, n_boot=200, random_state=1)[2]
        second = mt.compare_signatures(_result(signature), noisy, n_boot=200, random_state=1)[2]
        assert first.attrs["rho_ci_low"] == second.attrs["rho_ci_low"]

    def test_works_on_organization_results(self):
        """The same machinery must accept differential_spatial_organization output."""
        rng = np.random.default_rng(7)
        d = rng.uniform(-1, 1, 40)
        rho, _ = mt.cross_cohort_profile(_result(d), _result(d), labels=("org_a", "org_b"))
        assert rho == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

class TestGuards:

    def test_rejects_too_few_shared(self, signature):
        a = _result(signature)
        b = _result(signature[:4], names=[f"m{i:03d}" for i in range(4)])
        with pytest.raises(InsufficientSamplesError, match="shared"):
            mt.cross_cohort_profile(a, b)

    def test_rejects_missing_columns(self, signature):
        bad = pd.DataFrame({"metabolite": [f"m{i:03d}" for i in range(60)]})
        with pytest.raises(InvalidParameterError, match="delta"):
            mt.cross_cohort_profile(_result(signature), bad)

    def test_rejects_duplicate_metabolites(self, signature):
        dup = _result(signature)
        dup.loc[1, "metabolite"] = dup.loc[0, "metabolite"]
        with pytest.raises(InvalidParameterError, match="duplicate"):
            mt.cross_cohort_profile(_result(signature), dup)

    def test_rejects_constant_effects(self, signature):
        constant = _result(np.zeros_like(signature))
        with pytest.raises(InvalidParameterError, match="constant"):
            mt.cross_cohort_profile(_result(signature), constant)

    def test_rejects_non_dataframe(self, signature):
        with pytest.raises(InvalidParameterError, match="DataFrame"):
            mt.cross_cohort_profile(_result(signature), {"metabolite": [], "delta": []})


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_end_to_end_from_real_pipeline():
    """
    Two cohorts analysed independently through the full pipeline, sharing a
    planted responder signature. The comparison should recover it.
    """
    import anndata as ad

    def cohort(seed, sign=1.0):
        rng = np.random.default_rng(seed)
        n_vars, n_pixels = 40, 120
        blocks, patients, groups = [], [], []
        for i in range(12):
            group = "R" if i < 6 else "NR"
            offset = rng.normal(0, 1.0, n_vars)
            if group == "R":
                offset[:8] += 6.0 * sign
            blocks.append(rng.normal(offset, 1.0, (n_pixels, n_vars)))
            patients += [f"P{i}"] * n_pixels
            groups += [group] * n_pixels
        X = np.vstack(blocks).astype(np.float32)
        X -= X.min()
        a = ad.AnnData(X=X)
        a.obs["patient"], a.obs["response"] = patients, groups
        a.var_names = [f"m{j:03d}" for j in range(n_vars)]
        return mt.differential_abundance(
            mt.pseudobulk(a, sample_key="patient"), "response", "R", "NR"
        )

    # Only 8 of 40 metabolites carry planted signal; the other 32 contribute
    # random ranks. So |rho| is capped well below 1 here by construction,
    # roughly the signal fraction. What matters is the sign and that the
    # discordant/concordant split flips cleanly, not the magnitude.
    rho_same, table_same = mt.cross_cohort_profile(cohort(10), cohort(11))
    assert rho_same > 0.4, "cohorts sharing a signature should correlate positively"
    strong_same = table_same[table_same["agreement"] != "weak"]
    assert (strong_same["agreement"] == "concordant").mean() > 0.8

    rho_opposite, table_opposite = mt.cross_cohort_profile(cohort(10), cohort(12, sign=-1.0))
    assert rho_opposite < -0.4, "an inverted planted signature should anti-correlate"
    strong_opposite = table_opposite[table_opposite["agreement"] != "weak"]
    assert (strong_opposite["agreement"] == "discordant").mean() > 0.8

