"""
MORTIS Cross-Cohort Comparison Module
=====================================
Comparing one cohort analysis against another: two drugs, two timepoints, two
studies.

The operation underneath every function here is the same — take the
per-metabolite effect sizes from two independent analyses and correlate them —
but what the correlation *means* depends entirely on what the two analyses
were. Two drugs gives you "do these drugs act on the same metabolites?". A
baseline and an on-treatment analysis of the same patients gives you "does the
signature persist or reorganise?". So the shared machinery lives in
:func:`compare_signatures` and the interpretation lives in the wrappers.

The one rule that matters
-------------------------
**Run each cohort through the identical pipeline first, then compare.** Analyse
one cohort as an appendix of the other — carrying clusters, normalisation
factors or feature selection across — and the comparison measures your
processing choices rather than the biology. Reviewers reject asymmetric
comparisons, correctly. Every function here takes two *finished, independent*
result tables, which makes the symmetric pattern the path of least resistance.

Spearman is used throughout rather than Pearson: effect sizes are already rank
statistics, and with a few hundred metabolites a handful of extreme values
would otherwise dominate a Pearson coefficient.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .exceptions import InsufficientSamplesError, InvalidParameterError

__all__ = ["compare_signatures", "cross_cohort_profile", "track_flow"]

#: |rho| below this counts as "no relationship" when labelling an outcome.
#: 0.3 is the conventional weak/moderate boundary for a rank correlation.
_RHO_WEAK = 0.3


def _aligned_effects(
    result_a: pd.DataFrame,
    result_b: pd.DataFrame,
    effect_col: str,
    min_shared: int,
    labels: Tuple[str, str],
) -> pd.DataFrame:
    """Inner-join two result tables on metabolite and return their effect sizes."""
    for name, frame in zip(labels, (result_a, result_b)):
        if not isinstance(frame, pd.DataFrame):
            raise InvalidParameterError(f"'{name}' must be a pandas DataFrame.")
        for column in ("metabolite", effect_col):
            if column not in frame.columns:
                raise InvalidParameterError(
                    f"'{name}' is missing required column '{column}'. "
                    "Pass the output of mortis.differential_abundance()."
                )
        if frame["metabolite"].duplicated().any():
            raise InvalidParameterError(f"'{name}' contains duplicate metabolite names.")

    merged = result_a[["metabolite", effect_col]].merge(
        result_b[["metabolite", effect_col]],
        on="metabolite", suffixes=(f"_{labels[0]}", f"_{labels[1]}"), how="inner",
    )
    if len(merged) < min_shared:
        raise InsufficientSamplesError(
            f"Only {len(merged)} metabolite(s) are shared between the two results, "
            f"need at least min_shared={min_shared}. Were both analyses run on the "
            "same feature set?"
        )
    return merged


def compare_signatures(
    result_a: pd.DataFrame,
    result_b: pd.DataFrame,
    effect_col: str = "delta",
    labels: Tuple[str, str] = ("a", "b"),
    min_shared: int = 10,
    n_boot: int = 0,
    random_state: int = 0,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Correlate the per-metabolite effect sizes of two independent analyses.

    Parameters
    ----------
    result_a, result_b : pandas.DataFrame
        Outputs of :func:`mortis.differential_abundance` (or
        :func:`mortis.differential_spatial_organization`), each computed
        independently on its own cohort.
    effect_col : str
        Column holding the effect size. Default ``"delta"``.
    labels : tuple of str
        Short names for the two analyses, used as column suffixes.
    min_shared : int
        Minimum metabolites in common. Default 10 — below that a correlation
        is not interpretable.
    n_boot : int
        Bootstrap resamples for a confidence interval on rho. Default 0.
    random_state : int
        Seed for the bootstrap.

    Returns
    -------
    (rho, pval, table)
        Spearman correlation, its p-value, and a per-metabolite DataFrame with
        both effect sizes, their ranks, and an ``agreement`` label
        (``"concordant"``, ``"discordant"``, or ``"weak"`` when either effect
        is near zero).

        When ``n_boot > 0`` the returned table carries ``rho_ci_low`` and
        ``rho_ci_high`` in ``table.attrs``.
    """
    merged = _aligned_effects(result_a, result_b, effect_col, min_shared, labels)
    col_a, col_b = f"{effect_col}_{labels[0]}", f"{effect_col}_{labels[1]}"
    a, b = merged[col_a].to_numpy(dtype=float), merged[col_b].to_numpy(dtype=float)

    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        raise InvalidParameterError(
            "One of the effect-size vectors is constant, so a rank correlation is undefined."
        )

    rho, pval = stats.spearmanr(a, b)

    merged[f"rank_{labels[0]}"] = stats.rankdata(-np.abs(a))
    merged[f"rank_{labels[1]}"] = stats.rankdata(-np.abs(b))

    weak = (np.abs(a) < _RHO_WEAK) | (np.abs(b) < _RHO_WEAK)
    merged["agreement"] = np.where(
        weak, "weak", np.where(np.sign(a) == np.sign(b), "concordant", "discordant")
    )
    merged.attrs["rho"], merged.attrs["pval"] = float(rho), float(pval)

    if n_boot > 0:
        rng = np.random.default_rng(random_state)
        n = len(merged)
        draws = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            draws[i] = stats.spearmanr(a[idx], b[idx])[0]
        draws = draws[np.isfinite(draws)]
        merged.attrs["rho_ci_low"] = float(np.percentile(draws, 2.5))
        merged.attrs["rho_ci_high"] = float(np.percentile(draws, 97.5))

    return float(rho), float(pval), merged


def cross_cohort_profile(
    result_a: pd.DataFrame,
    result_b: pd.DataFrame,
    labels: Tuple[str, str] = ("cohort_a", "cohort_b"),
    effect_col: str = "delta",
    min_shared: int = 10,
    n_boot: int = 0,
    random_state: int = 0,
) -> Tuple[float, pd.DataFrame]:
    """
    Do two cohorts show the *same* signature? Typically: two drugs.

    Correlates the per-metabolite effect sizes from two separately analysed
    cohorts. The result is interpreted as:

    ==================  =====================================================
    rho                 reading
    ==================  =====================================================
    near +1             the same metabolites move the same way — a shared
                        response signature
    near 0              the signatures are unrelated — response is
                        cohort-specific (drug-specific, site-specific)
    near -1             the same metabolites move in opposite directions,
                        which is a strong and unusual claim; check first that
                        the group labels were not swapped in one analysis
    ==================  =====================================================

    A near-zero rho is a real, reportable finding, not a failed experiment. It
    says the two treatments do not converge on a common metabolic response, and
    that is often the more interesting answer.

    Both inputs must come from cohorts run through the *same* pipeline
    independently — see the module docstring.

    Returns
    -------
    (rho, table)
        Spearman rho and the per-metabolite comparison table.
    """
    rho, pval, table = compare_signatures(
        result_a, result_b, effect_col=effect_col, labels=labels,
        min_shared=min_shared, n_boot=n_boot, random_state=random_state,
    )
    counts = table["agreement"].value_counts()
    if abs(rho) < _RHO_WEAK:
        reading = "signatures are largely unrelated - the response looks cohort-specific"
    elif rho > 0:
        reading = "signatures agree - a shared response signature"
    else:
        reading = "signatures are inverted - check the group labels were not swapped"

    print(
        f"[MORTIS] Cross-cohort {labels[0]} vs {labels[1]}: rho={rho:.3f} (p={pval:.3g}) "
        f"over {len(table)} shared metabolites; {reading}."
    )
    print(
        f"[MORTIS]   concordant={counts.get('concordant', 0)}, "
        f"discordant={counts.get('discordant', 0)}, weak={counts.get('weak', 0)}"
    )
    return rho, table


def track_flow(
    baseline: pd.DataFrame,
    on_treatment: pd.DataFrame,
    labels: Tuple[str, str] = ("baseline", "on_treatment"),
    effect_col: str = "delta",
    min_shared: int = 10,
    n_boot: int = 0,
    random_state: int = 0,
) -> Tuple[float, pd.DataFrame]:
    """
    Does a signature *persist* over treatment, flip, or reorganise?

    Both inputs are responder-vs-non-responder comparisons: one computed at
    baseline, one on treatment. Correlating them asks what happened to the
    separating signature between the two timepoints.

    ==================  =====================================================
    rho                 reading
    ==================  =====================================================
    near +1             **persists** — the same metabolites separate
                        responders at both timepoints
    near 0              **reorganises** — responders are still distinguishable
                        but by a different set of metabolites
    near -1             **flips** — the direction of separation reverses
    ==================  =====================================================

    Note this is a different question from :func:`mortis.paired_differential_abundance`,
    which asks how each patient changed. This asks how the *between-group
    difference* changed, so it does not require matched patients at the two
    timepoints — only that both analyses used the same feature set.

    Returns
    -------
    (rho, table)
        Spearman rho and the per-metabolite comparison table.
    """
    rho, pval, table = compare_signatures(
        baseline, on_treatment, effect_col=effect_col, labels=labels,
        min_shared=min_shared, n_boot=n_boot, random_state=random_state,
    )
    if abs(rho) < _RHO_WEAK:
        verdict = "REORGANISES - a different metabolite set separates the groups on treatment"
    elif rho > 0:
        verdict = "PERSISTS - the same metabolites separate the groups at both timepoints"
    else:
        verdict = "FLIPS - the direction of separation reverses on treatment"

    print(
        f"[MORTIS] Signature flow {labels[0]} -> {labels[1]}: rho={rho:.3f} (p={pval:.3g}) "
        f"over {len(table)} shared metabolites."
    )
    print(f"[MORTIS]   {verdict}")
    return rho, table
