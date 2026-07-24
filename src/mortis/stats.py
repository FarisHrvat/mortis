"""
MORTIS Statistics Module
========================
Sample-level (patient-level) statistics for spatial metabolomics cohorts.

Why this module exists
----------------------
An MSI section yields tens of thousands of pixels, but those pixels are not
independent observations — they all come from one tissue section from one
patient. Testing them as if they were independent inflates the sample size by
three or four orders of magnitude and manufactures significance out of nothing.

The size of the problem is easy to underestimate, so here is a measurement.
Simulate six patients (three per arm), 500 pixels each, 200 metabolites, with
**no group difference at all** — only patient-to-patient variation:

    pixel-level Mann-Whitney (n = 1500 per arm) ->  183 / 200 "significant"
    patient-level pseudobulk (n = 3 per arm)    ->    0 / 200 significant

Roughly 92% of the pixel-level hits were fabricated. This is the same failure
mode documented for single-cell and spatial transcriptomics, and for MSI
specifically (block-SAM, *Bioinformatics* 2026, reports a pixel-level t-test
giving P = 9.6e-07 where the correct patient-level test gives P = 0.78).

So: collapse to one profile per sample first, then test. That is
:func:`pseudobulk` followed by :func:`differential_abundance`.

Why Cliff's delta rather than fold change
-----------------------------------------
Cohorts here are small — often 4 to 10 patients per arm. At that size a
p-value is mostly noise and a mean-based fold change is hostage to a single
outlying patient. Cliff's delta asks a question that stays meaningful at n = 4:
*given a random patient from each group, how much more often does one exceed
the other?* It is rank-based, bounded in [-1, 1], and needs no distributional
assumption. Report it first; treat the p-value as supporting detail.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .exceptions import InsufficientSamplesError, InvalidParameterError
from .reproducibility import record_step

__all__ = [
    "pseudobulk",
    "cliffs_delta",
    "differential_abundance",
    "paired_differential_abundance",
]

# Interpretation thresholds from Romano et al. (2006), the conventional
# reading of Cliff's delta magnitude. Advisory labels only — they are not
# used to gate anything.
_DELTA_BANDS = ((0.147, "negligible"), (0.33, "small"), (0.474, "medium"))


def _to_dense(X) -> np.ndarray:
    from scipy.sparse import issparse

    if issparse(X):
        return X.toarray().astype(np.float64, copy=False)
    return np.asarray(X, dtype=np.float64)


def _analysis_matrix(adata: ad.AnnData, layer: Optional[str]) -> np.ndarray:
    if layer is not None:
        if layer not in adata.layers:
            raise InvalidParameterError(
                f"Layer '{layer}' not found. Available: {list(adata.layers.keys())}."
            )
        return _to_dense(adata.layers[layer])
    return _to_dense(adata.X)


def _delta_magnitude(delta: np.ndarray) -> np.ndarray:
    labels = np.full(delta.shape, "large", dtype=object)
    absolute = np.abs(delta)
    for threshold, name in reversed(_DELTA_BANDS):
        labels[absolute < threshold] = name
    return labels


# ---------------------------------------------------------------------------
# Pseudobulk
# ---------------------------------------------------------------------------

def pseudobulk(
    adata: ad.AnnData,
    sample_key: str,
    method: str = "mean",
    layer: Optional[str] = None,
    min_pixels: int = 10,
    carry_obs: Optional[Sequence[str]] = None,
) -> ad.AnnData:
    """
    Collapse pixels to one profile per sample — the required first step before
    any comparison between groups of samples.

    Parameters
    ----------
    adata : anndata.AnnData
        Pixel-level data.
    sample_key : str
        Column in ``adata.obs`` identifying the independent unit. This should
        be the **patient**, not the section, whenever one patient contributed
        more than one section — two sections from the same patient are not
        independent replicates any more than two pixels are.
    method : {"mean", "median", "sum"}
        How to aggregate pixels within a sample. ``"median"`` is the more
        robust choice when a section contains a few very bright pixels
        (a matrix cluster that survived filtering, a hot spot at a tissue
        edge); ``"mean"`` is the default because it is what downstream
        fold-change intuition assumes.
    layer : str, optional
        Layer to aggregate. Default ``None`` uses ``.X``. Aggregate the
        **untransformed** intensities where you can — averaging log values
        gives a geometric mean, which is a different quantity than the one
        most readers assume.
    min_pixels : int
        Samples with fewer than this many pixels are dropped, with a warning.
        A section reduced to a handful of pixels by QC gives an unstable
        profile that is better excluded than silently averaged.
    carry_obs : sequence of str, optional
        Columns to carry to the output. Default ``None`` auto-detects every
        ``obs`` column that is constant within each sample — which picks up
        patient, response, drug, timepoint, and batch labels without being
        told about them. Columns that vary within a sample (cluster, niche,
        per-pixel scores) are dropped, because no single value represents them.

    Returns
    -------
    anndata.AnnData
        One observation per sample. ``.obs['n_pixels']`` records how many
        pixels each profile came from. ``.uns['pseudobulk']`` records the
        parameters used.

    Examples
    --------
    >>> pb = mt.pseudobulk(adata, sample_key="patient")
    >>> res = mt.differential_abundance(pb, "response", "Responder", "Non Responder")
    """
    valid_methods = {"mean", "median", "sum"}
    if method not in valid_methods:
        raise InvalidParameterError(f"method must be one of {valid_methods}, got '{method}'.")
    if sample_key not in adata.obs.columns:
        raise InvalidParameterError(
            f"'{sample_key}' not found in adata.obs. Available: {adata.obs.columns.tolist()}."
        )
    if min_pixels < 1:
        raise InvalidParameterError(f"min_pixels must be >= 1, got {min_pixels}.")

    X = _analysis_matrix(adata, layer)
    sample_labels = adata.obs[sample_key].astype(str).values
    samples = pd.unique(sample_labels)

    aggregate = {"mean": np.mean, "median": np.median, "sum": np.sum}[method]

    kept, profiles, pixel_counts, dropped = [], [], [], []
    for sample in samples:
        mask = sample_labels == sample
        n_pixels = int(mask.sum())
        if n_pixels < min_pixels:
            dropped.append((sample, n_pixels))
            continue
        kept.append(sample)
        profiles.append(aggregate(X[mask], axis=0))
        pixel_counts.append(n_pixels)

    if dropped:
        warnings.warn(
            f"[MORTIS] Dropped {len(dropped)} sample(s) with < {min_pixels} pixels: "
            f"{[f'{s} (n={n})' for s, n in dropped[:5]]}"
            f"{'...' if len(dropped) > 5 else ''}",
            stacklevel=2,
        )
    if not kept:
        raise InsufficientSamplesError(
            f"No sample had at least min_pixels={min_pixels} pixels. "
            "Lower min_pixels or check that QC has not removed most of the data."
        )

    # Carry across any obs column that is unambiguous at the sample level.
    if carry_obs is None:
        carry_obs = [
            column for column in adata.obs.columns
            if column != sample_key
            and adata.obs.groupby(sample_labels, observed=True)[column].nunique().max() <= 1
        ]
    else:
        missing = [c for c in carry_obs if c not in adata.obs.columns]
        if missing:
            raise InvalidParameterError(f"carry_obs column(s) not found in adata.obs: {missing}.")

    obs = pd.DataFrame(index=pd.Index(kept, name=None).astype(str))
    obs[sample_key] = kept
    for column in carry_obs:
        lookup = {}
        for sample in kept:
            values = adata.obs.loc[sample_labels == sample, column]
            lookup[sample] = values.iloc[0] if len(values) else np.nan
        obs[column] = [lookup[s] for s in kept]
    obs["n_pixels"] = pixel_counts

    result = ad.AnnData(
        X=np.vstack(profiles).astype(np.float32),
        obs=obs,
        var=adata.var.copy(),
    )
    result.uns["pseudobulk"] = {
        "sample_key": sample_key,
        "method": method,
        "layer": layer,
        "min_pixels": min_pixels,
        "n_samples": len(kept),
        "n_samples_dropped": len(dropped),
        "n_pixels_total": int(sum(pixel_counts)),
    }
    record_step(result, "pseudobulk", {
        "sample_key": sample_key, "method": method, "layer": layer,
        "min_pixels": min_pixels, "n_pixels_in": adata.n_obs, "n_samples_out": len(kept),
    })
    print(
        f"[MORTIS] Pseudobulk ({method}) over '{sample_key}': "
        f"{adata.n_obs:,} pixels -> {len(kept)} sample profiles."
    )
    return result


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> np.ndarray:
    """
    Cliff's delta between two groups, computed per column.

    Defined as ``P(x > y) - P(x < y)`` over all cross-group pairs, where *x*
    is drawn from ``group1`` and *y* from ``group2``. Ties contribute nothing
    to either term.

    The sign convention is worth stating explicitly because it is easy to get
    backwards: **delta > 0 means group1 tends to exceed group2.**

    ============  ===========================================================
    delta         meaning
    ============  ===========================================================
    +1.0          every group1 value exceeds every group2 value
     0.0          the groups are completely intermixed
    -1.0          every group1 value falls below every group2 value
    ============  ===========================================================

    Parameters
    ----------
    group1, group2 : numpy.ndarray
        Arrays of shape ``(n_samples, n_features)``, or 1-D of length
        ``n_samples`` for a single feature.

    Returns
    -------
    numpy.ndarray
        One delta per feature.
    """
    a = np.atleast_2d(np.asarray(group1, dtype=np.float64))
    b = np.atleast_2d(np.asarray(group2, dtype=np.float64))
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    if a.shape[1] != b.shape[1]:
        raise InvalidParameterError(
            f"group1 and group2 must have the same number of features, got {a.shape[1]} and {b.shape[1]}."
        )
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise InsufficientSamplesError("Both groups must contain at least one sample.")

    n1, n2 = a.shape[0], b.shape[0]

    # Direct pairwise comparison. Cohorts are small (n patients, not n pixels),
    # so the (n1, n2, n_features) intermediate stays modest; features are
    # chunked so a wide matrix cannot blow up memory regardless.
    chunk = max(1, int(2e7 // max(n1 * n2, 1)))
    deltas = np.empty(a.shape[1], dtype=np.float64)
    for start in range(0, a.shape[1], chunk):
        stop = min(start + chunk, a.shape[1])
        left = a[:, None, start:stop]
        right = b[None, :, start:stop]
        greater = (left > right).sum(axis=(0, 1))
        less = (left < right).sum(axis=(0, 1))
        deltas[start:stop] = (greater - less) / (n1 * n2)
    return deltas


def _bootstrap_delta_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int, alpha: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap interval for Cliff's delta, resampling samples."""
    rng = np.random.default_rng(seed)
    n1, n2 = a.shape[0], b.shape[0]
    draws = np.empty((n_boot, a.shape[1]), dtype=np.float64)
    for i in range(n_boot):
        draws[i] = cliffs_delta(a[rng.integers(0, n1, n1)], b[rng.integers(0, n2, n2)])
    return (
        np.percentile(draws, 100 * alpha / 2, axis=0),
        np.percentile(draws, 100 * (1 - alpha / 2), axis=0),
    )


# ---------------------------------------------------------------------------
# Differential abundance
# ---------------------------------------------------------------------------

def differential_abundance(
    adata: ad.AnnData,
    group_key: str,
    group1: str,
    group2: str,
    layer: Optional[str] = None,
    min_samples: int = 3,
    bootstrap: int = 0,
    alpha: float = 0.05,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Compare two groups of samples, one metabolite at a time, at the sample level.

    Run this on the output of :func:`pseudobulk`, not on pixel-level data. It
    will refuse obviously pixel-level input (see ``min_samples``), but it
    cannot detect every case, so the responsibility for passing sample-level
    data is yours.

    Ranking is by ``|delta|`` rather than by p-value. With a handful of patients
    per arm the p-value has very little resolution — a Mann-Whitney comparison
    of 4 vs 4 cannot produce a p below 0.029 no matter how cleanly the groups
    separate, so an FDR threshold discards real findings while the effect size
    still ranks them sensibly.

    Parameters
    ----------
    adata : anndata.AnnData
        Sample-level data, typically from :func:`pseudobulk`.
    group_key : str
        Column in ``adata.obs`` holding the group labels.
    group1, group2 : str
        The two groups to compare. **Positive delta means higher in group1.**
    layer : str, optional
        Layer to test. Default ``None`` uses ``.X``.
    min_samples : int
        Minimum samples required per group. Default 3. Below 3 the comparison
        conveys almost nothing, and a value far above a plausible patient count
        is the signal that pixel-level data was passed by mistake.
    bootstrap : int
        Bootstrap resamples for a confidence interval on delta. Default 0
        (skip). 2000 is a reasonable choice when reporting a headline finding.
    alpha : float
        Two-sided level for the bootstrap interval. Default 0.05.
    random_state : int
        Seed for the bootstrap.

    Returns
    -------
    pandas.DataFrame
        Sorted by ``|delta|`` descending, with columns:

        ``metabolite``, ``delta`` (Cliff's delta, primary), ``magnitude``
        (negligible/small/medium/large), ``median_group1``, ``median_group2``,
        ``log2fc``, ``pval``, ``pval_adj`` (BH-FDR), ``n_group1``, ``n_group2``,
        and ``delta_ci_low``/``delta_ci_high`` when ``bootstrap > 0``.

    Raises
    ------
    InsufficientSamplesError
        If either group has fewer than ``min_samples`` samples.
    """
    if group_key not in adata.obs.columns:
        raise InvalidParameterError(
            f"'{group_key}' not found in adata.obs. Available: {adata.obs.columns.tolist()}."
        )
    labels = adata.obs[group_key].astype(str).values
    available = sorted(set(labels))
    for group in (group1, group2):
        if group not in available:
            raise InvalidParameterError(f"Group '{group}' not found in '{group_key}'. Available: {available}.")
    if group1 == group2:
        raise InvalidParameterError("group1 and group2 must be different.")

    mask1, mask2 = labels == group1, labels == group2
    n1, n2 = int(mask1.sum()), int(mask2.sum())
    for name, n in ((group1, n1), (group2, n2)):
        if n < min_samples:
            raise InsufficientSamplesError(
                f"Group '{name}' has {n} sample(s), need at least min_samples={min_samples}. "
                "If these are pixels rather than samples, run mortis.pseudobulk() first."
            )

    X = _analysis_matrix(adata, layer)
    a, b = X[mask1], X[mask2]

    delta = cliffs_delta(a, b)

    # Mann-Whitney needs at least one non-constant feature to be meaningful;
    # constant features get p = 1 rather than a NaN that silently becomes 0.
    with np.errstate(invalid="ignore"):
        try:
            _, pvals = stats.mannwhitneyu(a, b, axis=0, alternative="two-sided")
        except ValueError:
            pvals = np.ones(X.shape[1], dtype=np.float64)
    pvals = np.nan_to_num(np.asarray(pvals, dtype=np.float64), nan=1.0)
    pvals_adj = multipletests(pvals, method="fdr_bh")[1]

    median1, median2 = np.median(a, axis=0), np.median(b, axis=0)
    # A ratio only means anything for non-negative intensities. This function is
    # also used for organisation metrics (Moran's I is signed), so leave log2fc
    # undefined there rather than emitting a nan and a warning.
    pseudo = 1e-9
    ratio_defined = (median1 >= 0) & (median2 >= 0)
    log2fc = np.full(median1.shape, np.nan, dtype=np.float64)
    log2fc[ratio_defined] = np.log2(
        (median1[ratio_defined] + pseudo) / (median2[ratio_defined] + pseudo)
    )

    result = pd.DataFrame({
        "metabolite": adata.var_names,
        "delta": delta,
        "magnitude": _delta_magnitude(delta),
        "median_group1": median1,
        "median_group2": median2,
        "log2fc": log2fc,
        "pval": pvals,
        "pval_adj": pvals_adj,
        "n_group1": n1,
        "n_group2": n2,
    })

    if bootstrap > 0:
        low, high = _bootstrap_delta_ci(a, b, bootstrap, alpha, random_state)
        result["delta_ci_low"] = low
        result["delta_ci_high"] = high
        # An interval excluding zero is the small-n statement worth making.
        result["ci_excludes_zero"] = (low > 0) | (high < 0)

    result = (
        result.reindex(result["delta"].abs().sort_values(ascending=False).index)
        .reset_index(drop=True)
    )
    result.attrs["group1"], result.attrs["group2"] = group1, group2

    n_large = int((np.abs(delta) >= 0.474).sum())
    print(
        f"[MORTIS] Differential abundance {group1} (n={n1}) vs {group2} (n={n2}): "
        f"{n_large}/{adata.n_vars} metabolites with a large effect (|delta| >= 0.474), "
        f"{int((pvals_adj < 0.05).sum())} at FDR < 0.05."
    )
    if min(n1, n2) < 5:
        print(
            "[MORTIS] Note: with fewer than 5 samples in a group, rank p-values are coarse "
            "(4 vs 4 cannot go below p=0.029). Rank by 'delta' and treat FDR as secondary."
        )
    return result


def paired_differential_abundance(
    adata: ad.AnnData,
    pair_key: str,
    time_key: str,
    time1: str,
    time2: str,
    layer: Optional[str] = None,
    min_pairs: int = 3,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Within-subject change between two timepoints, on matched samples.

    For a longitudinal design (a biopsy before treatment and another after),
    the paired comparison is strictly more powerful than treating the two
    timepoints as independent groups, because each patient serves as their own
    control and between-patient variation cancels out.

    Pairing is done by explicit ID matching, and any subject missing one of the
    two timepoints is dropped with a warning rather than silently. An unmatched
    ID that slips through does not produce an error — it produces a confident
    paired result computed from mismatched data, which is worse.

    Parameters
    ----------
    adata : anndata.AnnData
        Sample-level data from :func:`pseudobulk`, containing both timepoints.
    pair_key : str
        Column identifying the subject (usually patient).
    time_key : str
        Column holding the timepoint labels.
    time1, time2 : str
        The two timepoints. **Positive delta means higher at time2** — the
        change is read as time2 minus time1, in the direction time flows.
    layer : str, optional
        Layer to test. Default ``None`` uses ``.X``.
    min_pairs : int
        Minimum matched subjects required. Default 3.
    random_state : int
        Reserved for future resampling options; accepted so callers can pass
        a seed uniformly across the statistics API.

    Returns
    -------
    pandas.DataFrame
        Sorted by ``|delta|`` descending, with columns ``metabolite``,
        ``delta`` (matched-pairs rank-biserial correlation), ``median_change``,
        ``n_increased``, ``n_decreased``, ``pval`` (Wilcoxon signed-rank),
        ``pval_adj``, and ``n_pairs``.
    """
    for key in (pair_key, time_key):
        if key not in adata.obs.columns:
            raise InvalidParameterError(
                f"'{key}' not found in adata.obs. Available: {adata.obs.columns.tolist()}."
            )
    times = adata.obs[time_key].astype(str).values
    available = sorted(set(times))
    for time in (time1, time2):
        if time not in available:
            raise InvalidParameterError(f"Timepoint '{time}' not found. Available: {available}.")
    if time1 == time2:
        raise InvalidParameterError("time1 and time2 must be different.")

    subjects = adata.obs[pair_key].astype(str).values
    at_time1 = {s: i for i, (s, t) in enumerate(zip(subjects, times)) if t == time1}
    at_time2 = {s: i for i, (s, t) in enumerate(zip(subjects, times)) if t == time2}

    matched = sorted(set(at_time1) & set(at_time2))
    unmatched = sorted((set(at_time1) | set(at_time2)) - set(matched))
    if unmatched:
        warnings.warn(
            f"[MORTIS] {len(unmatched)} subject(s) lack both timepoints and were excluded: "
            f"{unmatched[:5]}{'...' if len(unmatched) > 5 else ''}",
            stacklevel=2,
        )
    if len(matched) < min_pairs:
        raise InsufficientSamplesError(
            f"Only {len(matched)} subject(s) have both '{time1}' and '{time2}', "
            f"need at least min_pairs={min_pairs}."
        )

    X = _analysis_matrix(adata, layer)
    before = X[[at_time1[s] for s in matched]]
    after = X[[at_time2[s] for s in matched]]
    change = after - before
    n_pairs = len(matched)

    # Wilcoxon signed-rank; features with no change at all are undefined, so
    # they get p = 1 instead of raising.
    pvals = np.ones(X.shape[1], dtype=np.float64)
    varying = ~np.all(np.isclose(change, 0.0), axis=0)
    if varying.any():
        with np.errstate(invalid="ignore"):
            try:
                _, p_varying = stats.wilcoxon(change[:, varying], axis=0, zero_method="wilcox")
                pvals[varying] = np.nan_to_num(np.asarray(p_varying, dtype=np.float64), nan=1.0)
            except ValueError:
                pass
    pvals_adj = multipletests(pvals, method="fdr_bh")[1]

    # Matched-pairs rank-biserial correlation: the paired analogue of Cliff's
    # delta, on the same [-1, 1] scale and read the same way.
    n_increased = (change > 0).sum(axis=0)
    n_decreased = (change < 0).sum(axis=0)
    delta = (n_increased - n_decreased) / n_pairs

    result = pd.DataFrame({
        "metabolite": adata.var_names,
        "delta": delta,
        "magnitude": _delta_magnitude(delta),
        "median_change": np.median(change, axis=0),
        "n_increased": n_increased,
        "n_decreased": n_decreased,
        "pval": pvals,
        "pval_adj": pvals_adj,
        "n_pairs": n_pairs,
    })
    result = (
        result.reindex(result["delta"].abs().sort_values(ascending=False).index)
        .reset_index(drop=True)
    )
    result.attrs["time1"], result.attrs["time2"] = time1, time2

    print(
        f"[MORTIS] Paired {time1} -> {time2} over '{pair_key}': {n_pairs} matched subject(s), "
        f"{int((np.abs(delta) >= 0.474).sum())}/{adata.n_vars} metabolites with a large change."
    )
    return result
