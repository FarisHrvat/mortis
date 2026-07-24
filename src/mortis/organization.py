"""
MORTIS Spatial Organization Module
==================================
Testing whether metabolites are *arranged* differently between groups of
patients, independently of whether there is more or less of them.

The question this answers
-------------------------
Abundance testing asks "is there more of metabolite X in responders?" That is
the question bulk metabolomics already answers, and running it on imaging data
throws away the only thing imaging adds. A metabolite can be present at
identical total abundance in two patients and still be organised completely
differently — diffuse and uniform in one, concentrated into sharp foci in the
other. Lactate spread evenly through a section and lactate pooled into hypoxic
cores are not the same biology, and no abundance test can tell them apart.

This module summarises each metabolite's *spatial pattern* in one section into
a small number of scalars, giving one value per patient per metabolite. Those
values then go through exactly the same patient-level machinery as abundance
(:func:`mortis.differential_abundance`), because the statistical problem is
identical once you have one number per patient.

The payoff is :func:`compare_abundance_and_organization`, which runs both axes
and labels every metabolite by which one moved. Metabolites in the
``"organization only"` class are invisible to every abundance-based method and
to bulk metabolomics entirely.

Prior art, stated plainly
-------------------------
Neither ingredient is new and this module does not pretend otherwise. Spatial
autocorrelation statistics for MSI are well established, and the Moran Imaging
toolbox (bioRxiv 2025) recently built spatial features for MSI on the same
foundation. Testing for *differential* spatial pattern across conditions was
introduced for spatial transcriptomics by DESpace (*Bioinformatics* 2024),
which models cluster-by-condition interactions with edgeR.

What is not available anywhere, as far as we can establish: a **patient-level
inferential test for differential spatial organization in spatial
metabolomics**. DESpace is transcriptomics and models counts with a negative
binomial, which does not transfer to MSI intensities; the MSI spatial-statistics
work extracts features without a cohort-level test. This module is that
combination, built on the small-n effect-size machinery in
:mod:`mortis.stats`.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd

from .analysis import _build_spatial_weights, _feature_tiles, _tile_width
from .exceptions import (
    InsufficientSamplesError,
    InvalidParameterError,
    MissingSpatialError,
)
from .reproducibility import record_step
from .stats import differential_abundance

__all__ = [
    "spatial_organization",
    "differential_spatial_organization",
    "compare_abundance_and_organization",
]

#: Metrics computed per sample per metabolite. Every one is deliberately scale
#: free — normalised so that a section with 30,000 pixels and a section with
#: 3,000 produce comparable numbers. Without that, the metric would mostly
#: measure section size and any group difference would track how big the
#: biopsies happened to be.
ORGANIZATION_METRICS = ("morans_i", "entropy", "hotspot_fraction", "gini")


def _sample_moran(block: np.ndarray, W, row_sums: np.ndarray, n: int, S0: float) -> np.ndarray:
    """Moran's I for every column of one tile, within a single section."""
    mean = block.mean(axis=0)
    WX = W @ block
    WX -= row_sums[:, None].astype(np.float32) * mean
    block = block - mean
    numerator = np.einsum("ij,ij->j", block, WX)
    denom = np.einsum("ij,ij->j", block, block)
    safe = np.where(denom < 1e-12, 1.0, denom)
    out = (n * numerator) / (S0 * safe)
    out[denom < 1e-12] = 0.0
    return out


def _normalised_entropy(block: np.ndarray) -> np.ndarray:
    """
    Shannon entropy of each ion image, divided by log(n_pixels).

    Treating a metabolite's pixel intensities as a distribution over locations,
    entropy is high when signal is spread evenly across the section and low
    when it is concentrated in a few pixels. Dividing by ``log(n_pixels)`` puts
    it on [0, 1] regardless of how many pixels the section has — without that
    normalisation a bigger section scores higher for purely combinatorial
    reasons and the metric becomes a proxy for biopsy size.
    """
    positive = np.clip(block, 0.0, None)
    totals = positive.sum(axis=0)
    totals[totals == 0] = 1.0
    p = positive / totals
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    return (-terms.sum(axis=0) / np.log(max(block.shape[0], 2))).astype(np.float64)


def _gini(block: np.ndarray) -> np.ndarray:
    """
    Gini coefficient of each ion image: 0 = perfectly uniform, 1 = all signal
    in one pixel. Complements entropy — both measure concentration, but Gini
    responds mainly to the top of the distribution, so a metabolite forming one
    intense focus against a flat background moves Gini more than entropy.
    """
    positive = np.clip(block, 0.0, None)
    n = positive.shape[0]
    ordered = np.sort(positive, axis=0)
    index = np.arange(1, n + 1, dtype=np.float64)[:, None]
    totals = ordered.sum(axis=0)
    totals[totals == 0] = 1.0
    return ((2.0 * (index * ordered).sum(axis=0)) / (n * totals) - (n + 1.0) / n).astype(np.float64)


def _hotspot_fraction(block: np.ndarray, W_star, n: int, k: int) -> np.ndarray:
    """
    Fraction of pixels sitting in a significant Getis-Ord Gi* hot spot
    (|z| > 1.96, positive side). Reads as "how much of this section is taken up
    by concentrated high-signal foci" — a direct measure of focal organisation
    that is already a proportion, so it compares across section sizes.
    """
    mean = block.mean(axis=0)
    sd = np.sqrt(np.maximum((block ** 2).mean(axis=0) - mean ** 2, 1e-12))
    const = np.sqrt(max((n * (1.0 / k) - 1.0) / (n - 1), 1e-12))
    gi = (W_star @ block - mean) / (sd * const)
    return (gi > 1.96).mean(axis=0).astype(np.float64)


def spatial_organization(
    adata: ad.AnnData,
    sample_key: str,
    metrics: Sequence[str] = ORGANIZATION_METRICS,
    n_neighbors: int = 6,
    layer: Optional[str] = None,
    min_pixels: int = 50,
    carry_obs: Optional[Sequence[str]] = None,
) -> ad.AnnData:
    """
    Summarise each metabolite's spatial pattern, once per sample.

    The spatial analogue of :func:`mortis.pseudobulk`: where pseudobulk
    collapses a section to *how much* of each metabolite is present, this
    collapses it to *how that metabolite is arranged*. The output has the same
    shape — samples x metabolites — so it feeds the same downstream tests.

    Each section is processed independently, with its own spatial weights
    graph. Nothing is ever compared across section boundaries.

    Parameters
    ----------
    adata : anndata.AnnData
        Pixel-level data with ``adata.obsm['spatial']``.
    sample_key : str
        Column in ``adata.obs`` identifying the section. Use the section here,
        not the patient — organisation is a property of a physical tissue
        section, and two sections from one patient should be summarised
        separately (aggregate them afterwards if the patient is your unit).
    metrics : sequence of str
        Any of ``"morans_i"`` (clustered vs diffuse), ``"entropy"`` (evenness
        of spread), ``"hotspot_fraction"`` (share of the section in Gi* hot
        spots), ``"gini"`` (concentration into few pixels).
    n_neighbors : int
        Neighbours in the spatial graph. Default 6.
    layer : str, optional
        Layer to summarise. Default ``None`` uses ``.X``.
    min_pixels : int
        Sections with fewer pixels are skipped — a spatial statistic on a
        handful of pixels is noise. Default 50.
    carry_obs : sequence of str, optional
        Sample-level metadata to carry over. Default auto-detects, as in
        :func:`mortis.pseudobulk`.

    Returns
    -------
    anndata.AnnData
        Samples x metabolites. ``.X`` holds the first requested metric and
        every metric is also in ``.layers``, so
        ``adata.layers['entropy']`` and friends are available. ``.obs`` carries
        sample metadata plus ``n_pixels``.

    Examples
    --------
    >>> org = mt.spatial_organization(adata, sample_key="section")
    >>> res = mt.differential_spatial_organization(
    ...     org, "response", "Responder", "Non Responder", metric="morans_i"
    ... )
    """
    if "spatial" not in adata.obsm:
        raise MissingSpatialError("adata.obsm['spatial'] is missing.")
    if sample_key not in adata.obs.columns:
        raise InvalidParameterError(
            f"'{sample_key}' not found in adata.obs. Available: {adata.obs.columns.tolist()}."
        )
    metrics = tuple(metrics)
    unknown = [m for m in metrics if m not in ORGANIZATION_METRICS]
    if unknown:
        raise InvalidParameterError(
            f"Unknown metric(s) {unknown}. Available: {list(ORGANIZATION_METRICS)}."
        )
    if not metrics:
        raise InvalidParameterError("At least one metric is required.")

    sample_labels = adata.obs[sample_key].astype(str).values
    samples = pd.unique(sample_labels)

    kept, pixel_counts, skipped = [], [], []
    collected = {m: [] for m in metrics}

    for sample in samples:
        mask = sample_labels == sample
        n = int(mask.sum())
        if n < min_pixels:
            skipped.append((sample, n))
            continue

        section = adata[mask]
        if layer is not None:
            if layer not in section.layers:
                raise InvalidParameterError(f"Layer '{layer}' not found.")
            section = ad.AnnData(
                X=section.layers[layer], obs=section.obs.copy(), var=section.var.copy()
            )
            section.obsm["spatial"] = adata.obsm["spatial"][mask]

        # A single-section graph: batch_key is absent from this view, so
        # _build_spatial_weights treats it as one block and no offsetting is
        # applied. Neighbours cannot leak between sections because sections are
        # never in the same call.
        W, _ = _build_spatial_weights(section, n_neighbors, batch_key="__none__")
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        S0 = float(W.sum())

        need_star = "hotspot_fraction" in metrics
        if need_star:
            W_star, _ = _build_spatial_weights(
                section, n_neighbors, batch_key="__none__", include_self=True
            )

        per_metric = {m: np.empty(adata.n_vars, dtype=np.float64) for m in metrics}
        for start, stop, block in _feature_tiles(section, _tile_width(n)):
            if "entropy" in metrics:
                per_metric["entropy"][start:stop] = _normalised_entropy(block)
            if "gini" in metrics:
                per_metric["gini"][start:stop] = _gini(block)
            if need_star:
                per_metric["hotspot_fraction"][start:stop] = _hotspot_fraction(
                    block, W_star, n, n_neighbors + 1
                )
            if "morans_i" in metrics:
                # Runs last: it centres the block in place.
                per_metric["morans_i"][start:stop] = _sample_moran(block, W, row_sums, n, S0)

        kept.append(sample)
        pixel_counts.append(n)
        for m in metrics:
            collected[m].append(per_metric[m])

    if skipped:
        warnings.warn(
            f"[MORTIS] Skipped {len(skipped)} section(s) with < {min_pixels} pixels: "
            f"{[f'{s} (n={n})' for s, n in skipped[:5]]}{'...' if len(skipped) > 5 else ''}",
            stacklevel=2,
        )
    if not kept:
        raise InsufficientSamplesError(
            f"No section had at least min_pixels={min_pixels} pixels."
        )

    if carry_obs is None:
        carry_obs = [
            column for column in adata.obs.columns
            if column != sample_key
            and adata.obs.groupby(sample_labels, observed=True)[column].nunique().max() <= 1
        ]

    obs = pd.DataFrame(index=pd.Index(kept).astype(str))
    obs[sample_key] = kept
    for column in carry_obs:
        obs[column] = [
            adata.obs.loc[sample_labels == s, column].iloc[0] for s in kept
        ]
    obs["n_pixels"] = pixel_counts

    stacked = {m: np.vstack(collected[m]).astype(np.float32) for m in metrics}
    result = ad.AnnData(X=stacked[metrics[0]].copy(), obs=obs, var=adata.var.copy())
    for m in metrics:
        result.layers[m] = stacked[m]
    result.uns["spatial_organization"] = {
        "sample_key": sample_key,
        "metrics": list(metrics),
        "n_neighbors": n_neighbors,
        "layer": layer,
        "min_pixels": min_pixels,
        "n_samples": len(kept),
        "n_samples_skipped": len(skipped),
    }
    record_step(result, "spatial_organization", {
        "sample_key": sample_key, "metrics": list(metrics),
        "n_neighbors": n_neighbors, "layer": layer, "min_pixels": min_pixels,
    })
    print(
        f"[MORTIS] Spatial organization over '{sample_key}': {len(kept)} section(s) x "
        f"{adata.n_vars} metabolites, metrics={list(metrics)}."
    )
    return result


def differential_spatial_organization(
    org: ad.AnnData,
    group_key: str,
    group1: str,
    group2: str,
    metric: str = "morans_i",
    min_samples: int = 3,
    bootstrap: int = 0,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Test whether metabolites are arranged differently between two groups.

    Takes the output of :func:`spatial_organization` and runs the same
    patient-level comparison used for abundance, on the chosen organisation
    metric. Positive ``delta`` means **more organised in group1** for metrics
    where higher is more structured (``morans_i``, ``hotspot_fraction``,
    ``gini``); for ``entropy`` higher means more *diffuse*, so a positive delta
    there means more spread out in group1.

    Parameters
    ----------
    org : anndata.AnnData
        Output of :func:`spatial_organization`.
    group_key, group1, group2 : str
        The comparison, as in :func:`mortis.differential_abundance`.
    metric : str
        Which organisation metric to test. Must be present in ``org.layers``.
    min_samples : int
        Minimum sections per group. Default 3.
    bootstrap : int
        Bootstrap resamples for a confidence interval on delta. Default 0.
    random_state : int
        Seed for the bootstrap.

    Returns
    -------
    pandas.DataFrame
        As :func:`mortis.differential_abundance`, with ``median_group1`` /
        ``median_group2`` holding the metric rather than an intensity.
    """
    if metric not in org.layers:
        raise InvalidParameterError(
            f"Metric '{metric}' not found in org.layers. Available: {list(org.layers.keys())}. "
            "Request it in spatial_organization(metrics=...)."
        )
    result = differential_abundance(
        org, group_key, group1, group2, layer=metric,
        min_samples=min_samples, bootstrap=bootstrap, random_state=random_state,
    )
    result = result.rename(columns={"log2fc": "log2fc_metric"})
    result.attrs["metric"] = metric
    return result


def compare_abundance_and_organization(
    abundance: pd.DataFrame,
    organization: pd.DataFrame,
    delta_threshold: float = 0.474,
    fdr_threshold: Optional[float] = 0.05,
) -> pd.DataFrame:
    """
    Join the two axes and label what actually moved for each metabolite.

    This is the point of the module. Four outcomes are possible, and the
    interesting one is not the obvious one:

    ``abundance only``
        More or less of it, arranged the same way. A bulk experiment would
        have found this.
    ``organization only``
        **The same amount, arranged differently.** Invisible to abundance
        testing and to bulk metabolomics entirely — only imaging can see it,
        and only if you look for it.
    ``both``
        Amount and arrangement both shifted. The strongest evidence, and worth
        checking that the organisation change is not just a consequence of
        having more signal to see.
    ``neither``
        No shift on either axis at this threshold.

    Parameters
    ----------
    abundance : pandas.DataFrame
        Output of :func:`mortis.differential_abundance`.
    organization : pandas.DataFrame
        Output of :func:`differential_spatial_organization`.
    delta_threshold : float
        ``|delta|`` at or above which an axis counts as having moved. Default
        0.474, the conventional boundary for a "large" Cliff's delta.
    fdr_threshold : float or None
        An axis must *also* reach this BH-adjusted p-value to count as having
        moved. Default 0.05. Pass ``None`` to classify on effect size alone.

        Requiring both is deliberate. On a benchmark with 3 planted
        organization-only differences among 30 metabolites
        (``tests/test_organization.py``), recovery by cohort size was:

        ======  ======================  ======================
        n/group effect size only        effect size + FDR<0.05
        ======  ======================  ======================
        4       3 true, 5 false         0 true, 0 false
        5       3 true, 6 false         0 true, 0 false
        6       3 true, 3 false         **3 true, 0 false**
        8       ~3 true, ~2 false       **3 true, 0 false**
        10      3 true, 3 false         **3 true, 0 false**
        ======  ======================  ======================

        Two things follow. Effect size alone always over-calls — it doubles or
        triples the hit list with noise at every cohort size tested. And below
        about six sections per group the FDR filter rejects even *perfect*
        separation: a Mann-Whitney comparison of 5 vs 5 bottoms out at
        p = 0.0079, which cannot survive multiple-testing correction across a
        few dozen metabolites.

        So **six sections per group is the practical floor** for this analysis.
        Below it, an empty result is the honest answer, not a reason to drop
        the filter. If you do relax it, say so explicitly and treat what comes
        back as hypothesis-generating.

    Returns
    -------
    pandas.DataFrame
        One row per metabolite, sorted so that ``organization only`` findings
        come first, with columns ``metabolite``, ``delta_abundance``,
        ``delta_organization``, ``pval_adj_abundance``, ``pval_adj_organization``
        and ``classification``.

    Examples
    --------
    >>> pb  = mt.pseudobulk(adata, sample_key="section")
    >>> org = mt.spatial_organization(adata, sample_key="section")
    >>> ab  = mt.differential_abundance(pb, "response", "R", "NR")
    >>> do  = mt.differential_spatial_organization(org, "response", "R", "NR")
    >>> mt.compare_abundance_and_organization(ab, do).head()
    """
    if not 0.0 <= delta_threshold <= 1.0:
        raise InvalidParameterError(
            f"delta_threshold must be between 0 and 1, got {delta_threshold}."
        )
    for name, frame in (("abundance", abundance), ("organization", organization)):
        for column in ("metabolite", "delta", "pval_adj"):
            if column not in frame.columns:
                raise InvalidParameterError(
                    f"'{name}' is missing required column '{column}'."
                )

    merged = abundance[["metabolite", "delta", "pval_adj"]].merge(
        organization[["metabolite", "delta", "pval_adj"]],
        on="metabolite", suffixes=("_abundance", "_organization"), how="inner",
    )
    if merged.empty:
        raise InvalidParameterError(
            "No metabolites in common between the two results. Were they computed "
            "from the same dataset?"
        )

    moved_abundance = merged["delta_abundance"].abs() >= delta_threshold
    moved_organization = merged["delta_organization"].abs() >= delta_threshold
    if fdr_threshold is not None:
        if not 0.0 < fdr_threshold <= 1.0:
            raise InvalidParameterError(
                f"fdr_threshold must be in (0, 1] or None, got {fdr_threshold}."
            )
        moved_abundance &= merged["pval_adj_abundance"] < fdr_threshold
        moved_organization &= merged["pval_adj_organization"] < fdr_threshold

    classification = np.select(
        [
            moved_abundance & moved_organization,
            ~moved_abundance & moved_organization,
            moved_abundance & ~moved_organization,
        ],
        ["both", "organization only", "abundance only"],
        default="neither",
    )
    merged["classification"] = classification

    order = {"organization only": 0, "both": 1, "abundance only": 2, "neither": 3}
    merged["_rank"] = merged["classification"].map(order)
    merged = (
        merged.sort_values(
            ["_rank", "delta_organization"],
            ascending=[True, False],
            key=lambda s: s.abs() if s.name == "delta_organization" else s,
        )
        .drop(columns="_rank")
        .reset_index(drop=True)
    )

    counts = merged["classification"].value_counts()
    print(
        f"[MORTIS] Abundance vs organization (|delta| >= {delta_threshold}): "
        f"{counts.get('organization only', 0)} organization-only, "
        f"{counts.get('both', 0)} both, "
        f"{counts.get('abundance only', 0)} abundance-only, "
        f"{counts.get('neither', 0)} neither."
    )
    if counts.get("organization only", 0):
        print(
            "[MORTIS] The organization-only metabolites are the ones no abundance test "
            "or bulk metabolomics experiment could have found."
        )
    return merged
