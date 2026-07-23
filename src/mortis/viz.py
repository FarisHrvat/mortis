"""
MORTIS Publication Figure Module
================================
House style, true-vector PDF export with optional LaTeX typesetting, and the
figures for the sample-level and spatial-organization analyses.

Why a separate module
---------------------
Figures made during exploration and figures that go into a manuscript have
different requirements, and mixing them produces the familiar late-stage mess:
inconsistent fonts across panels, text that turns into uneditable outlines the
moment a co-author opens the PDF in Illustrator, and no way to tell which
version of the analysis a given figure came from.

Three things this module guarantees:

**Text stays text.** ``pdf.fonttype = 42`` makes matplotlib emit a Type0 font
with a CIDFontType2 descendant, so the PDF carries the TrueType outlines
(``/FontFile2``) and a glyph-to-character map (``/ToUnicode``) instead of
converting text to filled paths. A collaborator can retype a label, restyle an
axis, or fix a typo without regenerating anything, and the PDF is searchable.
Matplotlib's default Type 3 output has neither, and many editors cannot select
that text at all. Both properties are asserted against the written file in
``tests/test_viz.py``, including a round-trip through ``pdftotext``.

**Vector stays vector.** Lines, text and axes are never rasterised. Only the
interior of a dense scatter is, and only above a pixel-count threshold, because
a 500,000-point scatter as vector produces a PDF no reader can open. Axes and
labels around it remain sharp at any zoom.

**Figures are traceable.** :func:`save_figure` writes the analysis parameters
into the PDF's own metadata, along with a short hash of them. Months later,
``pdfinfo figure.pdf`` says exactly which run produced it. Reviewers ask where a
number came from; a figure that answers on its own is worth the few lines.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .exceptions import InvalidParameterError

__all__ = [
    "set_publication_style",
    "reset_style",
    "save_figure",
    "plot_effect_size",
    "plot_delta_volcano",
    "plot_abundance_vs_organization",
    "plot_signature_comparison",
]

#: Colours are chosen to stay distinguishable in greyscale and under the common
#: forms of colour-vision deficiency: they differ in lightness, not only in hue.
PALETTE = {
    "up": "#B2182B",            # higher in group 1
    "down": "#2166AC",          # higher in group 2
    "neutral": "#BABABA",
    "organization": "#762A83",  # the organization-only finding class
    "both": "#1B7837",
    "abundance": "#E08214",
}

_LATEX_PREAMBLE = r"""
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{siunitx}
"""

_ORIGINAL_RCPARAMS: Dict[str, Any] = {}


def _latex_available() -> bool:
    return all(shutil.which(binary) for binary in ("latex", "dvipng"))


def set_publication_style(
    latex: bool = False,
    font_family: str = "sans-serif",
    base_size: float = 8.0,
    linewidth: float = 0.8,
    dpi: int = 300,
) -> None:
    """
    Apply the house style to every subsequent figure.

    Call once at the top of a script. Every figure then shares fonts, sizes and
    line weights, which is what makes a figure set look like one paper rather
    than six.

    Parameters
    ----------
    latex : bool
        Typeset all text with a real LaTeX installation, so figure text matches
        the manuscript exactly and ``$\\alpha$``-style maths renders properly.
        Requires ``latex`` and ``dvipng`` on PATH; if they are missing this
        falls back to matplotlib's built-in mathtext with a warning rather than
        failing, since mathtext handles most maths and needs no installation.

        Real LaTeX is markedly slower per figure, so it is off by default —
        turn it on for the final render.
    font_family : str
        ``"sans-serif"`` (Helvetica/Arial-like, the journal default) or
        ``"serif"`` (Times-like, matching most LaTeX manuscripts).
    base_size : float
        Base font size in points. 8 pt suits a single-column figure at final
        print size; raise it for slides.
    linewidth : float
        Default line and axis width in points.
    dpi : int
        Raster resolution, applied to PNG output and to any rasterised scatter
        interior inside a PDF.
    """
    if base_size <= 0:
        raise InvalidParameterError(f"base_size must be > 0, got {base_size}.")
    if font_family not in ("sans-serif", "serif"):
        raise InvalidParameterError(
            f"font_family must be 'sans-serif' or 'serif', got '{font_family}'."
        )

    if not _ORIGINAL_RCPARAMS:
        _ORIGINAL_RCPARAMS.update(mpl.rcParams.copy())

    use_latex = latex
    if latex and not _latex_available():
        warnings.warn(
            "latex=True but 'latex'/'dvipng' were not found on PATH. Falling back to "
            "matplotlib's mathtext, which renders most maths without a LaTeX install. "
            "Install TeX Live or MacTeX for exact manuscript typesetting.",
            stacklevel=2,
        )
        use_latex = False

    mpl.rcParams.update({
        # --- the two settings that make a PDF editable downstream ---
        # Type 42 (TrueType) keeps text selectable and re-editable. Matplotlib's
        # default Type 3 is not reliably editable in Illustrator or Inkscape.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",   # SVG text stays text, not paths

        "text.usetex": use_latex,
        "font.family": font_family,
        "font.size": base_size,
        "axes.titlesize": base_size + 1,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,
        "figure.titlesize": base_size + 2,

        "axes.linewidth": linewidth,
        "grid.linewidth": linewidth * 0.6,
        "lines.linewidth": linewidth * 1.5,
        "xtick.major.width": linewidth,
        "ytick.major.width": linewidth,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 100,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "figure.autolayout": False,
    })
    if use_latex:
        mpl.rcParams["text.latex.preamble"] = _LATEX_PREAMBLE

    print(
        f"[MORTIS] Publication style set (font={font_family}, {base_size}pt, "
        f"LaTeX={'on' if use_latex else 'off'}, PDF text editable)."
    )


def reset_style() -> None:
    """Restore matplotlib's settings from before the first style call."""
    if _ORIGINAL_RCPARAMS:
        mpl.rcParams.update(_ORIGINAL_RCPARAMS)
    else:
        mpl.rcdefaults()


def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    formats: Sequence[str] = ("pdf",),
    provenance: Optional[Dict[str, Any]] = None,
    close: bool = False,
) -> Dict[str, Path]:
    """
    Write a figure in one or more formats, with provenance in the PDF metadata.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to write.
    path : str or pathlib.Path
        Output path. Any extension is replaced by each requested format, so
        ``"fig1"`` and ``"fig1.pdf"`` behave the same.
    formats : sequence of str
        Any of ``"pdf"`` (vector, editable text — the one to submit),
        ``"svg"`` (vector, for further editing), ``"png"`` (raster, for
        drafts and slides), ``"eps"`` (legacy journals).
    provenance : dict, optional
        Analysis parameters to embed. A short SHA-256 of the JSON is added as
        ``mortis_hash`` so two figures can be compared at a glance. PDF only;
        other formats have nowhere to put it.
    close : bool
        Close the figure afterwards. Useful in loops that would otherwise keep
        every figure in memory.

    Returns
    -------
    dict
        Format name to written path.

    Examples
    --------
    >>> mt.save_figure(fig, "figures/fig2", formats=("pdf", "png"),
    ...                provenance={"cohort": "vedolizumab", "seed": 0})
    """
    valid = {"pdf", "svg", "png", "eps"}
    formats = tuple(formats)
    unknown = [f for f in formats if f not in valid]
    if unknown:
        raise InvalidParameterError(f"Unknown format(s) {unknown}. Available: {sorted(valid)}.")
    if not formats:
        raise InvalidParameterError("At least one format is required.")

    base = Path(path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    metadata = None
    if provenance is not None:
        payload = json.dumps(provenance, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        metadata = {
            "Title": base.name,
            "Creator": "MORTIS",
            "Subject": payload[:800],
            "Keywords": f"mortis_hash={digest}",
            "CreationDate": datetime.now(timezone.utc),
        }

    written: Dict[str, Path] = {}
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        if fmt == "pdf" and metadata is not None:
            fig.savefig(out, format="pdf", metadata=metadata)
        else:
            fig.savefig(out, format=fmt)
        written[fmt] = out

    if close:
        plt.close(fig)

    suffix = f" [hash {metadata['Keywords'].split('=')[1]}]" if metadata else ""
    print(f"[MORTIS] Wrote {', '.join(str(p.name) for p in written.values())}{suffix}")
    return written


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise InvalidParameterError(
            f"'{name}' is missing required column(s) {missing}. "
            f"Available: {frame.columns.tolist()}."
        )


def _wrap(labels: Sequence[str], width: int = 34) -> list:
    """Truncate long compound names so they do not eat the axes."""
    return [lbl if len(lbl) <= width else lbl[: width - 1] + "…" for lbl in labels]


def plot_effect_size(
    result: pd.DataFrame,
    top_n: int = 25,
    group_labels: Tuple[str, str] = ("group 1", "group 2"),
    fdr_threshold: float = 0.05,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Horizontal bars of Cliff's delta for the strongest effects.

    Effect size leads and the p-value is an annotation, which is the right way
    round for a cohort of this size: with a handful of patients per arm a rank
    p-value has almost no resolution, while a monotone effect size stays
    interpretable. Bars that clear ``fdr_threshold`` are drawn solid, the rest
    are outlined — visible, but visibly weaker evidence.

    Parameters
    ----------
    result : pandas.DataFrame
        Output of :func:`mortis.differential_abundance` or
        :func:`mortis.differential_spatial_organization`.
    top_n : int
        How many metabolites to show, taken by ``|delta|``.
    group_labels : tuple of str
        Names for the two groups, used on the axis so the direction is
        unambiguous.
    fdr_threshold : float
        Bars at or below this adjusted p-value are drawn filled.
    figsize : tuple, optional
        Defaults to a height that scales with the number of bars.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes instead of creating a figure.
    """
    _require_columns(result, ("metabolite", "delta", "pval_adj"), "result")
    if top_n < 1:
        raise InvalidParameterError(f"top_n must be >= 1, got {top_n}.")

    top = result.reindex(result["delta"].abs().sort_values(ascending=False).index).head(top_n)
    top = top.iloc[::-1]

    if ax is None:
        height = max(2.0, 0.18 * len(top) + 1.0)
        fig, ax = plt.subplots(figsize=figsize or (4.2, height))
    else:
        fig = ax.figure

    y = np.arange(len(top))
    deltas = top["delta"].to_numpy()
    significant = top["pval_adj"].to_numpy() < fdr_threshold
    colors = np.where(deltas >= 0, PALETTE["up"], PALETTE["down"])

    for yi, delta, color, sig in zip(y, deltas, colors, significant):
        ax.barh(
            yi, delta, height=0.75,
            color=color if sig else "none",
            edgecolor=color, linewidth=0.9,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(_wrap(top["metabolite"].astype(str).tolist()))
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel(
        f"Cliff's $\\delta$   ($\\leftarrow$ higher in {group_labels[1]}"
        f"    |    higher in {group_labels[0]} $\\rightarrow$)"
    )

    n1 = int(result["n_group1"].iloc[0]) if "n_group1" in result.columns else None
    n2 = int(result["n_group2"].iloc[0]) if "n_group2" in result.columns else None
    if n1 is not None and n2 is not None:
        ax.set_title(f"{group_labels[0]} (n={n1}) vs {group_labels[1]} (n={n2})", loc="left")

    handles = [
        mpl.patches.Patch(facecolor=PALETTE["up"], edgecolor=PALETTE["up"],
                          label=f"FDR $<$ {fdr_threshold}"),
        mpl.patches.Patch(facecolor="none", edgecolor=PALETTE["up"],
                          label=f"FDR $\\geq$ {fdr_threshold}"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def plot_delta_volcano(
    result: pd.DataFrame,
    fdr_threshold: float = 0.05,
    delta_threshold: float = 0.474,
    label_top: int = 8,
    figsize: Tuple[float, float] = (4.2, 3.6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Effect size against significance, with Cliff's delta on the x-axis.

    A conventional volcano puts log fold change on x, which at small n is
    dominated by whichever patient happened to be extreme. Delta is bounded in
    [-1, 1], so the plot has fixed limits and the axis means the same thing in
    every figure of the paper.
    """
    _require_columns(result, ("metabolite", "delta", "pval_adj"), "result")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    delta = result["delta"].to_numpy()
    # Adjusted p-values can be exactly 0 after correction; clip so log stays finite.
    padj = np.clip(result["pval_adj"].to_numpy(), 1e-300, None)
    y = -np.log10(padj)

    passes = (padj < fdr_threshold) & (np.abs(delta) >= delta_threshold)
    colors = np.where(
        ~passes, PALETTE["neutral"], np.where(delta >= 0, PALETTE["up"], PALETTE["down"])
    )
    # Rasterise only the point cloud; axes, ticks and labels stay vector.
    ax.scatter(
        delta, y, c=colors, s=12, linewidths=0, alpha=0.85,
        rasterized=len(delta) > 5000,
    )

    ax.axhline(-np.log10(fdr_threshold), color="black", linewidth=0.6, linestyle="--")
    for x in (-delta_threshold, delta_threshold):
        ax.axvline(x, color="black", linewidth=0.6, linestyle="--")

    if label_top > 0:
        ranked = result.assign(_score=np.abs(delta) * y)
        for _, row in ranked.nlargest(label_top, "_score").iterrows():
            ax.annotate(
                str(row["metabolite"])[:24],
                (row["delta"], -np.log10(max(row["pval_adj"], 1e-300))),
                fontsize=mpl.rcParams["font.size"] - 2.5,
                xytext=(3, 3), textcoords="offset points",
            )

    ax.set_xlim(-1.08, 1.08)
    ax.set_xlabel("Cliff's $\\delta$")
    ax.set_ylabel("$-\\log_{10}$ FDR")
    fig.tight_layout()
    return fig


def plot_abundance_vs_organization(
    merged: pd.DataFrame,
    delta_threshold: float = 0.474,
    label_top: int = 8,
    figsize: Tuple[float, float] = (4.4, 4.0),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    The two-axis figure: how much moved, against how it was arranged.

    Each metabolite is placed by its abundance effect (x) and its spatial
    organization effect (y). The interesting region is the top and bottom of
    the vertical band around ``x = 0`` — metabolites present at the same
    abundance in both groups but arranged differently. Those points are
    invisible to abundance testing and to bulk metabolomics, and this figure is
    the argument for having done imaging at all.

    Parameters
    ----------
    merged : pandas.DataFrame
        Output of :func:`mortis.compare_abundance_and_organization`.
    delta_threshold : float
        Where to draw the guide lines. Should match the value passed to the
        classifier or the shading will disagree with the labels.
    """
    _require_columns(
        merged,
        ("metabolite", "delta_abundance", "delta_organization", "classification"),
        "merged",
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    colour_for = {
        "organization only": PALETTE["organization"],
        "both": PALETTE["both"],
        "abundance only": PALETTE["abundance"],
        "neither": PALETTE["neutral"],
    }
    # Draw "neither" first so the findings sit on top of the cloud.
    for label in ("neither", "abundance only", "both", "organization only"):
        subset = merged[merged["classification"] == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["delta_abundance"], subset["delta_organization"],
            c=colour_for[label], s=22 if label != "neither" else 12,
            linewidths=0, alpha=0.9 if label != "neither" else 0.55,
            label=f"{label} (n={len(subset)})", zorder=3 if label != "neither" else 1,
            rasterized=len(subset) > 5000,
        )

    for value in (-delta_threshold, delta_threshold):
        ax.axvline(value, color="black", linewidth=0.5, linestyle=":")
        ax.axhline(value, color="black", linewidth=0.5, linestyle=":")
    ax.axvline(0.0, color="black", linewidth=0.7)
    ax.axhline(0.0, color="black", linewidth=0.7)

    if label_top > 0:
        focus = merged[merged["classification"] == "organization only"]
        for _, row in focus.head(label_top).iterrows():
            ax.annotate(
                str(row["metabolite"])[:22],
                (row["delta_abundance"], row["delta_organization"]),
                fontsize=mpl.rcParams["font.size"] - 2.5,
                xytext=(4, 3), textcoords="offset points",
            )

    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel("abundance effect   (Cliff's $\\delta$)")
    ax.set_ylabel("organization effect   (Cliff's $\\delta$)")
    ax.legend(loc="upper left", frameon=False, fontsize=mpl.rcParams["font.size"] - 2)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_signature_comparison(
    table: pd.DataFrame,
    rho: Optional[float] = None,
    labels: Tuple[str, str] = ("cohort A", "cohort B"),
    figsize: Tuple[float, float] = (4.0, 3.8),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Effect sizes from two cohorts against each other, with the correlation.

    Points on the diagonal are metabolites the two cohorts agree on; points in
    the off-diagonal quadrants move in opposite directions. A cloud with no
    structure is a real result — the two cohorts do not share a signature.

    Parameters
    ----------
    table : pandas.DataFrame
        Second element returned by :func:`mortis.cross_cohort_profile` or
        :func:`mortis.track_flow`.
    rho : float, optional
        Spearman correlation to annotate. Defaults to ``table.attrs['rho']``
        when the table carries it.
    """
    _require_columns(table, ("metabolite", "agreement"), "table")
    delta_columns = [c for c in table.columns if c.startswith("delta_")]
    if len(delta_columns) < 2:
        raise InvalidParameterError(
            f"'table' must carry two delta_* columns, found {delta_columns}."
        )
    x_col, y_col = delta_columns[:2]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    colour_for = {
        "concordant": PALETTE["both"],
        "discordant": PALETTE["up"],
        "weak": PALETTE["neutral"],
    }
    for label in ("weak", "concordant", "discordant"):
        subset = table[table["agreement"] == label]
        if subset.empty:
            continue
        ax.scatter(
            subset[x_col], subset[y_col], c=colour_for[label],
            s=20 if label != "weak" else 11, linewidths=0,
            alpha=0.9 if label != "weak" else 0.5,
            label=f"{label} (n={len(subset)})",
            zorder=3 if label != "weak" else 1,
            rasterized=len(subset) > 5000,
        )

    limit = 1.08
    ax.plot([-limit, limit], [-limit, limit], color="black", linewidth=0.5, linestyle="--", zorder=0)
    ax.axvline(0.0, color="black", linewidth=0.7)
    ax.axhline(0.0, color="black", linewidth=0.7)

    if rho is None:
        rho = table.attrs.get("rho")
    if rho is not None:
        annotation = f"$\\rho = {rho:.3f}$"
        if "rho_ci_low" in table.attrs:
            annotation += (
                f"\n95\\% CI [{table.attrs['rho_ci_low']:.2f}, {table.attrs['rho_ci_high']:.2f}]"
                if mpl.rcParams["text.usetex"] else
                f"\n95% CI [{table.attrs['rho_ci_low']:.2f}, {table.attrs['rho_ci_high']:.2f}]"
            )
        ax.annotate(
            annotation, xy=(0.04, 0.96), xycoords="axes fraction",
            va="top", ha="left", fontsize=mpl.rcParams["font.size"],
        )

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel(f"{labels[0]}   (Cliff's $\\delta$)")
    ax.set_ylabel(f"{labels[1]}   (Cliff's $\\delta$)")
    ax.legend(loc="lower right", frameon=False, fontsize=mpl.rcParams["font.size"] - 2)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig
