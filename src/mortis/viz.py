"""
MORTIS Publication Figure Module
================================
House style, true-vector PDF export with optional LaTeX typesetting, and the
figures for the sample-level and spatial-organization analyses.

Why a separate module
---------------------
Figures you make while poking at data and figures that go into a manuscript want
different things, and mixing the two produces a mess everyone recognises at
submission time: eight panels in five fonts, text that turns to uneditable
outlines the moment a co-author opens it in Illustrator, and absolutely no way
to tell which version of the analysis produced figure 3b.

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
import os
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from .exceptions import InvalidParameterError, listing, suggest

__all__ = [
    "set_publication_style",
    "reset_style",
    "save_figure",
    "plot_effect_size",
    "plot_delta_volcano",
    "plot_abundance_vs_organization",
    "plot_signature_comparison",
    "plot_ion_images",
    "plot_organization_heatmap",
    "plot_class_enrichment",
    "plot_pathway_dotplot",
    "diverging_cmap",
    "ion_cmap",
]

_VECTOR_FORMATS = ("pdf", "svg", "eps", "ps")
_RASTER_FORMATS = ("png", "jpg", "jpeg", "tiff", "tif", "webp")
_SAVE_FORMATS = frozenset(_VECTOR_FORMATS + _RASTER_FORMATS)

#: matplotlib spells a couple of these differently from the file extension.
_EXTENSION = {"jpeg": "jpg", "tif": "tiff"}

#: Distinguishable in greyscale and under colour-vision deficiency.
PALETTE = {
    "up": "#B2182B",            # higher in group 1   (RdBu-11)
    "down": "#2166AC",          # higher in group 2   (RdBu-11)
    "neutral": "#B8B4AE",
    "organization": "#01665E",  # the organization-only finding class (BrBG-11)
    "both": "#35978F",          # both axes moved                     (BrBG-11)
    "abundance": "#8C510A",     # abundance only                      (BrBG-11)
}

_LATEX_PREAMBLE = r"""
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{siunitx}
"""

_ORIGINAL_RCPARAMS: Dict[str, Any] = {}

#: Set by the last set_publication_style(). Read by the colormaps.
_ACTIVE_THEME = "print"


def _ink(theme: Optional[str] = None) -> str:
    """Foreground colour for rules, axes and reference lines under the active theme.

    Reference lines follow the theme rather than being fixed to black, which
    is right on paper but invisible on a dark ground: a volcano plot's zero
    line would disappear into the background.
    """
    return {"print": "black", "light": "#10161c", "dark": "#d8d5cf"}[theme or _ACTIVE_THEME]


def ion_cmap(theme: Optional[str] = None):
    """
    Sequential colormap for ion images.

    Viridis is the safe scientific default and it reads as software rather than
    as a figure, the purple-to-yellow ramp is instantly recognisable as "a
    plotting library made this". This is a quieter ramp in the same
    perceptually-ordered spirit: near-black through petrol and teal to a warm
    pale, so intensity still maps monotonically to lightness but the result sits
    closer to how imaging is presented in the literature.

    Pass any matplotlib colormap name to ``plot_ion_images(cmap=...)`` if you
    would rather have viridis, magma or a house style back.
    """
    from matplotlib.colors import LinearSegmentedColormap

    dark_ground = (theme or _ACTIVE_THEME) == "dark"
    stops = (["#080c10", "#123044", "#1c5a6b", "#3f9088", "#8bc0a8", "#eae3d2"]
             if dark_ground else
             ["#0d1b26", "#17415a", "#226b7c", "#4a9c92", "#9ccbb2", "#f4efe2"])
    return LinearSegmentedColormap.from_list("mortis_ion", stops)


def diverging_cmap(theme: Optional[str] = None):
    """
    A diverging colormap whose midpoint matches the background it is drawn on.

    Standard diverging maps (RdBu, coolwarm) pass through white at zero. That is
    right on paper and wrong on a dark ground, where every near-zero cell of a
    heatmap lights up as a white block, the values closest to "nothing here"
    end up the most visually prominent thing in the figure.

    This keeps the package's red and blue endpoints and swaps the centre for the
    theme's own background, so zero recedes instead of shouting.
    """
    from matplotlib.colors import LinearSegmentedColormap

    theme = theme or _ACTIVE_THEME
    if theme != "dark":
        return plt.get_cmap("RdBu_r")
    return LinearSegmentedColormap.from_list(
        "mortis_dark_diverging", ["#4a8fc7", "#2c4a63", "#0c1116", "#5e2c2c", "#e2665c"]
    )


def _latex_available() -> bool:
    return all(shutil.which(binary) for binary in ("latex", "dvipng"))


def set_publication_style(
    latex: bool = False,
    font_family: str = "sans-serif",
    base_size: float = 8.0,
    linewidth: float = 0.8,
    dpi: int = 300,
    theme: str = "print",
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

        Real LaTeX is markedly slower per figure, so it is off by default,
        turn it on for the final render.
    font_family : str
        ``"sans-serif"`` (Helvetica/Arial-like, the journal default),
        ``"serif"`` (Times-like, matching most LaTeX manuscripts),
        ``"monospace"``, or the name of a font installed on this machine,
        for example ``"Arial"`` or ``"Helvetica Neue"``. A named font that
        is not installed raises, rather than letting matplotlib fall back to
        DejaVu Sans and hand you figures that differ between machines.
    base_size : float
        Base font size in points. 8 pt suits a single-column figure at final
        print size; raise it for slides.
    linewidth : float
        Default line and axis width in points.
    dpi : int
        Raster resolution, applied to PNG output and to any rasterised scatter
        interior inside a PDF.
    theme : {"print", "light", "dark"}
        ``"print"`` (default) is what a journal wants: black on an opaque white
        page, regardless of what your desktop is set to.

        ``"light"`` and ``"dark"`` render on a **transparent** background with
        ink, axes and ticks recoloured to sit on that ground. Use them for
        slides, posters and web pages, where a figure with a baked-in white
        rectangle looks pasted on rather than placed. The data colours are
        unchanged, so a figure stays recognisable across all three.
    """
    if base_size <= 0:
        raise InvalidParameterError(f"base_size must be > 0, got {base_size}.")
    generic = font_family in ("sans-serif", "serif", "monospace")
    if not generic:
        # matplotlib silently substitutes DejaVu Sans for a missing font
        available = {f.name for f in font_manager.fontManager.ttflist}
        if font_family not in available:
            raise InvalidParameterError(
                f"No font called {font_family!r} is installed, so matplotlib "
                f"would quietly fall back to DejaVu Sans and your figures "
                f"would not match. Installed families close to that name: "
                f"{listing(sorted(n for n in available if font_family.lower()[:4] in n.lower()) or sorted(available), limit=10)}."
                f"{suggest(font_family, available)}"
            )
    if theme not in ("print", "light", "dark"):
        raise InvalidParameterError(
            f"theme must be 'print', 'light' or 'dark', got '{theme}'."
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
        # type 42 stays editable in illustrator; the default type 3 does not
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",   # SVG text stays text, not paths

        "text.usetex": use_latex,
        "font.family": font_family if generic else [font_family],
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
        "figure.autolayout": False,
    })

    global _ACTIVE_THEME
    _ACTIVE_THEME = theme
    ink = {"print": "black", "light": "#10161c", "dark": "#d8d5cf"}[theme]
    mpl.rcParams.update({
        "savefig.transparent": theme != "print",
        "figure.facecolor": "white" if theme == "print" else "none",
        "axes.facecolor": "white" if theme == "print" else "none",
        "savefig.facecolor": "white" if theme == "print" else "none",
        "savefig.edgecolor": "none",
        "text.color": ink,
        "axes.labelcolor": ink,
        "axes.edgecolor": ink,
        "xtick.color": ink,
        "ytick.color": ink,
        "axes.titlecolor": ink,
        "legend.labelcolor": ink,
    })
    if use_latex:
        mpl.rcParams["text.latex.preamble"] = _LATEX_PREAMBLE

    print(
        f"[MORTIS] Publication style set (theme={theme}, font={font_family}, {base_size}pt, "
        f"LaTeX={'on' if use_latex else 'off'}, PDF text editable)."
    )


def reset_style() -> None:
    """Restore matplotlib's settings from before the first style call."""
    if _ORIGINAL_RCPARAMS:
        mpl.rcParams.update(_ORIGINAL_RCPARAMS)
    else:
        mpl.rcdefaults()


def _pdf_timestamp() -> datetime:
    """
    The creation date to stamp into a PDF.

    A wall-clock timestamp is the one thing that stops two identical runs from
    producing identical files, which is annoying when you are trying to prove
    they are identical. Honour SOURCE_DATE_EPOCH the way reproducible-builds
    tooling does, so ``SOURCE_DATE_EPOCH=0 python analysis.py`` gives
    byte-stable PDFs; fall back to now when nobody asked.

    A malformed value is refused rather than ignored. Matplotlib reads the same
    variable on its way into the PDF backend and dies on it with
    ``invalid literal for int()``, so quietly falling back would only delay the
    crash and make it harder to place.
    """
    stamp = os.environ.get("SOURCE_DATE_EPOCH")
    if stamp is None:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromtimestamp(int(stamp), tz=timezone.utc)
    except ValueError:
        raise InvalidParameterError(
            f"SOURCE_DATE_EPOCH is set to {stamp!r}, which is not a whole number "
            f"of seconds since 1970. Set it to an integer (SOURCE_DATE_EPOCH=0 "
            f"works) to get byte-identical PDFs, or unset it to stamp the "
            f"current time."
        ) from None


def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    formats: Sequence[str] = ("pdf",),
    provenance: Optional[Dict[str, Any]] = None,
    close: bool = False,
    dpi: Optional[int] = None,
    transparent: bool = False,
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
        Vector: ``"pdf"`` (editable text, the one to submit), ``"svg"`` (for
        further editing), ``"eps"`` and ``"ps"`` (legacy journals).
        Raster: ``"png"``, ``"jpg"``, ``"tiff"`` and ``"webp"``. TIFF is
        written with lossless LZW compression, which is what journals asking
        for TIFF expect; JPEG is written at quality 95 and has no
        transparency.
    dpi : int, optional
        Resolution for the raster formats. Defaults to whatever the active
        style set, which is 300 after
        :func:`set_publication_style`. Ignored by the vector formats, which
        have no resolution.
    transparent : bool
        Draw the figure background transparent. Ignored for JPEG, which has
        no alpha channel and would come out on black.
    provenance : dict, optional
        Analysis parameters to embed. A short SHA-256 of the JSON goes in as
        ``mortis_hash``, so two figures made from the same parameters carry the
        same tag. It identifies the run, not the pixels. PDF only; the other
        formats have nowhere to put it.

        The embedded timestamp is taken from ``SOURCE_DATE_EPOCH`` when that
        environment variable is set, which is what makes byte-identical PDFs
        possible across runs. Otherwise it is the current time.
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
    formats = tuple(str(f).lower().lstrip(".") for f in formats)
    unknown = [f for f in formats if f not in _SAVE_FORMATS]
    if unknown:
        raise InvalidParameterError(
            f"Unknown figure format(s) {unknown}. MORTIS writes "
            f"{listing(sorted(_SAVE_FORMATS))}."
            f"{suggest(unknown[0], _SAVE_FORMATS)}"
        )
    if not formats:
        raise InvalidParameterError(
            "At least one format is required, for example formats=('pdf', 'png')."
        )
    if dpi is not None and dpi <= 0:
        raise InvalidParameterError(
            f"dpi has to be positive, got {dpi}. 300 is the usual submission "
            "figure, 600 for line art, 1200 where a journal asks for it."
        )
    if dpi is None and any(f in _RASTER_FORMATS for f in formats):
        # savefig.dpi defaults to the string "figure", so it cannot be cast
        configured = mpl.rcParams.get("savefig.dpi")
        dpi = int(fig.dpi) if configured in (None, "figure") else int(configured)

    # with_suffix("") would eat the ".dark" in "two_axis.dark"
    base = Path(path)
    if base.suffix.lower().lstrip(".") in _SAVE_FORMATS:
        base = base.with_suffix("")
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
            "CreationDate": _pdf_timestamp(),
        }

    written: Dict[str, Path] = {}
    for fmt in formats:
        # append, since with_suffix() would replace the ".dark"
        out = base.parent / f"{base.name}.{_EXTENSION.get(fmt, fmt)}"
        options: Dict[str, Any] = {"format": fmt, "transparent": transparent}
        if fmt in _RASTER_FORMATS:
            options["dpi"] = dpi
        if fmt in {"jpg", "jpeg"}:
            # jpeg has no alpha; a transparent figure comes out black
            options["transparent"] = False
            options["pil_kwargs"] = {"quality": 95}
        if fmt == "tiff" or fmt == "tif":
            # LZW is lossless and is what the journals that ask for TIFF want.
            options["pil_kwargs"] = {"compression": "tiff_lzw"}
        if fmt == "pdf" and metadata is not None:
            options["metadata"] = metadata
        fig.savefig(out, **options)
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
    """
    Break long compound names over two lines rather than cutting them off.

    Systematic names run past forty characters -- "2-Aminobicyclo[3.1.0]hexane-
    2,6-dicarboxylic acid" is 48 -- and an ellipsis in the middle of one makes
    it unidentifiable, which is not a saving. Split at a separator near the
    halfway point so the whole name survives; ``tight_layout`` then finds the
    room for it.
    """
    out = []
    for label in labels:
        text = str(label)
        if len(text) <= width:
            out.append(text)
            continue
        middle = len(text) // 2
        breaks = [i for i, ch in enumerate(text) if ch in "-_/(, " and 4 < i < len(text) - 4]
        if breaks:
            cut = min(breaks, key=lambda i: abs(i - middle))
            out.append(text[: cut + 1].rstrip() + "\n" + text[cut + 1 :].lstrip())
        else:
            out.append(text[:middle] + "\n" + text[middle:])
    return out


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
    are outlined, visible, but visibly weaker evidence.

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
    ax.set_yticklabels(_wrap(top["metabolite"].astype(str).tolist(), 30))
    ax.axvline(0.0, color=_ink(), linewidth=0.8)
    ax.set_xlim(-1.05, 1.05)
    # under the axis ends; one long label runs off at small widths
    ax.set_xlabel("Cliff's $\\delta$")
    ax.annotate(f"$\\leftarrow$ {group_labels[1]}", xy=(0.0, -0.115),
                xycoords="axes fraction", ha="left", va="top",
                fontsize=mpl.rcParams["font.size"] - 2, color=PALETTE["down"])
    ax.annotate(f"{group_labels[0]} $\\rightarrow$", xy=(1.0, -0.115),
                xycoords="axes fraction", ha="right", va="top",
                fontsize=mpl.rcParams["font.size"] - 2, color=PALETTE["up"])

    n1 = int(result["n_group1"].iloc[0]) if "n_group1" in result.columns else None
    n2 = int(result["n_group2"].iloc[0]) if "n_group2" in result.columns else None
    if n1 is not None and n2 is not None:
        ax.set_title(f"{group_labels[0]} (n={n1}) vs {group_labels[1]} (n={n2})", loc="left")

    # only when both states occur, and below the axes
    if bool(significant.any()) and bool((~significant).any()):
        handles = [
            mpl.patches.Patch(facecolor=PALETTE["up"], edgecolor=PALETTE["up"],
                              label=f"FDR $<$ {fdr_threshold}"),
            mpl.patches.Patch(facecolor="none", edgecolor=PALETTE["up"],
                              label=f"FDR $\\geq$ {fdr_threshold}"),
        ]
        ax.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
            ncol=2, frameon=False, fontsize=mpl.rcParams["font.size"] - 1,
        )
    fig.tight_layout()
    return fig


def plot_effect_forest(
    result: pd.DataFrame,
    top_n: int = 20,
    group_labels: Tuple[str, str] = ("group 1", "group 2"),
    delta_threshold: float = 0.474,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Ranked effect sizes with their bootstrap confidence intervals.

    The figure to use when the cohort is too small for a volcano to say
    anything. With five sections per arm a two-sided rank test cannot return a
    p below 0.0079, so after correction across a panel every point sits on the
    same line and the significance axis carries no information at all. The
    interval does: it shows how much of the effect is supported and how much is
    the sample size talking.

    An interval that excludes zero is the statement worth making at small n,
    and it is drawn solid. Everything else is hollow.

    Parameters
    ----------
    result : pandas.DataFrame
        Output of :func:`mortis.differential_abundance` run with
        ``bootstrap > 0``, so that ``delta_ci_low`` and ``delta_ci_high`` are
        present.
    top_n : int
        How many compounds to show, taken in order of ``|delta|``.
    group_labels : tuple of str
        ``(group1, group2)``, used for the direction labels under the axis.
    delta_threshold : float
        Where to mark the "large effect" boundary. Drawn as a tick on the axis
        rather than a line across the data.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> res = mt.differential_abundance(pb, "state", "Fibrotic", "Healthy",
    ...                                 bootstrap=2000)
    >>> mt.plot_effect_forest(res, group_labels=("Fibrotic", "Healthy"))
    """
    _require_columns(result, ("metabolite", "delta"), "result")
    missing = [c for c in ("delta_ci_low", "delta_ci_high") if c not in result.columns]
    if missing:
        raise InvalidParameterError(
            f"This figure draws confidence intervals and the result table has no "
            f"{missing} column(s). Re-run differential_abundance with "
            f"bootstrap=2000 (or any positive number) to get them."
        )
    if top_n < 1:
        raise InvalidParameterError(f"top_n must be at least 1, got {top_n}.")

    ordered = result.reindex(
        result["delta"].abs().sort_values(ascending=False).index
    ).head(top_n)
    # Largest effect at the top, which is where a reader starts.
    ordered = ordered.iloc[::-1]

    height = max(2.2, 0.24 * len(ordered) + 1.1)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (COLUMN_WIDE, height))
    else:
        fig = ax.figure

    y = np.arange(len(ordered))
    delta = ordered["delta"].to_numpy()
    low = ordered["delta_ci_low"].to_numpy()
    high = ordered["delta_ci_high"].to_numpy()
    excludes_zero = (low > 0) | (high < 0)
    colour = np.where(delta >= 0, PALETTE["up"], PALETTE["down"])

    for yi, d, lo, hi, solid, c in zip(y, delta, low, high, excludes_zero, colour):
        ax.plot([lo, hi], [yi, yi], color=c, linewidth=1.1, solid_capstyle="butt",
                alpha=0.9 if solid else 0.45)
        ax.plot([d], [yi], marker="o", markersize=4.2,
                markerfacecolor=c if solid else "none",
                markeredgecolor=c, markeredgewidth=1.0)

    ax.axvline(0.0, color=_ink(), linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(_wrap(ordered["metabolite"].astype(str).tolist(), 30))
    ax.set_ylim(-0.7, len(ordered) - 0.3)
    ax.set_xlim(-1.12, 1.12)
    ax.set_xlabel("Cliff's $\\delta$   (95% bootstrap CI)")

    # Threshold as a tick on the frame, not a line across the data.
    for value in (-delta_threshold, delta_threshold):
        ax.plot([value, value], [-0.7, -0.55], color=_ink(), linewidth=0.9,
                alpha=0.6, clip_on=False)

    ax.annotate(f"$\\leftarrow$ {group_labels[1]}", xy=(0.0, -0.13),
                xycoords="axes fraction", ha="left", va="top",
                fontsize=mpl.rcParams["font.size"] - 1, color=PALETTE["down"])
    ax.annotate(f"{group_labels[0]} $\\rightarrow$", xy=(1.0, -0.13),
                xycoords="axes fraction", ha="right", va="top",
                fontsize=mpl.rcParams["font.size"] - 1, color=PALETTE["up"])

    n1 = int(result["n_group1"].iloc[0]) if "n_group1" in result.columns else None
    n2 = int(result["n_group2"].iloc[0]) if "n_group2" in result.columns else None
    if n1 is not None and n2 is not None:
        ax.set_title(f"{group_labels[0]} (n={n1}) vs {group_labels[1]} (n={n2})",
                     loc="left")
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

    ax.axhline(-np.log10(fdr_threshold), color=_ink(), linewidth=0.6, alpha=0.22)
    for x in (-delta_threshold, delta_threshold):
        ax.axvline(x, color=_ink(), linewidth=0.6, alpha=0.22)

    if label_top > 0:
        ranked = result.assign(_score=np.abs(delta) * y).nlargest(label_top, "_score")
        _annotate_spread(
            ax,
            [(r["delta"], -np.log10(max(r["pval_adj"], 1e-300))) for _, r in ranked.iterrows()],
            [str(r["metabolite"])[:24] for _, r in ranked.iterrows()],
        )

    ax.set_xlim(-1.08, 1.08)
    ax.set_xlabel("Cliff's $\\delta$")
    ax.set_ylabel("$-\\log_{10}$ FDR")
    fig.tight_layout()
    return fig


#: Nature column widths in inches. Draw at print width so type is not rescaled.
COLUMN_SINGLE, COLUMN_WIDE, COLUMN_DOUBLE = 3.50, 4.72, 7.20

#: Axis labels for the organisation metrics.
_METRIC_LABELS = {
    "morans_i": "Moran's $I$",
    "entropy": "normalised entropy",
    "hotspot_fraction": "hotspot fraction",
    "gini": "Gini coefficient",
}


def _scatter_stacked(ax, frame, x_col, y_col, *, colour, base_size, label,
                     alpha, zorder):
    """
    Draw a scatter that admits when points sit on top of each other.

    Cliff's delta on a small cohort takes very few distinct values -- three
    sections per arm gives ten -- so a whole panel can land on a handful of
    coordinates and the figure shows a dozen dots while the legend claims two
    hundred. Collapse exact duplicates and scale the marker area by the count.
    Nudging the points apart would be easier and would be a lie about where
    they are.

    Returns the largest number of points sharing one coordinate, so the caller
    can say so on the figure.
    """
    counts = frame.groupby([x_col, y_col], sort=True).size().reset_index(name="n")
    ax.scatter(
        counts[x_col], counts[y_col], c=colour,
        s=base_size * np.sqrt(counts["n"]), linewidths=0, alpha=alpha,
        label=f"{label} (n={len(frame)})", zorder=zorder,
        rasterized=len(counts) > 5000,
    )
    return int(counts["n"].max())


def _note_stacking(ax, most: int, corner: str = "right") -> None:
    """
    Small print along the bottom when more than one point shares a marker.

    ``corner`` picks the side, because the legend is not always in the same
    place and the note landing under it helps nobody.
    """
    if most <= 1:
        return
    # below the axes; on a small cohort points sit against the frame
    x = 1.0 if corner == "right" else 0.0
    ax.annotate(
        f"marker area \u221d metabolites at that point (up to {most})",
        xy=(x, 0.0), xycoords="axes fraction", xytext=(0, -34),
        textcoords="offset points", ha=corner, va="top",
        fontsize=mpl.rcParams["font.size"] - 2, color=_ink(), alpha=0.65,
    )


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
    the vertical band around ``x = 0``, metabolites present at the same
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
    # delta is quantised, so points overlap. scale marker area by the count.
    stacked = 0
    # Draw "neither" first so the findings sit on top of the cloud.
    for label in ("neither", "abundance only", "both", "organization only"):
        subset = merged[merged["classification"] == label]
        if subset.empty:
            continue
        stacked = max(stacked, _scatter_stacked(
            ax, subset, "delta_abundance", "delta_organization",
            colour=colour_for[label],
            base_size=22 if label != "neither" else 12,
            label=label,
            alpha=0.9 if label != "neither" else 0.55,
            zorder=3 if label != "neither" else 1,
        ))
    _note_stacking(ax, stacked)

    # zero lines only; threshold guides would read as a grid
    ax.axvline(0.0, color=_ink(), linewidth=0.7, alpha=0.45)
    ax.axhline(0.0, color=_ink(), linewidth=0.7, alpha=0.45)
    for value in (-delta_threshold, delta_threshold):
        ax.plot([value, value], [-1.08, -1.04], color=_ink(), linewidth=0.8,
                alpha=0.55, clip_on=False, zorder=4)
        ax.plot([-1.08, -1.04], [value, value], color=_ink(), linewidth=0.8,
                alpha=0.55, clip_on=False, zorder=4)

    if label_top > 0:
    # one label per coordinate: "Spermidine +3 more"
        organization_only = merged[merged["classification"] == "organization only"]
        points, labels = [], []
        for (x, y), group in organization_only.groupby(
            ["delta_abundance", "delta_organization"], sort=False
        ):
            name = str(group["metabolite"].iloc[0])
            name = name if len(name) <= 16 else name[:15] + "\u2026"
            if len(group) > 1:
                name = f"{name} +{len(group) - 1} more"
            points.append((x, y))
            labels.append(name)
            if len(points) == label_top:
                break
        if points:
            _annotate_spread(
                ax, points, labels,
                avoid=merged[["delta_abundance", "delta_organization"]].to_numpy(),
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
    structure is a real result, the two cohorts do not share a signature.

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
    stacked = 0
    for label in ("weak", "concordant", "discordant"):
        subset = table[table["agreement"] == label]
        if subset.empty:
            continue
        stacked = max(stacked, _scatter_stacked(
            ax, subset, x_col, y_col,
            colour=colour_for[label],
            base_size=20 if label != "weak" else 11,
            label=label,
            alpha=0.9 if label != "weak" else 0.5,
            zorder=3 if label != "weak" else 1,
        ))

    # Legend sits lower right on this one, so the note goes to the left.
    _note_stacking(ax, stacked, corner="left")

    limit = 1.08
    ax.plot([-limit, limit], [-limit, limit], color=_ink(), linewidth=0.6, alpha=0.2, zorder=0)
    ax.axvline(0.0, color=_ink(), linewidth=0.7, alpha=0.45)
    ax.axhline(0.0, color=_ink(), linewidth=0.7, alpha=0.45)

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


def _annotate_spread(ax, points, labels, fontsize=None, avoid=None):
    """
    Annotate points, keeping the text off both other labels and the data.

    Effect sizes are bounded and quantised -- Cliff's delta from five samples
    per group takes 51 values, from two it takes 5 -- so the compounds worth
    naming routinely land on or beside each other. A fixed offset stacks the
    labels; nudging them apart in one direction only moves the collision onto
    the scatter.

    Each label is tried in eight positions around its point, nearest first, and
    the first that hits neither a placed label nor a marker in ``avoid`` wins.
    If every candidate collides the label is placed anyway at the last one,
    because a slightly crowded label beats a missing name.
    """
    fontsize = fontsize if fontsize is not None else mpl.rcParams["font.size"] - 2.5
    line_height = fontsize * 1.45

    # Display space, so "too close" means what it looks like on the page.
    obstacles: List[Tuple[float, float]] = []
    if avoid is not None and len(avoid):
        obstacles = [tuple(xy) for xy in ax.transData.transform(np.asarray(avoid, float))]

    # (dx, dy) in points, ordered outward: right, left, above, below, diagonals.
    candidates = [
        (6, 3), (-6, 3), (6, -line_height), (-6, -line_height),
        (6, line_height), (-6, line_height),
        (6, -2 * line_height), (-6, -2 * line_height),
    ]
    placed: List[Tuple[float, float]] = []
    for (x, y), label in zip(points, labels):
        px, py = ax.transData.transform((float(x), float(y)))
        width = 0.62 * fontsize * len(str(label))
        chosen = candidates[-1]
        for dx, dy in candidates:
            cx, cy = px + dx, py + dy
            clash = any(
                abs(cx - qx) < width and abs(cy - qy) < line_height for qx, qy in placed
            ) or any(
                qx - cx < width and cx - qx < 6 and abs(cy - qy) < line_height * 0.8
                for qx, qy in obstacles
            )
            if not clash:
                chosen = (dx, dy)
                break
        placed.append((px + chosen[0], py + chosen[1]))
        ax.annotate(
            str(label), (x, y),
            xytext=chosen, textcoords="offset points", fontsize=fontsize,
            ha="left" if chosen[0] > 0 else "right",
            va="bottom" if chosen[1] >= 0 else "top",
        )


def _draw_ion_panel(ax, xy: np.ndarray, values: np.ndarray, vmin, vmax, cmap):
    """
    Render one ion image into ``ax``, as a raster where possible.

    Drawing pixels as square scatter markers is the obvious approach and it
    looks wrong: marker size is set in points while pixel spacing is in data
    units, so unless the two happen to coincide the squares either overlap or
    leave the background showing through as a grid of white seams. On a figure
    meant for publication those seams are the first thing a reader notices.

    Rasterising onto the acquisition grid and handing it to ``imshow`` removes
    the problem by construction: cells tile exactly, unmeasured positions stay
    transparent, and the file is smaller. MSI coordinates are integer stage
    positions so this nearly always applies. When they are not on a grid, a
    subsetted or warped object, the function falls back to a scatter, which is
    imperfect but honest about the data it was given.
    """
    x, y = xy[:, 0], xy[:, 1]
    x_int, y_int = np.rint(x), np.rint(y)
    on_grid = np.allclose(x, x_int, atol=1e-6) and np.allclose(y, y_int, atol=1e-6)

    if on_grid:
        cols = (x_int - x_int.min()).astype(np.int64)
        rows = (y_int - y_int.min()).astype(np.int64)
        height, width = int(rows.max()) + 1, int(cols.max()) + 1
        # a wide span with few pixels would allocate a huge empty grid
        if height * width <= max(64 * len(x), 10_000):
            grid = np.full((height, width), np.nan, dtype=float)
            grid[rows, cols] = values
            palette = (mpl.colormaps[cmap] if isinstance(cmap, str) else cmap).copy()
            palette.set_bad(alpha=0.0)          # unmeasured positions stay clear
            return ax.imshow(
                np.ma.masked_invalid(grid), cmap=palette, vmin=vmin, vmax=vmax,
                interpolation="nearest", origin="upper",
            )

    ax.invert_yaxis()
    return ax.scatter(
        x, y, c=np.clip(values, vmin, vmax), cmap=cmap, vmin=vmin, vmax=vmax,
        s=max(0.5, 900 / max(np.sqrt(len(x)), 1) ** 1.5),
        marker="s", linewidths=0, rasterized=len(x) > 5000,
    )


def _panel_layout(samples: np.ndarray, groups: Optional[np.ndarray],
                  n_cols: Optional[int]):
    """
    Where each section goes in a grid of panels, grouped by arm.

    Each group starts on a fresh row so its name can sit once on the left of
    its block, rather than on a second line under every panel title. Repeating
    the arm ten times is noise; saying it twice is a figure.

    Returns ``(placement, n_rows, n_cols)`` where placement holds
    ``(row, col, sample, group, is_first_of_group)``.
    """
    if groups is not None:
        order = sorted(pd.unique(samples),
                       key=lambda s: (str(groups[samples == s][0]), str(s)))
        group_of = {s: str(groups[samples == s][0]) for s in order}
    else:
        order = sorted(pd.unique(samples))
        group_of = {s: "" for s in order}

    if n_cols is None and groups is not None:
        n_cols = int(pd.Series(list(group_of.values())).value_counts().max())
    n_cols = int(n_cols or min(len(order), 5))

    placement, row = [], 0
    for group in dict.fromkeys(group_of[s] for s in order):
        members = [s for s in order if group_of[s] == group]
        for i, sample in enumerate(members):
            placement.append((row + i // n_cols, i % n_cols, sample, group, i == 0))
        row += int(np.ceil(len(members) / n_cols))
    return placement, max(row, 1), n_cols


def plot_embedding(
    adata,
    color: str,
    basis: str = "X_umap",
    palette: Optional[Sequence[str]] = None,
    point_size: float = 2.0,
    figsize: Optional[Tuple[float, float]] = None,
    legend_title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    A UMAP or PCA scatter in the same style, and the same colours, as the rest
    of the figures.

    scanpy's own embedding plot is fine on its own and wrong in a set: it puts
    a grey panel behind the points and picks its palette independently, so the
    same cluster comes out blue in one figure and orange in the next. Here the
    categories are coloured by :func:`plot_spatial_panels`'s palette, so a
    domain keeps its colour wherever it appears.

    Parameters
    ----------
    color : str
        A categorical column in ``adata.obs``.
    basis : str
        Key in ``adata.obsm``. Defaults to the UMAP embedding.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if basis not in adata.obsm:
        available = sorted(k for k in adata.obsm.keys())
        raise InvalidParameterError(
            f"There is no {basis!r} embedding on this object. Run "
            f"mortis.run_umap(adata) to make one. Present: {listing(available)}."
        )
    if color not in adata.obs.columns:
        raise InvalidParameterError(
            f"There is no column called {color!r} in adata.obs to colour by. "
            f"Columns present: {listing(adata.obs.columns)}."
            f"{suggest(color, adata.obs.columns)}"
        )

    xy = np.asarray(adata.obsm[basis], dtype=float)[:, :2]
    labels = adata.obs[color].astype(str).values
    categories = list(pd.Categorical(labels).categories)
    colours = list(palette) if palette else _categorical_palette(len(categories))
    colour_of = dict(zip(categories, colours))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (COLUMN_WIDE, 3.0))
    else:
        fig = ax.figure

    # Densest category first, so the sparse ones stay visible on top of it.
    for category in sorted(categories, key=lambda c: -(labels == c).sum()):
        mask = labels == category
        ax.scatter(
            xy[mask, 0], xy[mask, 1], s=point_size, linewidths=0,
            color=colour_of[category], label=category, alpha=0.85,
            rasterized=int(mask.sum()) > 20000,
        )

    ax.set_xlabel(f"{basis.lstrip('X_').upper()} 1")
    ax.set_ylabel(f"{basis.lstrip('X_').upper()} 2")
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="none", markersize=4,
                         markerfacecolor=colour_of[c], markeredgecolor="none",
                         label=c)
        for c in categories
    ]
    ax.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
        frameon=False, fontsize=mpl.rcParams["font.size"] - 1,
        title=legend_title or color, title_fontsize=mpl.rcParams["font.size"] - 1,
    )
    fig.tight_layout()
    return fig


def plot_spatial_panels(
    adata,
    color: str,
    sample_key: str = "section",
    group_key: Optional[str] = None,
    n_cols: Optional[int] = None,
    palette: Optional[Sequence[str]] = None,
    panel_size: float = 1.35,
    point_size: Optional[float] = None,
    legend_title: Optional[str] = None,
    legend: str = "auto",
) -> plt.Figure:
    """
    One panel per section, coloured by a categorical column in ``.obs``.

    The figure to use for clusters, spatial domains or any other label, in
    place of drawing every section on one pair of axes. Sections do not share a
    coordinate frame -- each was acquired separately and their x and y start
    wherever the stage happened to be -- so overlaying them produces a cloud
    that looks like tissue and is not.

    Parameters
    ----------
    color : str
        A categorical column in ``adata.obs``: cluster labels, domains, or any
        annotation. Rename the categories before plotting if you want the
        legend to say what they mean.
    sample_key : str
        The column identifying the section, one panel each.
    group_key : str, optional
        Arm or condition. Panels are grouped by it and each block is labelled
        once on the left.
    point_size : float, optional
        Marker size. Default scales with how many pixels a panel holds, so a
        3,000-pixel biopsy and a 60,000-pixel resection both come out solid
        rather than one being speckled and the other a blob.
    legend : {"auto", "right", "below"}
        Where the key goes. A legend on the right costs about an inch and a
        half of width, and a figure wider than the page it is printed on gets
        scaled down until its labels are unreadable. ``"auto"`` puts it below
        once the panels alone are wider than a text column.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> adata = mt.rename_clusters(adata, {"0": "Fibrotic core", "1": "Septa"})
    >>> mt.plot_spatial_panels(adata, "cluster", group_key="state")
    """
    if color not in adata.obs.columns:
        raise InvalidParameterError(
            f"There is no column called {color!r} in adata.obs to colour by. "
            f"Columns present: {listing(adata.obs.columns)}."
            f"{suggest(color, adata.obs.columns)}"
        )
    if sample_key not in adata.obs.columns:
        raise InvalidParameterError(
            f"There is no column called {sample_key!r} in adata.obs, so the "
            f"pixels cannot be split into one panel per section. Columns "
            f"present: {listing(adata.obs.columns)}."
            f"{suggest(sample_key, adata.obs.columns)}"
        )
    if "spatial" not in adata.obsm:
        raise InvalidParameterError(
            "Panels are drawn from adata.obsm['spatial'], which is not set on "
            "this object. mortis.read_file() fills it in from the 'x' and 'y' "
            "columns."
        )

    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    samples = adata.obs[sample_key].astype(str).values
    labels = adata.obs[color].astype(str).values
    groups = adata.obs[group_key].astype(str).values if group_key else None

    categories = list(pd.Categorical(labels).categories)
    colours = list(palette) if palette else _categorical_palette(len(categories))
    colour_of = dict(zip(categories, colours))

    placement, n_rows, n_cols = _panel_layout(samples, groups, n_cols)
    if legend == "auto":
        # a text column is ~6.2in; past that the figure is scaled down
        legend = "below" if panel_size * n_cols + 1.6 > 6.2 else "right"
    if legend not in ("right", "below"):
        raise InvalidParameterError(
            f"legend must be 'auto', 'right' or 'below', got {legend!r}."
        )

    legend_rows = int(np.ceil(len(categories) / 3)) if legend == "below" else 0
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_size * n_cols + (1.6 if legend == "right" else 0.5),
                 panel_size * n_rows + 0.4 + 0.22 * legend_rows),
        squeeze=False,
    )
    for ax in axes.ravel():
        ax.set_visible(False)

    for r, c, sample, group, first_of_group in placement:
        ax = axes[r][c]
        ax.set_visible(True)
        mask = samples == sample
        xy = coords[mask]
        size = point_size if point_size is not None else _panel_point_size(xy)
        ax.scatter(
            xy[:, 0], xy[:, 1],
            c=[colour_of[label] for label in labels[mask]],
            s=size, linewidths=0, marker="s", rasterized=len(xy) > 20000,
        )
        ax.set_title(str(sample), fontsize=mpl.rcParams["font.size"] - 1, pad=3)
        ax.set_aspect("equal", anchor="N")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_visible(False)
        if first_of_group and group:
            ax.set_ylabel(group, fontsize=mpl.rcParams["font.size"], labelpad=6)

    handles = [
        mpl.lines.Line2D([], [], marker="s", linestyle="none", markersize=4,
                         markerfacecolor=colour_of[c], markeredgecolor="none",
                         label=c)
        for c in categories
    ]
    if legend == "right":
        fig.legend(
            handles=handles, loc="center left", bbox_to_anchor=(0.995, 0.5),
            frameon=False, fontsize=mpl.rcParams["font.size"] - 1,
            title=legend_title or color,
            title_fontsize=mpl.rcParams["font.size"] - 1,
        )
        fig.tight_layout(rect=(0, 0, 0.99, 1))
    else:
        fig.legend(
            handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
            ncol=min(3, len(categories)), frameon=False,
            fontsize=mpl.rcParams["font.size"] - 1,
            title=legend_title or color,
            title_fontsize=mpl.rcParams["font.size"] - 1,
        )
        fig.tight_layout(rect=(0, 0.045 + 0.03 * legend_rows, 1, 1))
    for ax in axes.ravel():
        if ax.get_visible():
            ax.set_anchor("N")
    return fig


def _panel_point_size(xy: np.ndarray) -> float:
    """
    Marker size that fills the tissue without smearing it.

    A fixed size makes a 3,000-pixel biopsy look like scattered dust and a
    60,000-pixel resection look like a solid blob. Scaling with the pixel
    pitch relative to the section's own extent keeps both readable.
    """
    if len(xy) < 2:
        return 4.0
    span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1.0)
    # Roughly one marker per grid step, in points squared, for a ~1.35in panel.
    return float(np.clip((97.0 / span) ** 2 * 1.6, 0.35, 12.0))


def _categorical_palette(n: int) -> List[str]:
    """
    Distinct colours for cluster labels.

    tab20 in its native order alternates light and dark within a hue, so
    adjacent cluster numbers get near-identical colours. Taking the dark
    entries first keeps the first ten clusters -- the ones that hold most of
    the tissue -- clearly apart.
    """
    tab20 = mpl.colormaps["tab20"].colors
    ordered = list(tab20[0::2]) + list(tab20[1::2])
    if n <= len(ordered):
        return [mpl.colors.to_hex(c) for c in ordered[:n]]
    extra = mpl.colormaps["tab20b"].colors
    return [mpl.colors.to_hex(c) for c in (ordered + list(extra))[:n]]


def plot_ion_images(
    adata,
    metabolite: str,
    sample_key: str,
    group_key: Optional[str] = None,
    n_cols: Optional[int] = None,
    percentile: Tuple[float, float] = (1.0, 99.0),
    shared_scale: bool = True,
    cmap: Optional[str] = None,
    panel_size: float = 1.5,
) -> plt.Figure:
    """
    One metabolite's ion image in every section, laid out for comparison.

    This is the figure that makes a spatial-organization result believable. A
    table saying "Cliff's delta = 1.0 on Moran's I" asks a reader to trust the
    statistic; a row of responder sections showing tight foci above a row of
    non-responder sections showing diffuse haze does not.

    Parameters
    ----------
    adata : anndata.AnnData
        Pixel-level data with ``adata.obsm['spatial']``.
    metabolite : str
        Which metabolite to render.
    sample_key : str
        Column in ``adata.obs`` identifying sections; one panel per section.
    group_key : str, optional
        Column to group panels by. Sections are sorted by this and each panel
        is labelled with it, so the two conditions read as blocks rather than
        being interleaved.
    n_cols : int, optional
        Panels per row. Defaults to the size of the largest group, so each
        group occupies its own row when they are of equal size.
    percentile : tuple of float
        Intensity percentiles for colour limits. Clipping at (1, 99) stops a
        handful of hot pixels flattening every real structure into one colour,
        which is the single most common way ion images get rendered useless.
    shared_scale : bool
        Use one colour scale across all panels. **Keep this on for anything
        comparative**, per-panel scaling makes a faint diffuse section look
        exactly as intense as a bright focal one, which is precisely the
        difference the figure exists to show.
    cmap : str, optional
        Default ``None`` uses :func:`ion_cmap`, a quieter perceptually-ordered
        ramp than viridis. Any matplotlib colormap name works; avoid ``jet``,
        which is not perceptually uniform and invents structure that is not
        in the data.
    panel_size : float
        Side length of each panel in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if "spatial" not in adata.obsm:
        raise InvalidParameterError(
            "Ion images are drawn from adata.obsm['spatial'], which is not set on "
            "this object. mortis.read_file() fills it in from the 'x' and 'y' "
            "columns."
        )
    if metabolite not in adata.var_names:
        raise InvalidParameterError(
            f"{metabolite!r} is not one of the {adata.n_vars} metabolites in this "
            f"object.{suggest(metabolite, adata.var_names)}"
        )
    if sample_key not in adata.obs.columns:
        raise InvalidParameterError(
            f"There is no column called {sample_key!r} in adata.obs, so the "
            f"pixels cannot be split into one panel per section. Columns "
            f"present: {listing(adata.obs.columns)}."
            f"{suggest(sample_key, adata.obs.columns)}"
        )
    if group_key is not None and group_key not in adata.obs.columns:
        raise InvalidParameterError(
            f"group_key={group_key!r} is not a column in adata.obs. Pass None to "
            f"drop the group labels, or use one of: "
            f"{listing(adata.obs.columns)}.{suggest(group_key, adata.obs.columns)}"
        )

    from scipy.sparse import issparse

    column = adata[:, metabolite].X
    values = np.asarray(column.todense() if issparse(column) else column).ravel().astype(float)
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    samples = adata.obs[sample_key].astype(str).values

    if group_key is not None:
        groups = adata.obs[group_key].astype(str).values
        order = sorted(
            pd.unique(samples), key=lambda s: (str(groups[samples == s][0]), str(s))
        )
        group_of = {s: str(groups[samples == s][0]) for s in order}
    else:
        order = sorted(pd.unique(samples))
        group_of = {s: "" for s in order}

    if shared_scale:
        finite = values[np.isfinite(values)]
        vmin, vmax = np.percentile(finite, percentile) if finite.size else (0.0, 1.0)
        if vmin == vmax:
            vmax = vmin + 1e-9

    if n_cols is None and group_key is not None:
        counts = pd.Series(list(group_of.values())).value_counts()
        n_cols = int(counts.max())
    n_cols = int(n_cols or min(len(order), 5))

    # fresh row per group, so its name sits once to the left
    placement, row = [], 0
    for group in dict.fromkeys(group_of[s] for s in order):
        members = [s for s in order if group_of[s] == group]
        for i, sample in enumerate(members):
            placement.append((row + i // n_cols, i % n_cols, sample, group, i == 0))
        row += int(np.ceil(len(members) / n_cols))
    n_rows = max(row, 1)

    # matplotlib parks the exponent over the top panel; put it in the label
    decade = 0
    if shared_scale and np.isfinite(vmax) and vmax > 0:
        decade = int(np.floor(np.log10(vmax)))
        decade = decade if abs(decade) >= 2 else 0
    scale = 10.0 ** -decade

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_size * n_cols, panel_size * n_rows + 0.5),
        squeeze=False,
    )
    for ax in axes.ravel():
        ax.set_visible(False)

    handle = None
    for r, c, sample, group, first_of_group in placement:
        ax = axes[r][c]
        ax.set_visible(True)
        mask = samples == sample
        xy, v = coords[mask], values[mask] * scale
        if not shared_scale:
            finite = v[np.isfinite(v)]
            lo, hi = np.percentile(finite, percentile) if finite.size else (0.0, 1.0)
            if lo == hi:
                hi = lo + 1e-9
        else:
            lo, hi = vmin * scale, vmax * scale
        handle = _draw_ion_panel(ax, xy, v, lo, hi, cmap or ion_cmap())
        ax.set_title(str(sample), fontsize=mpl.rcParams["font.size"] - 1, pad=3)
        # equal aspect centres the box, so titles sit at different heights
        ax.set_aspect("equal", anchor="N")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if first_of_group and group:
            ax.set_ylabel(group, fontsize=mpl.rcParams["font.size"], labelpad=6)

    fig.suptitle(metabolite, fontsize=mpl.rcParams["font.size"] + 1, y=0.995)
    fig.subplots_adjust(top=1 - 0.38 / (panel_size * n_rows + 0.5))
    if shared_scale and handle is not None:
        bar = fig.colorbar(
            handle, ax=[ax for ax in axes.ravel() if ax.get_visible()],
            fraction=0.022, pad=0.02, aspect=26,
        )
        units = "intensity (a.u.)" if not decade else f"intensity ($\\times10^{{{decade}}}$ a.u.)"
        bar.set_label(units, fontsize=mpl.rcParams["font.size"] - 1)
        bar.outline.set_visible(False)
    else:
        fig.tight_layout()

    # a colourbar re-runs the layout and resets the anchor
    for ax in axes.ravel():
        if ax.get_visible():
            ax.set_anchor("N")
    return fig


def plot_organization_heatmap(
    org,
    metric: str = "morans_i",
    group_key: Optional[str] = None,
    top_n: int = 30,
    result: Optional[pd.DataFrame] = None,
    cmap: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Sections against metabolites, coloured by an organization metric.

    Rows are grouped by condition with a separating line, so a metric that
    genuinely differs between groups shows up as two visually distinct bands.
    If it does not, that is worth seeing too, before the effect sizes get
    written up.

    Parameters
    ----------
    org : anndata.AnnData
        Output of :func:`mortis.spatial_organization`.
    metric : str
        Which layer to display.
    group_key : str, optional
        Column in ``org.obs`` used to order and split the rows.
    top_n : int
        How many metabolites to show.
    result : pandas.DataFrame, optional
        Output of :func:`mortis.differential_spatial_organization`. When given,
        metabolites are chosen by ``|delta|`` from it rather than by variance,
        so the heatmap shows the compounds actually being claimed.
    cmap : str, optional
        Default ``None`` picks a diverging map centred on the current theme's
        background (see :func:`diverging_cmap`), so near-zero cells recede
        rather than glowing white on a dark ground.
    """
    if metric not in org.layers:
        available = sorted(k for k in org.layers.keys() if k is not None)
        raise InvalidParameterError(
            f"Metric '{metric}' not in org.layers. Available: {available}."
        )
    if group_key is not None and group_key not in org.obs.columns:
        raise InvalidParameterError(
            f"group_key={group_key!r} is not a column in org.obs. Sections carry "
            f"whatever you passed to spatial_organization(carry_obs=...); this "
            f"object has: {listing(org.obs.columns)}."
            f"{suggest(group_key, org.obs.columns)}"
        )

    matrix = np.asarray(org.layers[metric], dtype=float)
    names = org.var_names.astype(str).to_numpy()

    if result is not None:
        if not {"metabolite", "delta"} <= set(result.columns):
            raise InvalidParameterError("'result' needs 'metabolite' and 'delta' columns.")
        ranked = result.reindex(result["delta"].abs().sort_values(ascending=False).index)
        wanted = [m for m in ranked["metabolite"].astype(str) if m in set(names)][:top_n]
        columns = [int(np.where(names == m)[0][0]) for m in wanted]
    else:
        columns = list(np.argsort(-np.nanvar(matrix, axis=0))[:top_n])

    matrix = matrix[:, columns]
    labels = names[columns]

    if group_key is not None:
        groups = org.obs[group_key].astype(str).to_numpy()
        row_order = np.argsort(groups, kind="stable")
    else:
        groups, row_order = None, np.arange(org.n_obs)
    matrix = matrix[row_order]

    fig, ax = plt.subplots(
        figsize=figsize or (max(4.0, 0.22 * len(labels) + 2.0), max(2.4, 0.2 * org.n_obs + 1.4))
    )
    limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    image = ax.imshow(
        matrix, aspect="auto", cmap=cmap or diverging_cmap(),
        vmin=-limit, vmax=limit, interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(_wrap(list(labels), 18), rotation=90)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(org.obs_names.astype(str).to_numpy()[row_order])

    if groups is not None:
        ordered = groups[row_order]
        boundaries = np.where(ordered[1:] != ordered[:-1])[0]
        for b in boundaries:
            ax.axhline(b + 0.5, color=_ink(), linewidth=1.2)
        # offset in points; neither data nor axes coords know the text width
        longest = max(len(s) for s in org.obs_names.astype(str))
        # ytick.labelsize may be 'medium' when no style was applied
        tick_points = float(mpl.rcParams["font.size"]) - 1.0
        pad_points = -(10.0 + 0.62 * tick_points * longest)

        # horizontal; a rotated "Non Responder" overruns a two-section band
        start = 0
        for end in list(boundaries) + [len(ordered) - 1]:
            ax.annotate(
                ordered[start],
                xy=(0.0, (start + end) / 2), xycoords=ax.get_yaxis_transform(),
                xytext=(pad_points, 0), textcoords="offset points",
                va="center", ha="right",
                fontsize=mpl.rcParams["font.size"] - 1, fontweight="bold",
            )
            start = end + 1

    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    bar.set_label(_METRIC_LABELS.get(metric, metric),
                  fontsize=mpl.rcParams["font.size"] - 1)
    bar.outline.set_visible(False)
    fig.tight_layout()
    return fig


def plot_class_enrichment(
    report: pd.DataFrame,
    fdr_threshold: float = 0.05,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Class-level effects, with the number of compounds behind each one.

    Chemical classes are where a small cohort has something to say: thirty
    phospholipids drifting together is evidence no individual compound at
    n = 6 could carry. The compound count is printed on every bar because a
    class of four and a class of forty deserve very different amounts of
    confidence, and a bar chart alone hides that completely.

    Parameters
    ----------
    report : pandas.DataFrame
        Output of :func:`mortis.class_enrichment`.
    fdr_threshold : float
        Classes at or below this adjusted p-value are drawn filled.
    """
    required = {"chemical_class", "median_delta", "n_compounds", "pval_adj"}
    missing = required - set(report.columns)
    if missing:
        raise InvalidParameterError(f"'report' is missing required column(s) {sorted(missing)}.")
    if report.empty:
        raise InvalidParameterError("'report' is empty - nothing to plot.")

    ordered = report.reindex(report["median_delta"].sort_values().index)
    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize or (4.2, max(2.0, 0.28 * len(ordered) + 1.0))
        )
    else:
        fig = ax.figure

    y = np.arange(len(ordered))
    deltas = ordered["median_delta"].to_numpy()
    significant = ordered["pval_adj"].to_numpy() < fdr_threshold
    colors = np.where(deltas >= 0, PALETTE["up"], PALETTE["down"])

    for yi, delta, color, sig in zip(y, deltas, colors, significant):
        ax.barh(
            yi, delta, height=0.7,
            color=color if sig else "none", edgecolor=color, linewidth=0.9,
        )

    # in the tick label; a zero-length bar would print over the name
    ax.set_yticks(y)
    ax.set_yticklabels([
        f"{name}  ({int(size)})"
        for name, size in zip(ordered["chemical_class"].astype(str),
                              ordered["n_compounds"])
    ])
    ax.axvline(0.0, color=_ink(), linewidth=0.8)
    ax.set_xlim(-1.15, 1.15)
    ax.set_xlabel("median Cliff's $\\delta$ within class  (compounds in brackets)")
    fig.tight_layout()
    return fig


def plot_pathway_dotplot(
    report: pd.DataFrame,
    top_n: int = 15,
    direction: Optional[str] = None,
    fdr_threshold: float = 0.05,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Enriched pathways as a dot plot, one panel per direction.

    Each of the three channels carries a different quantity, which is the
    thing dot plots most often get wrong. Putting ``-log10 FDR`` on both the
    x-axis and the colour looks informative while telling you one fact twice.

    ==========  ==========================================================
    x           enrichment (odds ratio): how much more of this pathway is
                in the shifted set than chance would give
    dot size    compounds hit: a pathway called on 12 measured members is
                worth more than the same p-value from 3
    colour      significance, as -log10 FDR
    ==========  ==========================================================

    Up and down are drawn separately because a pathway with half its members
    rising and half falling is not coherently dysregulated, and pooling them
    would show it as one confident dot.

    Parameters
    ----------
    report : pandas.DataFrame
        Output of :func:`mortis.pathway_ora`.
    top_n : int
        Pathways per panel, taken by adjusted p-value.
    direction : str, optional
        Restrict to one of ``"any"``, ``"up"``, ``"down"``. Default shows the
        directional panels when the report has them.
    fdr_threshold : float
        Reference level marked on the colour bar.
    figsize : tuple, optional
        Defaults to a height that grows with the number of pathways shown, so
        four pathways do not get stretched across a full-page panel.
    """
    required = {"pathway", "direction", "n_hits", "pval_adj"}
    missing = required - set(report.columns)
    if missing:
        raise InvalidParameterError(f"'report' is missing required column(s) {sorted(missing)}.")
    if report.empty:
        raise InvalidParameterError("'report' is empty - no pathways were enriched.")

    if direction is not None:
        panels = [direction]
    else:
        directional = [d for d in ("up", "down") if d in set(report["direction"])]
        panels = directional or ["any"]

    subsets = {
        panel: report[report["direction"] == panel].nsmallest(top_n, "pval_adj").iloc[::-1]
        for panel in panels
    }
    n_rows = max((len(sub) for sub in subsets.values()), default=1)

    # odds ratios go infinite when every member shifted, so cap
    has_odds = "odds_ratio" in report.columns
    if has_odds:
        finite = report.loc[np.isfinite(report["odds_ratio"]), "odds_ratio"]
        cap = float(finite.max()) * 1.15 if len(finite) else 1.0

    significance = -np.log10(np.clip(report["pval_adj"].to_numpy(dtype=float), 1e-300, None))
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(float(significance.max()), 2.0))

    counts = report["n_hits"].to_numpy(dtype=float)
    size_lo, size_hi = float(counts.min()), float(counts.max())

    def _size(values):
        if size_hi <= size_lo:
            return np.full(len(values), 90.0)
        return 30.0 + 220.0 * (np.asarray(values, dtype=float) - size_lo) / (size_hi - size_lo)

    fig, axes = plt.subplots(
        1, len(panels),
        figsize=figsize or (3.9 * len(panels) + 1.4, max(1.8, 0.34 * n_rows + 1.3)),
        squeeze=False, sharex=True,
        # long pathway names; the default gap lets dots slide under them
        gridspec_kw={"wspace": 0.62},
    )
    scatter = None
    for ax, panel in zip(axes.ravel(), panels):
        subset = subsets[panel]
        if subset.empty:
            ax.set_visible(False)
            continue
        y = np.arange(len(subset))
        padj = np.clip(subset["pval_adj"].to_numpy(dtype=float), 1e-300, None)
        colour = -np.log10(padj)
        if has_odds:
            x = np.clip(subset["odds_ratio"].to_numpy(dtype=float), None, cap)
        else:
            x = colour

        scatter = ax.scatter(
            x, y, s=_size(subset["n_hits"]), c=colour, cmap="viridis",
            norm=norm, linewidths=0.4, edgecolors=mpl.rcParams["figure.facecolor"] if _ACTIVE_THEME == "print" else "none",
        )
        ax.set_yticks(y)
        ax.set_yticklabels(_wrap(subset["pathway"].astype(str).tolist(), 34))
        ax.set_ylim(-0.7, len(subset) - 0.3)
        ax.set_xlabel("enrichment (odds ratio)" if has_odds else "$-\\log_{10}$ FDR")
        ax.set_title(panel, loc="left", fontweight="bold")
        if has_odds:
            ax.axvline(1.0, color=_ink(), linewidth=0.6, linestyle="--")

    if scatter is not None:
        bar = fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
        bar.set_label("$-\\log_{10}$ FDR", fontsize=mpl.rcParams["font.size"] - 1)
        bar.ax.axhline(-np.log10(fdr_threshold), color=_ink(), linewidth=1.0)
        bar.outline.set_visible(False)

        # Size legend below the panels, where it cannot sit on top of a dot.
        ticks = sorted({int(size_lo), int(round((size_lo + size_hi) / 2)), int(size_hi)})
        handles = [
            plt.scatter([], [], s=_size([t])[0], c=PALETTE["neutral"], linewidths=0, label=str(t))
            for t in ticks
        ]
        axes.ravel()[0].legend(
            handles=handles, title="compounds hit", loc="upper center",
            bbox_to_anchor=(0.5, -0.28), ncol=len(ticks), frameon=False,
            fontsize=mpl.rcParams["font.size"] - 2,
            title_fontsize=mpl.rcParams["font.size"] - 2, handletextpad=1.1,
            columnspacing=2.4, borderpad=0.8,
        )
    return fig

