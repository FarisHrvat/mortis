"""
MORTIS Plotting Module
======================
Publication-ready, fully customisable visualisations for spatial metabolomics.

Every plot function shares a consistent set of style parameters so you can
produce figures that match your journal requirements without post-processing.
Set `show=False` to return the Figure object silently, allowing you to modify
ANY aspect of the plot (titles, legends, colors) before saving or displaying.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import anndata as ad
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import issparse

_WHITE_RED = LinearSegmentedColormap.from_list("white_red", ["#ffffff", "#8b0000"])
_WHITE_BLUE = LinearSegmentedColormap.from_list("white_blue", ["#ffffff", "#003580"])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_dense(X) -> np.ndarray:
    if issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)

def _save_or_show(fig: plt.Figure, save: Union[bool, str, None], show: bool = True, dpi: int = 300) -> plt.Figure:
    """Save figure to disk, display it, or return it silently for user modification."""
    if save is True:
        out = "mortis_plot.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=dpi)
        print(f"[MORTIS] Saved: {out}")
        if not show:
            plt.close(fig)
    elif isinstance(save, str):
        fig.savefig(save, bbox_inches="tight", dpi=dpi)
        print(f"[MORTIS] Saved: {save}")
        if not show:
            plt.close(fig)

    if show and save is None:
        plt.show()
    return fig

def _get_spot_size(adata: ad.AnnData, user_s: Optional[float] = None) -> float:
    """Calculate the mathematically optimal dot size, or obey user explicitly."""
    if user_s is not None:
        return float(user_s)
    return max(0.1, 50000.0 / max(1, adata.n_obs))

def _pop_spot_size_kwarg(kwargs: dict) -> Optional[float]:
    """
    Pop the marker-size kwarg from a **kwargs dict, accepting either the
    matplotlib-native `s=` or the more descriptive `spot_size=` alias (both
    are used across the codebase and in user-facing docs) so passing either
    name never silently falls through to `ax.scatter(**kwargs)`, where an
    unrecognised `spot_size` key would raise an AttributeError.
    """
    spot_size = kwargs.pop("spot_size", None)
    s = kwargs.pop("s", None)
    return s if s is not None else spot_size

def _get_metabolite_values(adata: ad.AnnData, name: str) -> np.ndarray:
    """Return per-pixel values for an obs column or metabolite name."""
    if name in adata.obs.columns:
        return adata.obs[name].to_numpy()
    if name in adata.var_names:
        idx = adata.var_names.get_loc(name)
        layer = "log1p" if "log1p" in adata.layers else None
        X = _to_dense(adata.layers[layer] if layer else adata.X)
        return X[:, idx]
    raise KeyError(
        f"'{name}' not found in adata.obs columns or adata.var_names. "
        f"Available obs columns: {adata.obs.columns.tolist()[:10]}..."
    )

def _apply_style(
    ax: plt.Axes,
    fontsize: int,
    show_grid: bool,
    show_axes_border: bool,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """Apply consistent style to an axes object."""
    if title:
        ax.set_title(title, fontsize=fontsize + 2, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    if show_grid:
        ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.6)
    else:
        ax.grid(False)
    if not show_axes_border:
        for spine in ax.spines.values():
            spine.set_visible(False)

def _auto_figsize(n_cols: int, n_rows: int, base: float = 4.5) -> Tuple[float, float]:
    """Compute a sensible figure size for a grid of subplots."""
    return (base * n_cols, base * n_rows)


# ---------------------------------------------------------------------------
# QC plot
# ---------------------------------------------------------------------------

def plot_qc(
    stats: dict,
    clean_adata: ad.AnnData,
    sample_name: str = "Sample",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    fontsize: int = 11,
    show_grid: bool = False,
    show_axes_border: bool = True,
    save: Union[bool, str, None] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Four-panel Quality Control report after background filtering."""
    fc = stats["fold_change"]
    cutoff = stats["cutoff"]
    mt = stats["mean_tissue"]
    mb = stats["mean_bg"]
    mask = stats["keep_mask"]

    fs = figsize or (14, 10)
    fig, axs = plt.subplots(2, 2, figsize=fs, dpi=dpi)
    fig.suptitle(
        f"Quality Control Report: {sample_name}",
        fontsize=fontsize + 4, fontweight="bold", y=1.01,
    )

    ax1 = axs[0, 0]
    ax2 = ax1.twinx()
    bins = np.linspace(0, max(3.0, cutoff * 1.5), 40)
    ax1.hist(fc, bins=bins, color="lightsteelblue", edgecolor="white", alpha=0.85)
    ax1.set_ylabel("Metabolite count", color="steelblue", fontsize=fontsize)
    ax1.tick_params(axis="y", labelcolor="steelblue", labelsize=fontsize - 1)
    thresholds = np.linspace(0, bins[-1], 120)
    retained = [int((fc >= t).sum()) for t in thresholds]
    ax2.plot(thresholds, retained, color="navy", linewidth=2)
    ax2.set_ylabel("Retained metabolites", color="navy", fontsize=fontsize)
    ax2.tick_params(axis="y", labelcolor="navy", labelsize=fontsize - 1)
    ax1.axvline(cutoff, color="crimson", linestyle="--", linewidth=2, label=f"Cutoff = {cutoff}×")
    ax2.scatter([cutoff], [mask.sum()], color="crimson", zorder=5, s=60)
    _apply_style(ax1, fontsize, show_grid, show_axes_border, title="Fold-Change Distribution & Retention Curve", xlabel="Tissue / Background fold-change")
    ax1.legend(loc="upper right", fontsize=fontsize - 1)

    ax = axs[0, 1]
    ax.scatter(np.log1p(mb[~mask]), np.log1p(mt[~mask]), color="crimson", s=8, alpha=0.4, label=f"Removed ({(~mask).sum()})")
    ax.scatter(np.log1p(mb[mask]), np.log1p(mt[mask]), color="seagreen", s=8, alpha=0.4, label=f"Kept ({mask.sum()})")
    lim = max(np.log1p(mb).max(), np.log1p(mt).max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, linewidth=1)
    _apply_style(ax, fontsize, show_grid, show_axes_border, title="Signal vs Noise (log1p intensities)", xlabel="Mean background intensity", ylabel="Mean tissue intensity")
    ax.legend(markerscale=2, fontsize=fontsize - 1)

    ax = axs[1, 0]
    ax.pie(
        [mask.sum(), (~mask).sum()],
        labels=[f"Kept\n({mask.sum()})", f"Removed\n({(~mask).sum()})"],
        autopct="%1.1f%%",
        colors=["#4CAF50", "#F44336"],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": fontsize},
    )
    ax.set_title("Metabolite Filtering Summary", fontsize=fontsize + 2, pad=8)

    ax = axs[1, 1]
    coords = clean_adata.obsm["spatial"]
    tic = _to_dense(clean_adata.X).sum(axis=1)
    spot_size = _get_spot_size(clean_adata, _pop_spot_size_kwarg(kwargs))
    sc_obj = ax.scatter(coords[:, 0], coords[:, 1], c=tic, cmap=_WHITE_RED, s=spot_size, rasterized=True, **kwargs)
    plt.colorbar(sc_obj, ax=ax, label="Total Ion Current", shrink=0.85)
    _apply_style(ax, fontsize, show_grid, show_axes_border, title="Clean Tissue Map (Post-Filtering)", xlabel="x", ylabel="y")
    ax.set_aspect("equal", "datalim")
    ax.set_facecolor("#e8e8e8")

    plt.tight_layout()
    return _save_or_show(fig, save, show=show, dpi=dpi)


# ---------------------------------------------------------------------------
# Spatial plots
# ---------------------------------------------------------------------------

def plot_spatial(
    adata: ad.AnnData,
    color: str,
    cmap: str = "viridis",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    fontsize: int = 11,
    title: Optional[str] = None,
    show_grid: bool = False,
    show_axes_border: bool = True,
    palette: str = "tab20",
    save: Union[bool, str, None] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Spatial scatter plot coloured by any observation column or metabolite."""
    spot_size = _get_spot_size(adata, _pop_spot_size_kwarg(kwargs))
    scatter_args = {"alpha": 1.0, "edgecolors": "none", "rasterized": True}
    scatter_args.update(kwargs)

    coords = adata.obsm["spatial"]
    values = _get_metabolite_values(adata, color)
    is_cat = (
        pd.api.types.is_object_dtype(values)
        or isinstance(getattr(values, "dtype", type(values)), pd.CategoricalDtype)
    )

    if figsize is None:
        aspect = (coords[:, 0].max() - coords[:, 0].min()) / max(1, (coords[:, 1].max() - coords[:, 1].min()))
        figsize = (min(10, 6 * aspect), 6)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if is_cat:
        categories = np.unique(values.astype(str))
        colors = sns.color_palette(palette, len(categories))
        for i, cat in enumerate(categories):
            mask = values.astype(str) == cat
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[colors[i]], s=spot_size, label=str(cat), **scatter_args)
        ax.legend(markerscale=max(1, 10/spot_size), bbox_to_anchor=(1.01, 1), loc="upper left",
                  fontsize=fontsize - 2, frameon=False)
    else:
        sc_obj = ax.scatter(coords[:, 0], coords[:, 1], c=values.astype(float),
                        cmap=cmap, s=spot_size, **scatter_args)
        plt.colorbar(sc_obj, ax=ax, shrink=0.8)

    _apply_style(ax, fontsize, show_grid, show_axes_border,
                 title=title or color, xlabel="x", ylabel="y")
    ax.set_aspect("equal", "datalim")
    ax.set_facecolor("#f0f0f0")
    plt.tight_layout()
    return _save_or_show(fig, save, show=show, dpi=dpi)


def plot_spatial_metabolite(
    adata: ad.AnnData,
    metabolite: str,
    cmap: str = "hot",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    fontsize: int = 11,
    show_grid: bool = False,
    show_axes_border: bool = True,
    save: Union[bool, str, None] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Single-metabolite spatial intensity map."""
    if metabolite not in adata.var_names:
        raise KeyError(
            f"Metabolite '{metabolite}' not found in adata.var_names. "
            "Check spelling or use adata.var_names to list available metabolites."
        )
    return plot_spatial(
        adata, color=metabolite, cmap=cmap,
        figsize=figsize, dpi=dpi, fontsize=fontsize, title=metabolite,
        show_grid=show_grid, show_axes_border=show_axes_border, save=save, show=show, **kwargs
    )


def plot_embedding_grid(
    adata: ad.AnnData,
    metabolites: List[str],
    ncols: int = 4,
    cmap: str = "viridis",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    fontsize: int = 9,
    show_grid: bool = False,
    show_axes_border: bool = False,
    save: Union[bool, str, None] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Grid of spatial intensity maps for a list of metabolites."""
    found = [m for m in metabolites if m in adata.var_names]
    missing = [m for m in metabolites if m not in adata.var_names]
    if missing:
        print(f"[MORTIS] Warning: skipping {len(missing)} metabolites not found: {missing[:5]}")
    if not found:
        raise ValueError("No valid metabolites found in adata.var_names.")

    coords = adata.obsm["spatial"]
    layer = "log1p" if "log1p" in adata.layers else None
    X = _to_dense(adata.layers[layer] if layer else adata.X)

    nrows = int(np.ceil(len(found) / ncols))
    fs = figsize or _auto_figsize(ncols, nrows, base=3.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=fs, dpi=dpi)
    axes = np.array(axes).flatten()

    spot_size = _get_spot_size(adata, _pop_spot_size_kwarg(kwargs))
    scatter_args = {"alpha": 1.0, "edgecolors": "none", "rasterized": True}
    scatter_args.update(kwargs)

    for i, met in enumerate(found):
        ax = axes[i]
        idx = adata.var_names.get_loc(met)
        vals = X[:, idx]
        sc_obj = ax.scatter(coords[:, 0], coords[:, 1], c=vals,
                        cmap=cmap, s=spot_size, **scatter_args)
        plt.colorbar(sc_obj, ax=ax, shrink=0.7, pad=0.02)
        _apply_style(ax, fontsize, show_grid, show_axes_border, title=met[:30])
        ax.set_aspect("equal", "datalim")
        ax.set_facecolor("#f0f0f0")
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(len(found), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return _save_or_show(fig, save, show=show, dpi=dpi)


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def plot_umap(
    adata: ad.AnnData,
    color: str = "cluster",
    palette: str = "tab20",
    cmap: str = "viridis",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    fontsize: int = 11,
    title: Optional[str] = None,
    show_grid: bool = False,
    show_axes_border: bool = True,
    save: Union[bool, str, None] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """UMAP embedding coloured by cluster, condition, or metabolite intensity."""
    from .exceptions import NoEmbeddingError
    if "X_umap" not in adata.obsm:
        raise NoEmbeddingError(
            "UMAP embedding not found. Run MORTIS.run_umap(adata) first."
        )

    spot_size = _get_spot_size(adata, _pop_spot_size_kwarg(kwargs))
    scatter_args = {"alpha": 0.8, "edgecolors": "none", "rasterized": True}
    scatter_args.update(kwargs)

    umap = adata.obsm["X_umap"]
    values = _get_metabolite_values(adata, color)
    is_cat = (
        pd.api.types.is_object_dtype(values)
        or isinstance(getattr(values, "dtype", type(values)), pd.CategoricalDtype)
    )
    fs = figsize or (7, 6)
    fig, ax = plt.subplots(figsize=fs, dpi=dpi)

    if is_cat:
        categories = np.unique(values.astype(str))
        colors = sns.color_palette(palette, len(categories))
        for i, cat in enumerate(categories):
            mask = values.astype(str) == cat
            ax.scatter(umap[mask, 0], umap[mask, 1],
                       c=[colors[i]], s=spot_size, label=str(cat), **scatter_args)
        ax.legend(markerscale=max(1, 10/spot_size), bbox_to_anchor=(1.01, 1), loc="upper left",
                  fontsize=fontsize - 2, frameon=False)
    else:
        sc_obj = ax.scatter(umap[:, 0], umap[:, 1], c=values.astype(float),
                        cmap=cmap, s=spot_size, **scatter_args)
        plt.colorbar(sc_obj, ax=ax, shrink=0.8)

    _apply_style(ax, fontsize, show_grid, show_axes_border,
                 title=title or f"UMAP — {color}",
                 xlabel="UMAP 1", ylabel="UMAP 2")
    ax.set_facecolor("#f8f8f8")
    plt.tight_layout()
    return _save_or_show(fig, save, show=show, dpi=dpi)


# ---------------------------------------------------------------------------
# Marker / differential expression plots
# ---------------------------------------------------------------------------

def plot_markers(
    adata: ad.AnnData,
    cluster_key: str = "cluster",
    n_top: int = 5,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    fontsize: int = 11,
    show_grid: bool = False,
    show_axes_border: bool = True,
    save: Union[bool, str, None] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Dot plot of top marker metabolites per cluster."""
    import scanpy as sc

    from .exceptions import NoClustersError

    if "rank_genes_groups" not in adata.uns:
        raise NoClustersError(
            "Marker results not found. Run MORTIS.find_markers(adata) first."
        )
    n_clusters = adata.obs[cluster_key].nunique() if cluster_key in adata.obs else 1
    fs = figsize or (max(8, n_top * n_clusters * 0.4), 5)
    fig, ax = plt.subplots(figsize=fs, dpi=dpi)

    kwargs.setdefault("show", False) # Always False internally to capture the object
    kwargs.setdefault("dendrogram", False) # Bypasses float32 symmetry error

    adata_t = adata.copy()
    if "log1p" in adata.layers: adata_t.X = adata_t.layers["log1p"]

    with plt.rc_context({"font.size": fontsize}):
        sc_fig = sc.pl.rank_genes_groups_dotplot(
            adata_t, n_genes=n_top, groupby=cluster_key, use_raw=False, ax=ax, **kwargs
        )

    if not show_axes_border:
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.tight_layout()

    final_fig = sc_fig['mainplot_ax'].figure if isinstance(sc_fig, dict) else plt.gcf()
    return _save_or_show(final_fig, save, show=show, dpi=dpi)


def plot_volcano(
    results_df: pd.DataFrame,
    group1: str = "Group 1",
    group2: str = "Group 2",
    fc_cutoff: float = 1.0,
    pval_cutoff: float = 0.05,
    max_log10_pval: float = 50.0,
    n_label: int = 10,
    label_top: bool = False,
    show_table: bool = False,
    n_table: int = 5,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    fontsize: int = 11,
    show_grid: bool = False,
    show_axes_border: bool = True,
    save: Union[bool, str, None] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Volcano plot with visual capping for extreme spatial P-values."""
    df = results_df.copy()
    min_pval_allowed = 10**(-max_log10_pval)
    df["-log10_pval"] = -np.log10(df["pval_adj"].clip(lower=min_pval_allowed))

    sig_up = (df["log2fc"] >= fc_cutoff) & (df["pval_adj"] < pval_cutoff)
    sig_dn = (df["log2fc"] <= -fc_cutoff) & (df["pval_adj"] < pval_cutoff)
    ns = ~(sig_up | sig_dn)

    if show_table:
        fs = figsize or (10, 11)
        fig = plt.figure(figsize=fs, dpi=dpi)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.35)
        ax = fig.add_subplot(gs[0])
        ax_table = fig.add_subplot(gs[1])
    else:
        fs = figsize or (9, 7)
        fig, ax = plt.subplots(figsize=fs, dpi=dpi)

    s = kwargs.pop("s", 12)
    scatter_args = {"alpha": 0.9, "edgecolors": "none"}
    scatter_args.update(kwargs)

    ax.scatter(df.loc[ns, "log2fc"], df.loc[ns, "-log10_pval"],
               color="lightgray", s=s*0.7, alpha=0.6, label="Not significant")
    ax.scatter(df.loc[sig_up, "log2fc"], df.loc[sig_up, "-log10_pval"],
               color="crimson", s=s, label=f"Up in {group2} ({sig_up.sum()})", **scatter_args)
    ax.scatter(df.loc[sig_dn, "log2fc"], df.loc[sig_dn, "-log10_pval"],
               color="steelblue", s=s, label=f"Up in {group1} ({sig_dn.sum()})", **scatter_args)

    ax.axvline(fc_cutoff, color="gray", linestyle="--", linewidth=1)
    ax.axvline(-fc_cutoff, color="gray", linestyle="--", linewidth=1)
    ax.axhline(-np.log10(pval_cutoff), color="gray", linestyle=":", linewidth=1)

    if label_top and n_label > 0:
        try:
            from adjustText import adjust_text
        except ImportError:
            raise ImportError("Please 'pip install adjustText' for non-overlapping labels.")

        top_hits = df[sig_up | sig_dn].nsmallest(n_label, "pval_adj")
        texts = [ax.text(row["log2fc"], row["-log10_pval"], row["metabolite"][:25], fontsize=fontsize - 3) for _, row in top_hits.iterrows()]
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='black', lw=0.5))

    _apply_style(ax, fontsize, show_grid, show_axes_border,
                 title=kwargs.get('title', f"Volcano Plot: {group1} vs {group2} (Capped at {max_log10_pval})"),
                 xlabel=f"log\u2082 Fold Change ({group2} / {group1})",
                 ylabel="-log\u2081\u2080 adjusted p-value")
    ax.legend(fontsize=fontsize - 1, frameon=False)

    if show_table:
        top_up = df[sig_up].nsmallest(n_table, "pval_adj")[
            ["metabolite", "log2fc", "pval_adj"]
        ].copy()
        if len(top_up) > 0:
            top_up["log2fc"] = top_up["log2fc"].round(3)
            top_up["pval_adj"] = top_up["pval_adj"].apply(lambda x: f"{x:.2e}")
            top_up.columns = ["Metabolite", "log2FC", "adj. p-value"]
            ax_table.axis("off")
            tbl = ax_table.table(
                cellText=top_up.values,
                colLabels=top_up.columns,
                cellLoc="center",
                loc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(fontsize - 1)
            tbl.auto_set_column_width(col=list(range(len(top_up.columns))))
            ax_table.set_title(
                f"Top {len(top_up)} upregulated in {group2}",
                fontsize=fontsize, pad=4,
            )

    plt.tight_layout()
    return _save_or_show(fig, save, show=show, dpi=dpi)


def plot_heatmap(
    adata: ad.AnnData,
    metabolites: List[str],
    groupby: str = "cluster",
    cmap: str = "RdBu_r",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    fontsize: int = 10,
    show_grid: bool = False,
    show_axes_border: bool = True,
    save: Union[bool, str, None] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Heatmap of mean metabolite intensities across groups."""
    import scanpy as sc

    from .exceptions import InvalidParameterError

    if groupby not in adata.obs.columns:
        raise InvalidParameterError(f"'{groupby}' not found in adata.obs. ")
    found = [m for m in metabolites if m in adata.var_names]
    if not found:
        raise InvalidParameterError("None of the provided metabolites were found in adata.var_names.")

    kwargs.setdefault("show", False)
    kwargs.setdefault("cmap", cmap)
    kwargs.setdefault("dendrogram", False)

    fs = figsize or (min(12, len(found)*0.5), 6)

    adata_t = adata.copy()
    if "log1p" in adata.layers: adata_t.X = adata_t.layers["log1p"]

    with plt.rc_context({"font.size": fontsize}):
        sc_fig = sc.pl.heatmap(adata_t, var_names=found, groupby=groupby, use_raw=False, figsize=fs, **kwargs)

    final_fig = sc_fig['heatmap_ax'].figure if isinstance(sc_fig, dict) else plt.gcf()
    return _save_or_show(final_fig, save, show=show, dpi=dpi)


def plot_violin(
    adata: ad.AnnData,
    metabolites: Union[str, List[str]],
    groupby: str = "cluster",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    fontsize: int = 10,
    show_grid: bool = False,
    show_axes_border: bool = True,
    palette: str = "tab20",
    save: Union[bool, str, None] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Violin plots of metabolite intensity distributions per group."""
    import scanpy as sc

    from .exceptions import InvalidParameterError

    if groupby not in adata.obs.columns:
        raise InvalidParameterError(f"'{groupby}' not found in adata.obs.")

    metabolites = [metabolites] if isinstance(metabolites, str) else metabolites
    found = [m for m in metabolites if m in adata.var_names]
    if not found:
        raise InvalidParameterError("None of the provided metabolites were found in adata.var_names.")

    kwargs.setdefault("show", False)

    fs = figsize or (min(10, len(found)*3), 4)

    adata_t = adata.copy()
    if "log1p" in adata.layers: adata_t.X = adata_t.layers["log1p"]

    with plt.rc_context({"figure.figsize": fs, "font.size": fontsize}):
        sc.pl.violin(adata_t, keys=found, groupby=groupby, use_raw=False, palette=palette, **kwargs)

    fig = plt.gcf()

    for ax in fig.axes:
        if show_grid:
            ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.6)
        else:
            ax.grid(False)
        if not show_axes_border:
            for spine in ax.spines.values():
                spine.set_visible(False)

    return _save_or_show(fig, save, show=show, dpi=dpi)


def plot_cluster_composition(
    adata: ad.AnnData,
    cluster_key: str = "cluster",
    groupby: str = "condition",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    fontsize: int = 11,
    show_grid: bool = False,
    show_axes_border: bool = True,
    palette: str = "tab20",
    normalize: bool = True,
    save: Union[bool, str, None] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Stacked bar chart showing cluster composition per condition or sample."""
    from .exceptions import InvalidParameterError, NoClustersError
    if cluster_key not in adata.obs.columns:
        raise NoClustersError(f"Cluster key '{cluster_key}' not found. ")
    if groupby not in adata.obs.columns:
        raise InvalidParameterError(f"'{groupby}' not found in adata.obs.")

    ct = pd.crosstab(adata.obs[groupby], adata.obs[cluster_key])
    if normalize:
        ct = ct.div(ct.sum(axis=1), axis=0) * 100

    clusters = ct.columns.tolist()
    colors = sns.color_palette(palette, len(clusters))
    n_groups = len(ct)
    fs = figsize or (max(5, n_groups * 1.2), 5)
    fig, ax = plt.subplots(figsize=fs, dpi=dpi)

    bar_args = {"edgecolor": "white", "linewidth": 0.5}
    bar_args.update(kwargs)

    bottom = np.zeros(n_groups)
    for i, clust in enumerate(clusters):
        vals = ct[clust].values
        ax.bar(ct.index, vals, bottom=bottom, color=colors[i],
               label=str(clust), **bar_args)
        bottom += vals

    ylabel = "Proportion (%)" if normalize else "Pixel count"
    _apply_style(ax, fontsize, show_grid, show_axes_border,
                 title=f"Cluster Composition by {groupby}",
                 xlabel=groupby, ylabel=ylabel)
    ax.legend(title=cluster_key, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=fontsize - 2, frameon=False)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return _save_or_show(fig, save, show=show, dpi=dpi)


# ---------------------------------------------------------------------------
# KILLER FEATURE PLOTTING
# ---------------------------------------------------------------------------

def plot_morans(
    morans_df: pd.DataFrame,
    n_top: int = 20,
    dpi: int = 300,
    save: Union[bool, str, None] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Horizontal bar chart of top spatially variable metabolites by Moran's I."""
    top = morans_df.head(n_top).copy()
    top["label"] = top["metabolite"].str[:50]

    fig, ax = plt.subplots(figsize=(9, max(4, n_top * 0.38)), dpi=dpi)

    # Pure color representation: Blue for positive spatial autocorrelation
    colors = ["#2196F3" if v > 0 else "#F44336" for v in top["morans_i"]]

    ax.barh(top["label"][::-1], top["morans_i"][::-1], color=colors[::-1], edgecolor="white", height=0.7, **kwargs)
    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_title(f"Top {n_top} Spatially Variable Metabolites")
    ax.set_xlabel("Moran's I (Spatial Autocorrelation)")

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return _save_or_show(fig, save, show=show, dpi=dpi)


def plot_spatial_gradient(
    gradient_df: pd.DataFrame,
    top_n: int = 5,
    dpi: int = 300,
    save: Union[bool, str, None] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Produces a continuous, shaded line-plot representation of spatial gradients."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    mets = gradient_df.drop(columns=["distance"]).mean().nlargest(top_n).index

    for met in mets:
        sns.lineplot(data=gradient_df, x="distance", y=met, ax=ax, label=met[:30], marker="o", linewidth=2.5, **kwargs)

    ax.set_title("Spatial Metabolic Gradient Profile")
    ax.set_xlabel("Physical Distance from Target (µm)")
    ax.set_ylabel("Smoothed Mean Intensity")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)

    plt.tight_layout()
    return _save_or_show(fig, save, show=show, dpi=dpi)


def plot_colocalization_network(
    edges_df: pd.DataFrame,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
    save: Union[bool, str, None] = None,
    show: bool = True,
    **kwargs
) -> plt.Figure:
    """Draws a physical interactome graph of spatially co-localized metabolites."""
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("networkx is required for interactome plotting. Install with: pip install networkx")

    G = nx.from_pandas_edgelist(edges_df, "source", "target", ["weight"])
    fig, ax = plt.subplots(figsize=figsize or (8, 8), dpi=dpi)
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=kwargs.get("node_size", 400), node_color="skyblue", alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=weights, edge_color="gray", alpha=0.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=kwargs.get("fontsize", 8), font_family="sans-serif", ax=ax)

    ax.set_title("Spatial Metabolite Interactome", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    return _save_or_show(fig, save, show=show, dpi=dpi)
