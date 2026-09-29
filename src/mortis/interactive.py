from typing import List

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import Button, PolygonSelector


def draw_ROIs(adata: ad.AnnData) -> ad.AnnData:
    """
    Single-window interactive ROI selector.
    User draws Tissue (Green) -> Clicks Confirm -> Draws Background (White) -> Clicks Confirm.
    """
    print("\n--- [MORTIS] INTERACTIVE ROI SELECTION ---")
    tic = adata.X.sum(axis=1)
    coords = adata.obsm['spatial']

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.2)

    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=tic, cmap='hot', s=5, alpha=0.9)
    plt.colorbar(scatter, label='Total Ion Current (TIC)')

    state = {
        'tissue_mask': np.zeros(len(coords), dtype=bool),
        'bg_mask': np.zeros(len(coords), dtype=bool)
    }

    ax.set_title("STEP 1: Draw TISSUE (Green)\nAdjust points, then click 'Confirm Tissue' below.", fontsize=12, fontweight='bold')

    selector = PolygonSelector(
        ax, lambda verts: None,
        props=dict(color='green', linewidth=2, alpha=0.8),
        handle_props=dict(marker='o', markersize=3, color='green')
    )

    ax_btn_t = plt.axes([0.3, 0.05, 0.15, 0.075])
    btn_tissue = Button(ax_btn_t, 'Confirm Tissue', color='lightgreen')

    ax_btn_b = plt.axes([0.55, 0.05, 0.18, 0.075])
    btn_bg = Button(ax_btn_b, 'Confirm Background', color='lightgray')
    btn_bg.ax.set_visible(False)

    def confirm_tissue(event):
        if not selector.verts or len(selector.verts) < 3:
            print("[MORTIS] Please draw a completed polygon first!")
            return

        path = Path(selector.verts)
        state['tissue_mask'] = path.contains_points(coords)
        print(f"[MORTIS] Tissue registered! ({state['tissue_mask'].sum()} pixels)")

        ax.scatter(coords[state['tissue_mask'], 0], coords[state['tissue_mask'], 1],
                   facecolors='none', edgecolors='green', s=10, alpha=0.6)

        ax.set_title("STEP 2: Draw BACKGROUND (White)\nAdjust points, then click 'Confirm Background' below.", fontsize=12, fontweight='bold')
        selector.clear()

        if hasattr(selector, '_polygon'):
            selector._polygon.set_color('white')
        if hasattr(selector, '_handles') and hasattr(selector._handles, 'artists'):
            for handle in selector._handles.artists:
                handle.set_color('white')
                handle.set_markeredgecolor('white')
                handle.set_markerfacecolor('white')

        btn_tissue.ax.set_visible(False)
        btn_bg.ax.set_visible(True)
        fig.canvas.draw()

    def confirm_background(event):
        if not selector.verts or len(selector.verts) < 3:
            print("[MORTIS] Please draw a completed polygon first!")
            return

        path = Path(selector.verts)
        state['bg_mask'] = path.contains_points(coords)
        print(f"[MORTIS] Background registered! ({state['bg_mask'].sum()} pixels)")
        plt.close(fig)

    btn_tissue.on_clicked(confirm_tissue)
    btn_bg.on_clicked(confirm_background)

    plt.show()

    adata.obs['is_tissue'] = state['tissue_mask']
    adata.obs['is_background'] = state['bg_mask']

    return adata


def draw_ROIs_for_folder(adatas: List[ad.AnnData]) -> List[ad.AnnData]:
    """
    Draw tissue and background ROIs for a list of sections, one after another.

    Opens the interactive selector once per section and collects the results, so
    a folder loaded with :func:`load_from_folder` can be annotated in a single
    pass. Needs an interactive matplotlib backend.

    Parameters
    ----------
    adatas : list of anndata.AnnData
        Sections to annotate, each with spatial coordinates.

    Returns
    -------
    list of anndata.AnnData
        The same sections, each with ``is_tissue`` and ``is_background`` in
        ``.obs``, ready for :func:`filter_background`.
    """
    processed_adatas = []
    for i, adata in enumerate(adatas):
        print(f"\n[MORTIS] Processing Unpaired Sample {i+1} of {len(adatas)}...")
        updated_adata = draw_ROIs(adata)
        processed_adatas.append(updated_adata)
    return processed_adatas
