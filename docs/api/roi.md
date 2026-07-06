# ROI Selection

Use these when your tissue/background files **aren't** already paired
by `load_from_folder()` — e.g. a single unlabeled file where tissue and
background occupy the same coordinate space and need to be drawn manually.

## `draw_ROIs`

```python
mt.draw_ROIs(adata: anndata.AnnData) -> anndata.AnnData
```

Opens an interactive Matplotlib window showing the total-ion-current
(TIC) map of your sample.

**How to use it:**

1. A window opens showing your tissue, colored by total signal.
2. Click points around the tissue region to draw a polygon (green),
   then click **Confirm Tissue**.
3. Draw a second polygon around a representative background region
   (white), then click **Confirm Background**.
4. The window closes automatically.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `adata` | `AnnData` | :material-check: | Must have `adata.obsm['spatial']` |

**Returns:** `AnnData` with `is_tissue`/`is_background` boolean columns
added to `.obs`.

```python
adata = mt.draw_ROIs(adata)
```

!!! note "Requires a display"
    This opens a real GUI window, so it only works in an environment
    with display access (a local machine, not a headless server/CI).

---

## `draw_ROIs_for_folder`

```python
mt.draw_ROIs_for_folder(adatas: list[anndata.AnnData]) -> list[anndata.AnnData]
```

Runs [`draw_ROIs()`](#draw_rois) sequentially for every sample in a list
— convenient after `mt.check_rois()` tells you which samples are missing
labels.

```python
try:
    mt.check_rois(adatas)
except mt.MissingROIError:
    adatas = mt.draw_ROIs_for_folder(adatas)
```
