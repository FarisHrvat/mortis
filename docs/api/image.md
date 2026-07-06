# Image (Histology Overlay)

After acquisition, you often have a co-registered histology image (H&E,
DAPI, etc.) alongside the MSI data. These functions load it, register it
to your MSI coordinate grid, and overlay the two.

## `load_image`

```python
mt.load_image(adata, image_path: str, image_key: str = "histology", copy: bool = False) -> anndata.AnnData
```

Load a grayscale/RGB TIFF (any bit-depth) into `adata.uns[image_key]`,
normalized to float32 `[0, 1]`.

**Raises:** `FileNotFoundError`; [`FileFormatError`](exceptions.md#fileformaterror).

```python
adata = mt.load_image(adata, "histology.tif")
adata = mt.load_image(adata, "dapi.tif", image_key="dapi")
print(adata.uns["histology"]["shape"])
```

## `align_image`

```python
mt.align_image(
    adata, image_key: str = "histology", scale_x: float = 1.0, scale_y: float = 1.0,
    offset_x: float = 0.0, offset_y: float = 0.0, copy: bool = False,
) -> anndata.AnnData
```

Affine registration to MSI coordinates:
`msi_x = image_col * scale_x + offset_x` (analogously for `y`).

**Raises:** [`InvalidParameterError`](exceptions.md#invalidparametererror)
if `image_key` isn't in `adata.uns`.

```python
adata = mt.align_image(adata, scale_x=0.5, scale_y=0.5, offset_x=10, offset_y=5)
```

## `extract_image_features`

```python
mt.extract_image_features(adata, image_key: str = "histology", radius: int = 3, copy: bool = False) -> anndata.AnnData
```

Mean image intensity in a circular neighbourhood around each MSI pixel
→ `adata.obs[f'{image_key}_intensity']` (grayscale) or `_R`/`_G`/`_B` (RGB).

**Raises:** [`MissingSpatialError`](exceptions.md#missingspatialerror);
[`InvalidParameterError`](exceptions.md#invalidparametererror).

```python
adata = mt.extract_image_features(adata, radius=5)
mt.plot_spatial(adata, color="histology_intensity")
```

## `plot_image_overlay`

```python
mt.plot_image_overlay(
    adata, image_key: str = "histology", color: str | None = None,
    alpha_spots: float = 0.7, spot_size: float = 4.0,
    figsize: tuple[int, int] = (10, 8), dpi: int = 150, save=None,
) -> matplotlib.figure.Figure
```

Overlay MSI spots on the histology image.

```python
mt.plot_image_overlay(adata, color="cluster", alpha_spots=0.6, save="overlay.pdf")
```
