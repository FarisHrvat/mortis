"""
MORTIS Image Module
===================
Load, align, and overlay histology / fluorescence TIFF images with spatial
metabolomics data.

After instrument acquisition, users often have a co-registered histology image
(H&E, DAPI, etc.) alongside the MSI data.  This module lets you:

* Load a TIFF (single-channel or RGB, any bit-depth) into the AnnData object.
* Align the image to the MSI coordinate grid via affine transformation.
* Extract per-pixel image features (mean intensity in a neighbourhood).
* Overlay the MSI spatial map on top of the histology image.

Public API
----------
load_image(adata, image_path, image_key, copy)
    Load a TIFF image and store it in ``adata.uns``.

align_image(adata, image_key, scale_x, scale_y, offset_x, offset_y, copy)
    Apply an affine (scale + translate) alignment to map image pixels to MSI
    coordinates.

extract_image_features(adata, image_key, radius, copy)
    Sample mean image intensity in a circular neighbourhood around each MSI
    pixel and store as an obs column.

plot_image_overlay(adata, image_key, color, alpha_spots, spot_size, figsize,
                   dpi, save)
    Overlay MSI spots on the histology image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .exceptions import FileFormatError, InvalidParameterError, MissingSpatialError
from .plotting import _get_metabolite_values, _save_or_show


def load_image(
    adata: ad.AnnData,
    image_path: str,
    image_key: str = "histology",
    copy: bool = False,
) -> ad.AnnData:
    """
    Load a TIFF image and store it in ``adata.uns[image_key]``.

    Supports single-channel (grayscale), multi-channel, and RGB TIFFs of any
    bit-depth.  The image is stored as a float32 array normalised to [0, 1].

    Parameters
    ----------
    adata : anndata.AnnData
        Target AnnData object.
    image_path : str
        Path to the TIFF file.
    image_key : str, optional
        Key under which the image is stored in ``adata.uns``.
        Default: ``'histology'``.
    copy : bool, optional
        Return a copy instead of modifying in-place.  Default: False.

    Returns
    -------
    anndata.AnnData
        AnnData with ``adata.uns[image_key]`` containing a dict:

        * ``'data'``: float32 ndarray, shape (H, W) or (H, W, C)
        * ``'shape'``: (H, W) or (H, W, C)
        * ``'path'``: original file path string

    Raises
    ------
    FileNotFoundError
        If the TIFF file does not exist.
    FileFormatError
        If the file cannot be read as a TIFF.

    Examples
    --------
    >>> adata = mt.load_image(adata, "histology.tif")
    >>> print(adata.uns['histology']['shape'])
    (1024, 1024, 3)
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError(
            "tifffile is required for image loading. "
            "Install it with: pip install tifffile"
        )

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Image file not found: '{image_path}'. "
            "Please check the path and try again."
        )

    try:
        img = tifffile.imread(str(path))
    except Exception as exc:
        raise FileFormatError(
            f"Could not read '{path.name}' as a TIFF file.\n"
            f"Original error: {exc}"
        ) from exc

    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / img.max()

    if copy:
        adata = adata.copy()

    adata.uns[image_key] = {
        "data": img,
        "shape": img.shape,
        "path": str(path),
        "aligned": False,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "offset_x": 0.0,
        "offset_y": 0.0,
    }
    print(
        f"[MORTIS] Loaded image '{path.name}' -> adata.uns['{image_key}'] "
        f"shape={img.shape}, dtype=float32"
    )
    return adata


def align_image(
    adata: ad.AnnData,
    image_key: str = "histology",
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    copy: bool = False,
) -> ad.AnnData:
    """
    Register the image to MSI coordinates via affine (scale + translate).

    The transformation maps image pixel coordinates to MSI spot coordinates:

        ``msi_x = image_col * scale_x + offset_x``
        ``msi_y = image_row * scale_y + offset_y``

    Parameters
    ----------
    adata : anndata.AnnData
        Must have the image loaded via :func:`load_image`.
    image_key : str, optional
        Key of the image in ``adata.uns``.  Default: ``'histology'``.
    scale_x : float, optional
        Pixels-per-MSI-unit in the x direction.  Default: 1.0.
    scale_y : float, optional
        Pixels-per-MSI-unit in the y direction.  Default: 1.0.
    offset_x : float, optional
        Translation offset in x (MSI units).  Default: 0.0.
    offset_y : float, optional
        Translation offset in y (MSI units).  Default: 0.0.
    copy : bool, optional
        Return a copy instead of modifying in-place.  Default: False.

    Returns
    -------
    anndata.AnnData
        AnnData with updated alignment parameters in ``adata.uns[image_key]``.

    Raises
    ------
    InvalidParameterError
        If the image key is not found in ``adata.uns``.

    Examples
    --------
    >>> adata = mt.align_image(adata, scale_x=0.5, scale_y=0.5, offset_x=10)
    """
    if image_key not in adata.uns:
        raise InvalidParameterError(
            f"Image key '{image_key}' not found in adata.uns. "
            "Run mt.load_image(adata...) first."
        )
    if copy:
        adata = adata.copy()

    adata.uns[image_key].update({
        "aligned": True,
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "offset_x": float(offset_x),
        "offset_y": float(offset_y),
    })
    print(
        f"[MORTIS] Image '{image_key}' aligned: "
        f"scale=({scale_x}, {scale_y}), offset=({offset_x}, {offset_y})"
    )
    return adata


def extract_image_features(
    adata: ad.AnnData,
    image_key: str = "histology",
    radius: int = 3,
    copy: bool = False,
) -> ad.AnnData:
    """
    Extract mean image intensity in a circular neighbourhood around each MSI
    pixel and store as ``adata.obs`` columns.

    For RGB images, three columns are added (``{image_key}_R``,
    ``{image_key}_G``, ``{image_key}_B``).  For grayscale, one column
    (``{image_key}_intensity``) is added.

    Parameters
    ----------
    adata : anndata.AnnData
        Must have ``adata.obsm['spatial']`` and the image loaded via
        :func:`load_image`.
    image_key : str, optional
        Key of the image in ``adata.uns``.  Default: ``'histology'``.
    radius : int, optional
        Neighbourhood radius in image pixels.  Default: 3.
    copy : bool, optional
        Return a copy instead of modifying in-place.  Default: False.

    Returns
    -------
    anndata.AnnData
        AnnData with new columns in ``adata.obs``.

    Raises
    ------
    MissingSpatialError
        If spatial coordinates are missing.
    InvalidParameterError
        If the image key is not found.

    Examples
    --------
    >>> adata = mt.extract_image_features(adata, radius=5)
    >>> print(adata.obs[['histology_R', 'histology_G', 'histology_B']].head())
    """
    if "spatial" not in adata.obsm:
        raise MissingSpatialError(
            "adata.obsm['spatial'] is missing. "
            "Ensure spatial coordinates are loaded."
        )
    if image_key not in adata.uns:
        raise InvalidParameterError(
            f"Image key '{image_key}' not found in adata.uns. "
            "Run mt.load_image(adata...) first."
        )
    if copy:
        adata = adata.copy()

    img_info = adata.uns[image_key]
    img = img_info["data"]
    sx = img_info.get("scale_x", 1.0)
    sy = img_info.get("scale_y", 1.0)
    ox = img_info.get("offset_x", 0.0)
    oy = img_info.get("offset_y", 0.0)

    coords = adata.obsm["spatial"]
    H = img.shape[0]
    W = img.shape[1]
    is_rgb = img.ndim == 3 and img.shape[2] >= 3

    col_idx = np.clip(((coords[:, 0] - ox) / sx).astype(int), 0, W - 1)
    row_idx = np.clip(((coords[:, 1] - oy) / sy).astype(int), 0, H - 1)

    # Build circular kernel offsets
    dy, dx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    circle = (dx ** 2 + dy ** 2) <= radius ** 2
    dy_off = dy[circle]
    dx_off = dx[circle]

    n = len(coords)
    if is_rgb:
        features = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            rows = np.clip(row_idx[i] + dy_off, 0, H - 1)
            cols = np.clip(col_idx[i] + dx_off, 0, W - 1)
            features[i] = img[rows, cols, :3].mean(axis=0)
        adata.obs[f"{image_key}_R"] = features[:, 0]
        adata.obs[f"{image_key}_G"] = features[:, 1]
        adata.obs[f"{image_key}_B"] = features[:, 2]
        print(f"[MORTIS] Extracted RGB features -> adata.obs['{image_key}_R/G/B']")
    else:
        img_2d = img if img.ndim == 2 else img[:, :, 0]
        features = np.zeros(n, dtype=np.float32)
        for i in range(n):
            rows = np.clip(row_idx[i] + dy_off, 0, H - 1)
            cols = np.clip(col_idx[i] + dx_off, 0, W - 1)
            features[i] = img_2d[rows, cols].mean()
        adata.obs[f"{image_key}_intensity"] = features
        print(f"[MORTIS] Extracted intensity features -> adata.obs['{image_key}_intensity']")

    return adata


def plot_image_overlay(
    adata: ad.AnnData,
    image_key: str = "histology",
    color: Optional[str] = None,
    alpha_spots: float = 0.7,
    spot_size: float = 4.0,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 150,
    save: Union[bool, str, None] = None,
) -> plt.Figure:
    """
    Overlay MSI spots on the histology image.

    Parameters
    ----------
    adata : anndata.AnnData
        Must have ``adata.obsm['spatial']`` and the image in ``adata.uns``.
    image_key : str, optional
        Key of the image in ``adata.uns``.  Default: ``'histology'``.
    color : str or None, optional
        Column in ``adata.obs`` or metabolite name to colour the spots.
        If ``None``, spots are shown in a single colour.
    alpha_spots : float, optional
        Transparency of the MSI spots (0 = invisible, 1 = opaque).
        Default: 0.7.
    spot_size : float, optional
        Marker size.  Default: 4.0.
    figsize : tuple of int, optional
        Figure size in inches ``(width, height)``.  Default: ``(10, 8)``.
    dpi : int, optional
        Resolution in dots per inch.  Default: 150.
    save : bool, str, or None, optional
        * ``None``: display interactively.
        * ``True``: save as ``'mortis_plot.pdf'``.
        * ``'path/to/file.png'``: save to the given path.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> mt.plot_image_overlay(adata, color='cluster', alpha_spots=0.6)
    """
    if image_key not in adata.uns:
        raise InvalidParameterError(
            f"Image key '{image_key}' not found in adata.uns. "
            "Run mt.load_image(adata...) first."
        )

    img_info = adata.uns[image_key]
    img = img_info["data"]
    sx = img_info.get("scale_x", 1.0)
    sy = img_info.get("scale_y", 1.0)
    ox = img_info.get("offset_x", 0.0)
    oy = img_info.get("offset_y", 0.0)

    coords = adata.obsm["spatial"]
    img_x_min = ox
    img_x_max = ox + img.shape[1] * sx
    img_y_min = oy
    img_y_max = oy + img.shape[0] * sy

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    display_img = img if img.ndim == 2 else img[:, :, :3]
    ax.imshow(
        display_img,
        extent=[img_x_min, img_x_max, img_y_max, img_y_min],
        cmap="gray" if img.ndim == 2 else None,
        aspect="auto",
        origin="upper",
    )

    if color is not None:
        values = _get_metabolite_values(adata, color)
        is_cat = pd.api.types.is_object_dtype(values) or isinstance(getattr(values, "dtype", type(values)), pd.CategoricalDtype)
        if is_cat:
            cats = np.unique(values)
            palette = sns.color_palette("tab20", len(cats))
            for i, cat in enumerate(cats):
                mask = values == cat
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=[palette[i]], s=spot_size, alpha=alpha_spots,
                    label=str(cat), rasterized=True,
                )
            ax.legend(markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left",
                      fontsize=8, frameon=False)
        else:
            sc = ax.scatter(
                coords[:, 0], coords[:, 1], c=values,
                cmap="viridis", s=spot_size, alpha=alpha_spots, rasterized=True,
            )
            plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(f"Image Overlay, {color}", fontsize=13)
    else:
        ax.scatter(coords[:, 0], coords[:, 1],
                   c="cyan", s=spot_size, alpha=alpha_spots, rasterized=True)
        ax.set_title("Image Overlay. MSI spots", fontsize=13)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    return _save_or_show(fig, save)

