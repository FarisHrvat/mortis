"""
MORTIS Preprocessing Module
===========================
All preprocessing steps for spatial metabolomics data, from background
filtering through normalization, scaling, and dimensionality reduction.

Performance notes
-----------------
* PCA and UMAP run on the GPU through cuML when it is installed, and on the
  CPU otherwise. Nothing to configure.
* Matrices are float32 and modified in place where that is safe.
* ``filter_background`` works off column sums rather than densifying.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from contextlib import contextmanager
from typing import List, Optional, Tuple

import anndata as ad
import numba as nb
import numpy as np
import scanpy as sc
from scipy.sparse import issparse
from threadpoolctl import threadpool_limits

from .exceptions import (
    InvalidParameterError,
    MissingROIError,
    MissingSpatialError,
    MortisError,
    NoEmbeddingError,
    listing,
    suggest,
)

_N_JOBS = int(os.environ.get("MORTIS_N_JOBS", os.cpu_count() or 1))
_SKLEARN_THREAD_LIMIT = max(1, _N_JOBS)

def _configure_runtime_threads():
    """
    Return a context manager that actually limits the BLAS/OpenMP thread
    pool for the CPU-bound call it wraps, honouring ``MORTIS_N_JOBS``.

    Setting ``os.environ['OMP_NUM_THREADS']`` (or MKL_/OPENBLAS_NUM_THREADS)
    at runtime has **no effect** here: OpenBLAS/MKL read those variables
    once, the first time their thread pool initialises (typically at
    NumPy/SciPy import, or the first BLAS call anywhere in the process),
    setting them later, deep inside a function call, is silently ignored.
    Measured on this package's own PCA step: an *actual* env-var-before-
    process-start change gave a 5.5x speedup (OMP_NUM_THREADS=1 vs 16 on a
    30k x 1952 matrix multiply), while the old ``os.environ.setdefault()``
    approach measured 0.0x difference (MORTIS_N_JOBS=1 vs 16 gave identical
    wall time). ``threadpoolctl`` talks to the already-loaded BLAS library
    directly, so it works regardless of when it's called.
    """
    return threadpool_limits(limits=_SKLEARN_THREAD_LIMIT)

@contextmanager
def _numba_thread_limit():
    """
    Context manager that sets Numba's thread count for the call it wraps,
    then restores the previous value.

    ``sc.pp.neighbors``/``sc.tl.umap`` dispatch to ``pynndescent`` (a
    Numba-JIT approximate nearest-neighbour library) for anything but tiny
    datasets, regardless of the ``metric`` chosen. Numba's own thread pool
    is a *separate* mechanism from the BLAS one ``_configure_runtime_threads``
    controls, and is not affected by it, so this keeps ``MORTIS_N_JOBS``
    consistent across both.

    IMPORTANT caveat found while measuring this (don't be misled by it):
    the *first* call to ``run_neighbors``/``run_umap`` in a process pays a
    one-time ~15-17s Numba JIT-compilation cost for pynndescent's kernels
    (on a 30k-pixel dataset), regardless of thread count, every
    *subsequent* call in the same process is ~1.3-1.5s regardless of
    whether it uses 1 or 16 threads. An early version of this fix
    benchmarked "1 thread vs 16 threads" back-to-back in the same process
    and attributed the entire ~12x difference to thread scaling; re-testing
    with the run order reversed (16 threads first, then 1) showed both
    taking ~17s on the *first* call and both ~1.4s afterward, i.e. the
    original benchmark was confounded by JIT warm-up, not a genuine
    threading effect. Thread count does still matter for very large
    datasets/high k where the post-compilation query itself is
    non-trivial, but it is not the explanation for the large first-call
    latency users will actually observe, and no amount of ``MORTIS_N_JOBS``
    tuning avoids that one-time cost.
    """
    prior = nb.get_num_threads()
    nb.set_num_threads(_SKLEARN_THREAD_LIMIT)
    try:
        yield
    finally:
        nb.set_num_threads(prior)

def _get_hardware_backend(use_hardware: bool) -> str:
    """Return "cuda" if cuML is importable, otherwise "cpu"."""
    if not use_hardware:
        return "cpu"
    if importlib.util.find_spec("cuml") is not None:
        return "cuda"
    return "cpu"

def _to_dense(X) -> np.ndarray:
    if issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    arr = np.asarray(X)
    return arr.astype(np.float32, copy=False)


def _to_dense_writable(X) -> np.ndarray:
    """Dense float32 that we are allowed to write into.

    astype(copy=False) hands back the caller's own array when it is already
    float32, and that array can be read-only: anndata serves one straight out
    of the file in backed mode, and a memory-mapped or shared array behaves
    the same way. An in-place ufunc on it raises instead of normalising.
    """
    arr = _to_dense(X)
    return arr if arr.flags.writeable else arr.copy()

def _col_means_masked(X, mask: np.ndarray) -> np.ndarray:
    mask_sum = max(mask.sum(), 1)
    if issparse(X):
        mask_float = mask.astype(np.float32).reshape(1, -1)
        sums = mask_float @ X
        return np.asarray(sums).ravel() / mask_sum
    return np.asarray(X[mask].sum(axis=0, dtype=np.float32)).ravel() / mask_sum

def _check_roi(adata: ad.AnnData) -> None:
    for col in ("is_tissue", "is_background"):
        if col not in adata.obs.columns:
            raise MissingROIError(
                f"Background filtering compares tissue against off-tissue pixels, "
                f"and adata.obs has no {col!r} column to tell them apart. Either "
                f"draw the regions with mortis.draw_ROIs(adata), or load the "
                f"tissue and background exports as a pair so the labels are set "
                f"for you. Columns present: {listing(adata.obs.columns)}."
            )

def _check_spatial(adata: ad.AnnData) -> None:
    if "spatial" not in adata.obsm:
        raise MissingSpatialError(
            "This object has no pixel coordinates in adata.obsm['spatial']. "
            "mortis.read_file() fills them in from the 'x' and 'y' columns; if "
            "you built the object yourself, set "
            "adata.obsm['spatial'] = adata.obs[['x', 'y']].to_numpy(float)."
        )

def _record_step(adata: ad.AnnData, step: str) -> None:
    steps = adata.uns.setdefault("preprocessed_steps", [])
    if step not in steps:
        steps.append(step)

# ---------------------------------------------------------------------------
# Background filtering
# ---------------------------------------------------------------------------

def filter_background(
    adatas: List[ad.AnnData],
    cutoff: float = 1.5,
    mode: str = "sample",
) -> Tuple[List[ad.AnnData], List[dict]]:
    """
    Drop off-tissue pixels by comparing them to the background ROI.

    A pixel is kept when its total signal exceeds ``cutoff`` times the mean of
    the background region, so the threshold is set by the slide itself rather
    than by an absolute intensity that would not carry between runs.

    Parameters
    ----------
    adatas : list of anndata.AnnData
        Sections with ``is_tissue`` and ``is_background`` in ``.obs``, which
        :func:`draw_ROIs` or paired tissue/background files provide.
    cutoff : float, optional
        Multiple of the background mean a pixel has to clear. Default 1.5.
        Near 1.0 keeps almost everything, near 3.0 keeps only clearly
        on-tissue pixels.
    mode : {'sample', 'group'}, optional
        ``'sample'`` computes a threshold per section, which suits slides that
        were acquired separately. ``'group'`` pools the background across the
        whole list and uses one threshold, which suits serial sections from a
        single run. Default ``'sample'``.

    Returns
    -------
    list of anndata.AnnData
        The sections with background pixels removed.
    list of dict
        One QC record per section: pixels before and after, the threshold, and
        the fraction kept.

    Raises
    ------
    InvalidParameterError
        If ``cutoff`` is not positive, or ``mode`` is neither option.
    MissingROIError
        If a section has no ROI labels.
    """
    if cutoff <= 0:
        raise InvalidParameterError(
            f"cutoff is a multiple of the background mean, so it has to be "
            f"positive; got {cutoff}. Values near 1.0 keep almost everything, "
            f"values near 3.0 keep only clearly on-tissue pixels."
        )
    if mode not in ("sample", "group"):
        raise InvalidParameterError(
            f"mode must be 'sample' (one threshold per section) or 'group' (one "
            f"threshold shared across the list), got {mode!r}."
        )
    for adata in adatas: _check_roi(adata)

    pseudo = 1e-9
    filtered_tissues, qc_stats = [], []

    if mode == "group":
        t_sums = np.zeros(adatas[0].n_vars, dtype=np.float64)
        b_sums = np.zeros(adatas[0].n_vars, dtype=np.float64)
        t_count, b_count = 0, 0
        for a in adatas:
            tm = a.obs["is_tissue"].values
            bm = a.obs["is_background"].values
            t_sums += _col_means_masked(a.X, tm) * tm.sum()
            b_sums += _col_means_masked(a.X, bm) * bm.sum()
            t_count += tm.sum()
            b_count += bm.sum()
        mean_t = (t_sums / max(t_count, 1)).astype(np.float32)
        mean_b = (b_sums / max(b_count, 1)).astype(np.float32)
        fc = mean_t / (mean_b + pseudo)
        keep = fc >= cutoff
        for adata in adatas:
            tissue_adata = adata[adata.obs["is_tissue"].values, :][:, keep].copy()
            filtered_tissues.append(tissue_adata)
            qc_stats.append({
                "mean_tissue": mean_t, "mean_bg": mean_b, "fold_change": fc,
                "keep_mask": keep, "cutoff": cutoff, "n_kept": int(keep.sum()),
                "n_removed": int((~keep).sum()),
            })
    else:
        for adata in adatas:
            tm = adata.obs["is_tissue"].values
            bm = adata.obs["is_background"].values
            mean_t = _col_means_masked(adata.X, tm)
            mean_b = _col_means_masked(adata.X, bm)
            fc = mean_t / (mean_b + pseudo)
            keep = fc >= cutoff
            tissue_adata = adata[tm, :][:, keep].copy()
            filtered_tissues.append(tissue_adata)
            qc_stats.append({
                "mean_tissue": mean_t, "mean_bg": mean_b, "fold_change": fc,
                "keep_mask": keep, "cutoff": cutoff, "n_kept": int(keep.sum()),
                "n_removed": int((~keep).sum()),
            })
    return filtered_tissues, qc_stats

# ---------------------------------------------------------------------------
# Normalization & transformation
# ---------------------------------------------------------------------------

def tic_normalize(
    adata: ad.AnnData,
    target_sum: Optional[float] = None,
    copy: bool = False
) -> ad.AnnData:
    """
    Total Ion Current (TIC) normalization.
    Scales row sums to the median TIC (or a target_sum) to preserve log1p effectiveness.

    Note: TIC normalization assumes total ion signal is constant across
    pixels/samples, which is often violated in tissue with heterogeneous
    ion suppression (Cairns et al. 2007; Wulff et al. 2018 found TIC
    underperforms median-based normalization in benchmark comparisons).
    Consider :func:`median_normalize` as a more robust default, especially
    when comparing tissue regions with very different metabolic activity.
    """
    if copy: adata = adata.copy()

    if issparse(adata.X):
        X = adata.X.tocsr(copy=False).astype(np.float32, copy=False)
        row_sums = np.asarray(X.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        target = target_sum if target_sum is not None else np.median(row_sums)
        X = X.multiply((target / row_sums).astype(np.float32)[:, None]).tocsr()
        adata.X = X
    else:
        X = _to_dense_writable(adata.X)
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        target = target_sum if target_sum is not None else np.median(row_sums)
        np.divide(X, row_sums / target, out=X)
        adata.X = X

    _record_step(adata, "tic_normalize")
    return adata

def median_normalize(
    adata: ad.AnnData,
    target_value: Optional[float] = None,
    copy: bool = False,
) -> ad.AnnData:
    """
    Median-intensity normalization.

    Scales each pixel by the median of its own non-zero ion intensities
    (rather than the sum, as in :func:`tic_normalize`). Benchmark studies
    in LC-MS/MSI metabolomics (Wulff et al. 2018; De Graeve et al. 2023)
    found median-based normalization more robust than TIC to a small
    number of very high-intensity ions dominating the pixel total, which
    is common in lipid-rich MSI data.

    Parameters
    ----------
    target_value : float, optional
        Value each pixel's median is scaled to. Defaults to the median
        of per-pixel medians across the dataset.
    """
    if copy: adata = adata.copy()
    X = _to_dense_writable(adata.X)

    row_medians = np.empty(X.shape[0], dtype=np.float32)
    for i in range(X.shape[0]):
        nonzero = X[i][X[i] > 0]
        row_medians[i] = np.median(nonzero) if nonzero.size else 0.0
    row_medians[row_medians == 0] = 1.0

    target = target_value if target_value is not None else float(np.median(row_medians))
    np.divide(X, (row_medians / target)[:, None], out=X)
    adata.X = X

    _record_step(adata, "median_normalize")
    return adata

def log1p_transform(adata: ad.AnnData, copy: bool = False) -> ad.AnnData:
    """
    Replace ``adata.X`` with ``log(1 + x)``.

    Run this after normalisation. Intensities from imaging MS span several
    orders of magnitude, and on the raw scale a handful of bright pixels
    decide every distance and every principal component.

    Sparse matrices are transformed in place on their stored values, so zeros
    stay zero and the matrix stays sparse.

    Parameters
    ----------
    adata : anndata.AnnData
        Normalised data.
    copy : bool, optional
        Return a copy instead of transforming in place. Default False.

    Returns
    -------
    anndata.AnnData
        Log-transformed data, with the step appended to
        ``adata.uns['mortis_steps']``.
    """
    if copy: adata = adata.copy()
    if issparse(adata.X):
        X = adata.X.tocsr(copy=True).astype(np.float32, copy=False)
        np.log1p(X.data, out=X.data)
        adata.X = X
    else:
        X = _to_dense_writable(adata.X)
        np.log1p(X, out=X)
        adata.X = X
    _record_step(adata, "log1p")
    return adata

def scale(adata: ad.AnnData, max_value: Optional[float] = 10.0, copy: bool = False, **kwargs) -> ad.AnnData:
    """
    Centre each metabolite to zero mean and unit variance.

    The log-transformed matrix is kept in ``adata.layers['log1p']`` first,
    because scaled values are the right input to PCA but the wrong input to
    any test of abundance. Statistics in MORTIS read that layer.

    Parameters
    ----------
    adata : anndata.AnnData
        Log-transformed data.
    max_value : float or None, optional
        Clip scaled values at this many standard deviations. Default 10.0.
        None leaves them unclipped.
    copy : bool, optional
        Return a copy instead of scaling in place. Default False.
    **kwargs
        Passed through to ``scanpy.pp.scale``.

    Returns
    -------
    anndata.AnnData
        Scaled data, with the unscaled matrix in ``adata.layers['log1p']``.
    """
    if copy: adata = adata.copy()
    adata.layers["log1p"] = _to_dense(adata.X).copy()
    sc.pp.scale(adata, max_value=max_value, **kwargs)
    _record_step(adata, "scale")
    return adata

# ---------------------------------------------------------------------------
# Dimensionality reduction & Batch Correction
# ---------------------------------------------------------------------------

def run_pca(
    adata: ad.AnnData,
    n_comps: int = 50,
    random_state: int = 0,
    use_hardware: bool = True,
    copy: bool = False,
    **kwargs
) -> ad.AnnData:
    """Principal component analysis. Uses cuML if it is installed, scikit-learn otherwise."""
    if adata.n_vars < 2:
        raise InvalidParameterError(
            f"PCA needs at least 2 metabolites to have an axis to rotate, and "
            f"this object has {adata.n_vars}. If a filtering step ran earlier, "
            f"it was probably stricter than intended."
        )
    if copy: adata = adata.copy()

    n_comps = min(n_comps, min(adata.n_obs, adata.n_vars) - 1)
    backend = _get_hardware_backend(use_hardware)

    if backend == "cuda":
        import cuml
        import cupy as cp
        print("[MORTIS] Running PCA on GPU (cuML).")
        X_gpu = cp.asarray(_to_dense(adata.X))
        pca = cuml.PCA(n_components=n_comps, random_state=random_state, **kwargs)
        adata.obsm['X_pca'] = pca.fit_transform(X_gpu).get()
    else:
        print("[MORTIS] Running PCA on CPU.")
        # arpack below ~10k pixels, randomized above
        kwargs.setdefault("svd_solver", "arpack" if adata.n_obs < 10_000 else "randomized")
        with _configure_runtime_threads():
            sc.tl.pca(adata, n_comps=n_comps, random_state=random_state, **kwargs)

    return adata

def correct_batches(
    adata: ad.AnnData,
    batch_key: str = "sample",
    covariates: Optional[List[str]] = None,
    recompute_pca: bool = True,
    use_hardware: bool = True,
    copy: bool = False,
    **kwargs
) -> ad.AnnData:
    """
    Batch correction with ComBat.

    Covariates you ask to protect are dropped if they only take one value in
    the data, since ComBat's design matrix would be singular and it would
    crash.

    If ComBat itself fails, this raises rather than falling back to a simpler
    correction. Substituting a method quietly would hand back data corrected
    by something other than what you asked for, and other than what your
    methods section is going to say.
    """
    if batch_key not in adata.obs:
        raise InvalidParameterError(
            f"There is no column called {batch_key!r} in adata.obs, so ComBat has "
            f"no batches to correct. Columns present: "
            f"{listing(adata.obs.columns)}.{suggest(batch_key, adata.obs.columns)}"
        )
    if copy: adata = adata.copy()

    valid_covariates = []
    if covariates:
        for cov in covariates:
            if cov in adata.obs.columns:
                if adata.obs[cov].nunique() > 1:
                    valid_covariates.append(cov)
                else:
                    warnings.warn(
                        f"Covariate {cov!r} takes the same value in every pixel, so "
                        "there is nothing for ComBat to protect and its design "
                        "matrix would be singular. Running without it.",
                        UserWarning, stacklevel=2,
                    )
            else:
                warnings.warn(
                    f"Covariate {cov!r} is not a column in adata.obs, so it cannot be "
                    f"protected. Columns present: {listing(adata.obs.columns)}."
                    f"{suggest(cov, adata.obs.columns)}",
                    UserWarning, stacklevel=2,
                )

    protecting = f"protecting {valid_covariates}" if valid_covariates else "no covariates protected"
    print(f"[MORTIS] ComBat on '{batch_key}' ({protecting}).")

    def _apply_combat(X_mat, obs_df):
        import scanpy as sc
        tmp = ad.AnnData(X=X_mat, obs=obs_df)
        try:
            sc.pp.combat(tmp, key=batch_key, covariates=valid_covariates or None, **kwargs)
        except Exception as exc:
            counts = obs_df[batch_key].value_counts()
            raise MortisError(
                f"ComBat could not correct '{batch_key}' and MORTIS will not "
                "silently substitute a weaker method for it.\n"
                f"Original error: {exc}\n"
                f"Batch sizes: {counts.to_dict()}.\n"
                "The usual causes are a batch with too few pixels to estimate a "
                "variance from, a covariate that is collinear with the batch, or "
                "a metabolite that is constant inside one batch. Drop the tiny "
                "batches, or use mortis.run_harmony() instead, which tolerates "
                "unbalanced designs."
            ) from exc
        return tmp.X

    if "log1p" in adata.layers:
        temp_X = _to_dense(adata.X)
        adata.X = _to_dense(adata.layers["log1p"])
        adata.X = _apply_combat(adata.X, adata.obs)
        adata.layers["log1p"] = adata.X.copy()
        adata.X = temp_X
    else:
        adata.X = _to_dense(adata.X)
        adata.X = _apply_combat(adata.X, adata.obs)

    if recompute_pca:
        print("[MORTIS] Recomputing PCA on the corrected data.")
        n_comps = adata.obsm['X_pca'].shape[1] if 'X_pca' in adata.obsm else 50
        run_pca(adata, n_comps=n_comps, use_hardware=use_hardware)

    return adata

def run_harmony(
    adata: ad.AnnData,
    batch_key: str = "sample",
    adjusted_basis: str = "X_pca_harmony",
    copy: bool = False,
    **kwargs
) -> ad.AnnData:
    """
    Batch-correct the PCA embedding using Harmony (Korsunsky et al. 2019,
    Nat Methods).

    Unlike :func:`correct_batches` (ComBat, which corrects the expression
    matrix itself), Harmony corrects the low-dimensional PCA embedding and
    is generally faster and more robust for large numbers of batches/samples.
    The corrected embedding is stored separately (default
    ``adata.obsm['X_pca_harmony']``) rather than overwriting ``X_pca``, so
    downstream steps must be pointed at it explicitly, e.g.::

        adata = mortis.run_harmony(adata, batch_key="sample")
        adata = mortis.run_neighbors(adata, use_rep="X_pca_harmony")

    Parameters
    ----------
    batch_key : str
        Column in ``adata.obs`` identifying the batch/sample to correct for.
    adjusted_basis : str
        Key under which the corrected embedding is stored in ``adata.obsm``.

    Raises
    ------
    NoEmbeddingError
        If ``adata.obsm['X_pca']`` has not been computed yet.
    InvalidParameterError
        If ``batch_key`` is not found in ``adata.obs``.
    """
    if "X_pca" not in adata.obsm:
        raise NoEmbeddingError("PCA embedding not found. Run mortis.run_pca(adata) first.")
    if batch_key not in adata.obs.columns:
        raise InvalidParameterError(f"'{batch_key}' not found in adata.obs.")
    if copy: adata = adata.copy()

    try:
        import harmonypy
    except ImportError:
        raise ImportError(
            "harmonypy is required for run_harmony. Install it with: pip install harmonypy"
        )

    print(f"[MORTIS] Running Harmony batch correction on '{batch_key}'")
    ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, [batch_key], **kwargs)
    Z = np.asarray(ho.Z_corr)
    # harmonypy's Z_corr orientation varies by version and backend
    if Z.shape[0] != adata.n_obs:
        Z = Z.T
    adata.obsm[adjusted_basis] = Z.astype(np.float32)
    return adata

def run_neighbors(
    adata: ad.AnnData,
    n_neighbors: int = 30,
    n_pcs: int = 30,
    metric: str = "cosine",
    random_state: int = 0,
    copy: bool = False,
    **kwargs
) -> ad.AnnData:
    """Compute nearest-neighbor graph, default to cosine metric for MSI."""
    if "X_pca" not in adata.obsm:
        raise NoEmbeddingError("PCA embedding not found. Run mortis.run_pca(adata) first.")
    if copy: adata = adata.copy()
    n_pcs = min(n_pcs, adata.obsm["X_pca"].shape[1])

    print(f"[MORTIS] Building kNN graph (metric={metric}, n_neighbors={n_neighbors}).")

    use_rep = kwargs.pop("use_rep", "X_pca" if "X_pca" in adata.obsm else None)

    with _configure_runtime_threads(), _numba_thread_limit():
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep, metric=metric, random_state=random_state, **kwargs)
    return adata

def run_umap(
    adata: ad.AnnData,
    min_dist: float = 0.3,
    spread: float = 1.0,
    random_state: int = 0,
    use_hardware: bool = True,
    copy: bool = False,
    **kwargs
) -> ad.AnnData:
    """
    Compute a UMAP embedding for visualisation.

    Runs on the Harmony-corrected basis when one is present, otherwise on PCA.
    Use it to look at the data, not to measure it: distances between UMAP
    clusters do not carry a quantitative meaning.

    Parameters
    ----------
    adata : anndata.AnnData
        Data with a neighbour graph from :func:`run_neighbors`.
    min_dist : float, optional
        How tightly points may pack together. Default 0.3. Lower values give
        denser clumps, higher values spread them out.
    spread : float, optional
        Scale of the embedding. Default 1.0.
    random_state : int, optional
        Seed. Default 0.
    use_hardware : bool, optional
        Use a GPU through cuML when one is available. Default True. The GPU
        and CPU paths do not give identical coordinates.
    copy : bool, optional
        Return a copy instead of writing in place. Default False.
    **kwargs
        Passed through to the UMAP implementation.

    Returns
    -------
    anndata.AnnData
        Data with coordinates in ``adata.obsm['X_umap']``.

    Raises
    ------
    NoEmbeddingError
        If no neighbour graph has been computed.
    """
    if "neighbors" not in adata.uns:
        raise NoEmbeddingError("Neighbour graph not found. Run mortis.run_neighbors(adata) first.")
    if copy: adata = adata.copy()

    backend = _get_hardware_backend(use_hardware)
    if backend == "cuda":
        import cuml
        print("[MORTIS] Running UMAP on GPU (cuML).")
        umap_model = cuml.UMAP(min_dist=min_dist, spread=spread, random_state=random_state, **kwargs)
        basis = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else "X_pca"
        adata.obsm['X_umap'] = umap_model.fit_transform(adata.obsm[basis])
    else:
        print("[MORTIS] Running UMAP on CPU.")
        with _configure_runtime_threads(), _numba_thread_limit():
            sc.tl.umap(adata, min_dist=min_dist, spread=spread, random_state=random_state, **kwargs)
    return adata

def preprocess(
    adata: ad.AnnData,
    n_pcs: int = 50,
    n_pcs_neighbors: int = 10,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    do_tic: bool = True,       # Toggle row normalization (TIC or median)
    normalize_method: str = "tic",  # "tic" or "median", see median_normalize()
    do_log1p: bool = True,     # Toggle Log1p
    do_neighbors: bool = True,  # Toggle the kNN graph, see the docstring
    target_sum: Optional[float] = None,
    scale_data: bool = False,
    max_value: float = 10.0,
    random_state: int = 0,
    use_hardware: bool = True,
    copy: bool = False,
    **kwargs
) -> ad.AnnData:
    """
    Full preprocessing pipeline with toggles for already-processed data:
    row normalization -> log1p -> (optional) scale -> PCA -> kNN graph.

    ``do_neighbors=False`` stops before the kNN graph. Clustering and UMAP
    need that graph, but the spatial statistics do not: they build their own
    graph from the pixel coordinates. On a cohort the expression-space graph
    is the most expensive thing here, so a workflow that only wants
    :func:`spatial_organization` or :func:`differential_abundance` should
    skip it.

    normalize_method : {"tic", "median"}, optional
        "tic" (default) divides each pixel by its total ion current.
        "median" divides each pixel by its median non-zero intensity, which
        is more robust to a few very high-intensity ions dominating the
        pixel total (see :func:`median_normalize`). Ignored if
        ``do_tic=False``.
    """
    if copy: adata = adata.copy()

    if do_tic:
        if normalize_method == "median":
            adata = median_normalize(adata, target_value=target_sum)
        elif normalize_method == "tic":
            adata = tic_normalize(adata, target_sum=target_sum)
        else:
            raise InvalidParameterError(
                f"normalize_method must be 'tic' or 'median', got '{normalize_method}'."
            )
    if do_log1p: adata = log1p_transform(adata)
    if scale_data: adata = scale(adata, max_value=max_value)

    adata = run_pca(adata, n_comps=n_pcs, random_state=random_state, use_hardware=use_hardware, **kwargs)
    if do_neighbors:
        adata = run_neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs_neighbors,
                              metric=metric, random_state=random_state)

    return adata

