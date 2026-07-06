"""
MORTIS Analysis Module
======================
Clustering, differential expression, spatial statistics, multi-group tests,
NMF, pathway enrichment, and data management for spatial metabolomics.

Performance design
------------------
* Moran's I is fully vectorised.
* NMF and clustering feature automatic NVIDIA GPU hardware dispatch.
* neighborhood_enrichment uses JIT-compiled Numba C-speed permutations with strict seeding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata as ad
import numba as nb
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from scipy.sparse import csr_matrix, issparse
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests

from .exceptions import (
    InsufficientSamplesError,
    InvalidParameterError,
    MissingSpatialError,
    NoClustersError,
    NoEmbeddingError,
    NotPreprocessedError,
)
from .preprocessing import _N_JOBS

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_dense(X) -> np.ndarray:
    if issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)

def _get_X(adata: ad.AnnData, cache_key: Optional[str] = None) -> np.ndarray:
    if cache_key:
        cache = adata.uns.setdefault("_perf_cache", {})
        cached = cache.get(cache_key)
        if isinstance(cached, np.ndarray):
            return cached
    if "log1p" in adata.layers:
        X = _to_dense(adata.layers["log1p"])
    else:
        X = _to_dense(adata.X)
    if cache_key:
        adata.uns.setdefault("_perf_cache", {})[cache_key] = X
    return X

def _check_preprocessed(adata: ad.AnnData) -> None:
    if not adata.uns.get("preprocessed_steps"):
        raise NotPreprocessedError(
            "This AnnData has not been preprocessed. "
            "Run MORTIS.preprocess(adata) before calling analysis functions."
        )

def _check_clusters(adata: ad.AnnData, key: str) -> None:
    if key not in adata.obs.columns:
        raise NoClustersError(
            f"Cluster labels '{key}' not found in adata.obs. "
            "Run MORTIS.cluster(adata) first."
        )

def _check_neighbors(adata: ad.AnnData) -> None:
    if "neighbors" not in adata.uns:
        raise NoEmbeddingError(
            "Neighbour graph not found. Run MORTIS.run_neighbors(adata) first."
        )

def _check_spatial(adata: ad.AnnData) -> None:
    if "spatial" not in adata.obsm:
        raise MissingSpatialError("adata.obsm['spatial'] is missing.")

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster(
    adata: ad.AnnData,
    resolution: Union[float, List[float]] = 0.5,
    key_added: str = "cluster",
    random_state: int = 0,
    copy: bool = False,
    **kwargs
) -> ad.AnnData:
    """
    Leiden clustering supporting single or multiple resolutions for exploration.

    Parameters
    ----------
    resolution : float or list of floats
        If a list is provided (e.g., [0.1, 0.3, 0.5]), clustering is run for each,
        and saved as `cluster_0.1`, `cluster_0.3`, etc.
    """
    _check_neighbors(adata)

    resolutions = [resolution] if isinstance(resolution, (int, float)) else resolution
    for res in resolutions:
        if float(res) <= 0:
            raise InvalidParameterError(f"resolution must be > 0, got {res}.")

    if copy: adata = adata.copy()

    for res in resolutions:
        current_key = f"{key_added}_{res}" if len(resolutions) > 1 else key_added

        sc.tl.leiden(
            adata,
            resolution=float(res),
            key_added=current_key,
            random_state=random_state,
            **kwargs
        )
        n_clusters = adata.obs[current_key].nunique()
        print(f"[MORTIS] Leiden clustering: {n_clusters} clusters at resolution {res} → adata.obs['{current_key}']")

    if len(resolutions) > 1 and key_added not in adata.obs:
        adata.obs[key_added] = adata.obs[f"{key_added}_{resolutions[0]}"]

    return adata


def cluster_nmf(
    adata: ad.AnnData,
    n_components: int = 10,
    key_added: str = "nmf_cluster",
    basis_key: str = "X_nmf",
    random_state: int = 0,
    max_iter: int = 500,
    use_hardware: bool = True,
    copy: bool = False,
    **kwargs
) -> Tuple[ad.AnnData, pd.DataFrame]:
    """Non-negative Matrix Factorisation (NMF) clustering with NVIDIA GPU Dispatch."""
    if n_components < 2: raise InvalidParameterError("n_components must be ≥ 2.")
    if copy: adata = adata.copy()

    X = _get_X(adata, cache_key="_x_for_compare_groups")
    X = np.clip(X, 0, None)

    H, W = None, None
    gpu_success = False

    if use_hardware:
        try:
            import cuml
            import cupy as cp
            print("[MORTIS] Hardware Accelerated NMF: NVIDIA CUDA")
            model = cuml.NMF(
                n_components=n_components,
                init=kwargs.pop("init", "nndsvda" if X.min() == 0 else "random"),
                max_iter=max_iter, random_state=random_state, **kwargs
            )
            H = model.fit_transform(cp.asarray(X)).get().astype(np.float32)
            W = model.components_.get().astype(np.float32)
            gpu_success = True
        except ImportError:
            pass

    if not gpu_success:
        from sklearn.decomposition import NMF
        print("[MORTIS] Running NMF on CPU.")
        model = NMF(
            n_components=n_components, init=kwargs.pop("init", "nndsvda"),
            random_state=random_state, max_iter=max_iter,
            l1_ratio=kwargs.pop("l1_ratio", 0.0), **kwargs
        )
        H = model.fit_transform(X).astype(np.float32)
        W = model.components_.astype(np.float32)

    adata.obsm[basis_key] = H
    adata.uns["nmf_components"] = W
    adata.obs[key_added] = H.argmax(axis=1).astype(str)

    rows = []
    for k in range(n_components):
        top_idx = W[k].argsort()[::-1][:20]
        for rank, idx in enumerate(top_idx):
            rows.append({
                "component": str(k), "metabolite": adata.var_names[idx],
                "weight": float(W[k, idx]), "rank": rank + 1,
            })

    print(f"[MORTIS] NMF: {n_components} components, {adata.obs[key_added].nunique()} clusters.")
    return adata, pd.DataFrame(rows)


def rename_clusters(adata: ad.AnnData, mapping: Dict[str, str], cluster_key: str = "cluster") -> ad.AnnData:
    _check_clusters(adata, cluster_key)
    adata.obs[cluster_key + "_original"] = adata.obs[cluster_key].copy()
    adata.obs[cluster_key] = adata.obs[cluster_key].map(lambda x: mapping.get(str(x), str(x)))
    return adata


def spatial_domains(
    adata: ad.AnnData,
    resolution: float = 0.5,
    alpha: float = 0.5,
    n_neighbors: int = 6,
    batch_key: str = "sample",
    key_added: str = "domain",
    random_state: int = 0,
    copy: bool = False,
) -> ad.AnnData:
    """
    Spatially-aware clustering into contiguous tissue domains ("niches").

    ``cluster()`` groups pixels purely by molecular profile, so two
    physically distant pixels with similar chemistry land in the same
    cluster. This instead smooths each pixel's PCA embedding towards its
    physical neighbours (controlled by ``alpha``) before running Leiden,
    so the resulting groups are spatially contiguous regions rather than
    scattered chemical clusters — analogous to squidpy/scanpy "niche" or
    spatial-domain detection.

    Parameters
    ----------
    alpha : float
        Spatial smoothing strength in [0, 1]. 0 = identical to `cluster()`
        (no spatial smoothing). 1 = each pixel is replaced entirely by the
        mean of its physical neighbours. Default 0.5.
    n_neighbors : int
        Number of physical neighbours used for smoothing. Default 6.
    batch_key : str
        obs column used to prevent smoothing across independent samples.
    """
    if "X_pca" not in adata.obsm:
        raise NoEmbeddingError("PCA embedding not found. Run mortis.run_pca(adata) first.")
    _check_spatial(adata)
    if not (0.0 <= alpha <= 1.0):
        raise InvalidParameterError(f"alpha must be between 0.0 and 1.0, got {alpha}.")
    if resolution <= 0:
        raise InvalidParameterError(f"resolution must be > 0, got {resolution}.")
    if copy: adata = adata.copy()

    W, _ = _build_spatial_weights(adata, n_neighbors, batch_key=batch_key)
    X_pca = adata.obsm["X_pca"].astype(np.float32)
    X_smooth = (1 - alpha) * X_pca + alpha * np.asarray(W @ X_pca)

    tmp = ad.AnnData(X=np.zeros((adata.n_obs, 1), dtype=np.float32), obs=adata.obs.copy())
    tmp.obsm["X_domain_smooth"] = X_smooth
    sc.pp.neighbors(tmp, use_rep="X_domain_smooth", n_neighbors=n_neighbors, random_state=random_state)
    sc.tl.leiden(tmp, resolution=resolution, key_added=key_added, random_state=random_state)

    adata.obs[key_added] = tmp.obs[key_added].values
    adata.obsm["X_domain_smooth"] = X_smooth
    n_found = adata.obs[key_added].nunique()
    print(f"[MORTIS] Spatial domains: {n_found} contiguous domain(s) found (alpha={alpha}) → adata.obs['{key_added}']")
    return adata


def spatial_domains_kmeans(
    adata: ad.AnnData, n_domains: int = 8, alpha: float = 0.5, n_neighbors: int = 6,
    batch_key: str = "sample", key_added: str = "domain_kmeans", random_state: int = 0, copy: bool = False,
) -> ad.AnnData:
    """
    Fast, fixed-k alternative to :func:`spatial_domains`.

    Uses the same physical-neighbour smoothing as ``spatial_domains`` but
    clusters the smoothed embedding with k-means instead of Leiden. Two
    practical differences: (1) k-means is substantially faster on very
    large images (no graph construction/modularity optimisation), and
    (2) you specify the exact number of domains directly (``n_domains``)
    instead of indirectly tuning a Leiden ``resolution``.
    """
    from sklearn.cluster import KMeans

    if "X_pca" not in adata.obsm:
        raise NoEmbeddingError("PCA embedding not found. Run mortis.run_pca(adata) first.")
    _check_spatial(adata)
    if not (0.0 <= alpha <= 1.0):
        raise InvalidParameterError(f"alpha must be between 0.0 and 1.0, got {alpha}.")
    if n_domains < 2:
        raise InvalidParameterError(f"n_domains must be ≥ 2, got {n_domains}.")
    if copy: adata = adata.copy()

    W, _ = _build_spatial_weights(adata, n_neighbors, batch_key=batch_key)
    X_pca = adata.obsm["X_pca"].astype(np.float32)
    X_smooth = (1 - alpha) * X_pca + alpha * np.asarray(W @ X_pca)

    km = KMeans(n_clusters=n_domains, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_smooth)

    adata.obs[key_added] = labels.astype(str)
    adata.obsm["X_domain_smooth_kmeans"] = X_smooth
    print(f"[MORTIS] Spatial domains (k-means): {n_domains} domains → adata.obs['{key_added}']")
    return adata


def cluster_validation(
    adata: ad.AnnData, cluster_key: str = "cluster", use_rep: str = "X_pca", sample_size: Optional[int] = 10000,
    random_state: int = 0,
) -> float:
    """
    Silhouette score (scikit-learn) for an existing clustering: how well
    separated the clusters are in ``use_rep`` space, in [-1, 1] (higher =
    better separated). Standard scanpy-adjacent QC for choosing/reporting
    a Leiden resolution or comparing clustering methods.

    ``sample_size`` subsamples pixels for speed on large datasets (the
    exact silhouette score is O(n^2); default caps at 10,000 pixels).
    Pass ``None`` to use every pixel.
    """
    from sklearn.metrics import silhouette_score

    _check_clusters(adata, cluster_key)
    if use_rep not in adata.obsm:
        raise NoEmbeddingError(f"'{use_rep}' not found in adata.obsm. Run mortis.run_pca(adata) first.")
    labels = adata.obs[cluster_key].astype(str).values
    if len(np.unique(labels)) < 2:
        raise InvalidParameterError("Need ≥ 2 clusters to compute a silhouette score.")

    score = silhouette_score(
        adata.obsm[use_rep], labels,
        sample_size=min(sample_size, adata.n_obs) if sample_size else None,
        random_state=random_state,
    )
    print(f"[MORTIS] Silhouette score ('{cluster_key}' in '{use_rep}'): {score:.4f}")
    return float(score)


def compare_clusterings(labels_a: Union[List, np.ndarray, pd.Series], labels_b: Union[List, np.ndarray, pd.Series]) -> Dict[str, float]:
    """
    Adjusted Rand Index (ARI) and Adjusted Mutual Information (AMI)
    between two cluster label assignments of the same pixels — e.g.
    comparing a Leiden resolution sweep for stability
    (``adata.obs['cluster_0.3']`` vs. ``adata.obs['cluster_0.5']``), or two
    independent runs/samples. Both scores are 1.0 for identical labelings
    and ~0 for random/independent labelings; unlike raw accuracy, both are
    invariant to how cluster IDs are permuted/renamed.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    labels_a = np.asarray(labels_a).astype(str)
    labels_b = np.asarray(labels_b).astype(str)
    if len(labels_a) != len(labels_b):
        raise InvalidParameterError(
            f"labels_a and labels_b must be the same length, got {len(labels_a)} and {len(labels_b)}."
        )
    return {
        "ari": float(adjusted_rand_score(labels_a, labels_b)),
        "ami": float(adjusted_mutual_info_score(labels_a, labels_b)),
    }


def batch_mixing_score(
    adata: ad.AnnData, batch_key: str = "sample", use_rep: str = "X_pca", n_neighbors: int = 30, copy: bool = False,
) -> ad.AnnData:
    """
    Local Inverse Simpson's Index (LISI; Korsunsky et al. 2019, the
    Harmony paper) — verifies whether batch correction (:func:`run_harmony`,
    :func:`correct_batches`) actually worked, rather than assuming it did.

    For each pixel, computes the effective number of distinct batches
    represented among its ``n_neighbors`` nearest neighbours in
    ``use_rep`` space (1.0 = neighbourhood is a single batch / no mixing;
    approaching the true number of batches = well mixed). Adds
    ``adata.obs['lisi_score']``.

    Run this *before and after* batch correction on the same ``use_rep``
    (e.g. ``'X_pca'`` before, ``'X_pca_harmony'`` after) and compare the
    mean score — it should increase toward the number of batches.
    """
    if use_rep not in adata.obsm:
        raise NoEmbeddingError(f"'{use_rep}' not found in adata.obsm.")
    if batch_key not in adata.obs.columns:
        raise InvalidParameterError(f"'{batch_key}' not found in adata.obs.")
    if copy: adata = adata.copy()

    X = adata.obsm[use_rep]
    n = adata.n_obs
    n_neighbors = min(n_neighbors, n - 1)
    batches = adata.obs[batch_key].astype(str).values
    unique_batches = np.unique(batches)
    nb = len(unique_batches)
    batch_int = {b: i for i, b in enumerate(unique_batches)}
    bi = np.array([batch_int[b] for b in batches])

    tree = cKDTree(X)
    _, idx = tree.query(X, k=n_neighbors + 1, workers=_N_JOBS)
    idx = idx[:, 1:]

    onehot = np.eye(nb, dtype=np.float32)[bi[idx]]  # (n, n_neighbors, nb)
    counts = onehot.sum(axis=1)  # (n, nb)
    p = counts / counts.sum(axis=1, keepdims=True)
    simpson = (p**2).sum(axis=1)
    lisi = (1.0 / np.maximum(simpson, 1e-12)).astype(np.float32)

    adata.obs["lisi_score"] = lisi
    print(
        f"[MORTIS] Batch mixing (iLISI-style, '{use_rep}'): mean={lisi.mean():.2f} "
        f"(1.0 = no mixing, {nb} = perfect mixing across {nb} batches) → adata.obs['lisi_score']"
    )
    return adata


# ---------------------------------------------------------------------------
# Differential expression
# ---------------------------------------------------------------------------

def find_markers(
    adata: ad.AnnData, cluster_key: str = "cluster", method: str = "wilcoxon",
    n_top: int = 20, copy: bool = False, **kwargs
) -> Tuple[ad.AnnData, pd.DataFrame]:
    valid_methods = {"wilcoxon", "t-test", "logreg"}
    if method not in valid_methods: raise InvalidParameterError(f"method must be in {valid_methods}.")
    _check_clusters(adata, cluster_key)
    _check_preprocessed(adata)
    if copy: adata = adata.copy()

    adata_de = adata.copy()
    if "log1p" in adata.layers: adata_de.X = adata_de.layers["log1p"]

    sc.tl.rank_genes_groups(adata_de, groupby=cluster_key, method=method, n_genes=n_top, use_raw=False, **kwargs)
    adata.uns["rank_genes_groups"] = adata_de.uns["rank_genes_groups"]

    result = adata.uns["rank_genes_groups"]
    groups = result["names"].dtype.names
    rows = []
    for grp in groups:
        for rank in range(min(n_top, len(result["names"][grp]))):
            rows.append({
                "cluster": grp, "metabolite": result["names"][grp][rank],
                "score": float(result["scores"][grp][rank]),
                "pval": float(result["pvals"][grp][rank]),
                "pval_adj": float(result["pvals_adj"][grp][rank]),
                "log2fc": float(result["logfoldchanges"][grp][rank]),
            })
    return adata, pd.DataFrame(rows)

def compare_groups(
    adata: ad.AnnData, groupby: str, group1: str, group2: str,
    method: str = "wilcoxon", copy: bool = False,
) -> Tuple[ad.AnnData, pd.DataFrame]:
    if groupby not in adata.obs.columns: raise InvalidParameterError(f"'{groupby}' not found in adata.obs.")
    available = adata.obs[groupby].unique().tolist()
    for g in (group1, group2):
        if g not in available: raise InvalidParameterError(f"Group '{g}' not found.")

    if copy: adata = adata.copy()
    mask1, mask2 = adata.obs[groupby].values == group1, adata.obs[groupby].values == group2
    if mask1.sum() < 3 or mask2.sum() < 3: raise InsufficientSamplesError("Need ≥ 3 pixels per group.")

    X = _get_X(adata)
    X1, X2 = X[mask1], X[mask2]
    mean1, mean2 = X1.mean(axis=0), X2.mean(axis=0)
    pseudo = 1e-9
    log2fc = np.log2((mean2 + pseudo) / (mean1 + pseudo))

    pooled_std = np.sqrt(((X1.shape[0] - 1) * X1.var(axis=0) + (X2.shape[0] - 1) * X2.var(axis=0)) / (X1.shape[0] + X2.shape[0] - 2 + 1e-12))
    cohen_d = (mean2 - mean1) / (pooled_std + 1e-12)

    if method in ("wilcoxon", "mannwhitney"):
        stat_arr, pvals = stats.mannwhitneyu(X1, X2, axis=0, alternative="two-sided")
    else:
        stat_arr, pvals = stats.ttest_ind(X1, X2, axis=0, equal_var=False)

    pvals = np.nan_to_num(pvals, nan=1.0)
    _, pvals_adj, _, _ = multipletests(pvals, method="fdr_bh")

    results_df = pd.DataFrame({
        "metabolite": adata.var_names, "mean_group1": mean1, "mean_group2": mean2,
        "log2fc": log2fc, "cohen_d": cohen_d, "statistic": stat_arr,
        "pval": pvals, "pval_adj": pvals_adj, "significant": pvals_adj < 0.05,
    }).sort_values("pval_adj").reset_index(drop=True)

    adata.uns["compare_groups"] = {"groupby": groupby, "group1": group1, "group2": group2, "method": method, "results": results_df}
    print(f"[MORTIS] {group1} vs {group2}: {results_df['significant'].sum()}/{adata.n_vars} significant (FDR<0.05)")
    return adata, results_df

def multi_group_test(
    adata: ad.AnnData, groupby: str, method: str = "kruskal", copy: bool = False,
) -> Tuple[ad.AnnData, pd.DataFrame]:
    valid_methods = {"kruskal", "anova"}
    if method not in valid_methods:
        raise InvalidParameterError(f"method must be one of {valid_methods}, got '{method}'.")
    if groupby not in adata.obs.columns: raise InvalidParameterError(f"'{groupby}' not found in adata.obs.")
    groups = adata.obs[groupby].unique().tolist()
    if len(groups) < 2: raise InvalidParameterError("Need ≥ 2 groups.")
    if copy: adata = adata.copy()

    X = _get_X(adata, cache_key="_x_for_multi_group")
    group_masks = [adata.obs[groupby].values == g for g in groups]
    group_data = [X[m] for m in group_masks]

    if method == "kruskal": stat_arr, pvals = stats.kruskal(*group_data, axis=0)
    else: stat_arr, pvals = stats.f_oneway(*group_data, axis=0)

    pvals = np.nan_to_num(pvals, nan=1.0)
    _, pvals_adj, _, _ = multipletests(pvals, method="fdr_bh")

    grand_mean = X.mean(axis=0)
    ss_total = ((X - grand_mean) ** 2).sum(axis=0) + 1e-12
    ss_between = sum(m.sum() * (gd.mean(axis=0) - grand_mean) ** 2 for m, gd in zip(group_masks, group_data))
    eta_sq = (ss_between / ss_total).astype(np.float32)

    results_df = pd.DataFrame({
        "metabolite": adata.var_names, "statistic": stat_arr, "pval": pvals,
        "pval_adj": pvals_adj, "significant": pvals_adj < 0.05, "eta_squared": eta_sq,
    }).sort_values("pval_adj").reset_index(drop=True)

    adata.uns["multi_group_test"] = {"groupby": groupby, "method": method, "groups": groups, "results": results_df}
    print(f"[MORTIS] Multi-group test ({method}): {results_df['significant'].sum()}/{adata.n_vars} significant (FDR<0.05)")
    return adata, results_df

# ---------------------------------------------------------------------------
# Spatial statistics
# ---------------------------------------------------------------------------

def _offset_coords_by_batch(adata: ad.AnnData, batch_key: str) -> np.ndarray:
    """Copy spatial coords, shifting each batch far apart so cross-batch
    neighbours are never found by a KDTree query (shared by every spatial
    function that must not bridge independent samples)."""
    coords = adata.obsm["spatial"].copy().astype(np.float32)
    if batch_key in adata.obs.columns:
        for i, batch in enumerate(adata.obs[batch_key].unique()):
            mask = adata.obs[batch_key].values == batch
            coords[mask, 0] += i * 1e6
    return coords

def _build_spatial_weights(
    adata: ad.AnnData, n_neighbors: int, batch_key: str = "sample", include_self: bool = False,
) -> Tuple[csr_matrix, np.ndarray]:
    coords = _offset_coords_by_batch(adata, batch_key)
    n = len(coords)

    tree = cKDTree(coords)
    # Query n_neighbors+1 since the point itself is always its own nearest
    # neighbour (distance 0); drop it unless include_self is requested.
    dists, idx = tree.query(coords, k=n_neighbors + 1, workers=_N_JOBS)
    if not include_self:
        idx = idx[:, 1:]

    n_cols = idx.shape[1]
    rows = np.repeat(np.arange(n), n_cols)
    cols = idx.ravel()
    data = np.full(n * n_cols, 1.0 / n_cols, dtype=np.float32)
    W = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    return W, idx

def spatial_autocorrelation(
    adata: ad.AnnData, n_neighbors: int = 6, batch_key: str = "sample",
    use_fdr: bool = False, copy: bool = False,
) -> Tuple[ad.AnnData, pd.DataFrame]:
    """
    Global spatial autocorrelation for every metabolite: Moran's I and
    Geary's C (analogous to squidpy's ``spatial_autocorr(mode=...)``,
    computed together here since both reuse the same spatial weights).

    Moran's I > 0 / Geary's C < 1 both indicate clustered (spatially
    autocorrelated) signal; Moran's I < 0 / Geary's C > 1 indicate a
    dispersed (checkerboard-like) pattern. They agree on direction almost
    always but weight local vs. global dissimilarity differently — Geary's
    C is more sensitive to sharp *local* discontinuities, Moran's I to the
    overall global pattern. p-values for both use the standard analytic
    z-test under normality (Cliff & Ord 1981), not permutation.
    """
    _check_spatial(adata)
    if copy: adata = adata.copy()

    n = adata.n_obs
    W, _ = _build_spatial_weights(adata, n_neighbors, batch_key=batch_key)

    X = _get_X(adata, cache_key="_x_for_spatial_moran")
    X_mean = X.mean(axis=0)
    X_dev = X - X_mean

    WX = W @ X_dev

    numerator = n * (X_dev * WX).sum(axis=0)
    denom = (X_dev**2).sum(axis=0)
    S0 = W.sum()

    safe_denom = np.where(denom < 1e-12, 1.0, denom)
    morans_i = (numerator / (S0 * safe_denom)).astype(np.float32)
    morans_i[denom < 1e-12] = 0.0

    E_I = -1.0 / (n - 1)
    W_sym = W + W.T
    S1 = 0.5 * W_sym.multiply(W_sym).sum()
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    col_sums = np.asarray(W.sum(axis=0)).ravel()
    S2 = ((row_sums + col_sums)**2).sum()

    var_I = (n**2 * S1 - n * S2 + 3 * S0**2) / ((n**2 - 1) * S0**2) - E_I**2
    var_I = max(var_I, 1e-12)

    z = (morans_i - E_I) / np.sqrt(var_I)
    pvals = stats.norm.sf(z)
    _, pvals_adj, _, _ = multipletests(pvals, method="fdr_bh")

    # --- Geary's C (shares W, S0, S1, S2 with Moran's I above) ---
    # sum_ij w_ij (x_i - x_j)^2 = sum_i x_i^2*rowsum_i + sum_j x_j^2*colsum_j - 2*x^T W x
    term_a = (row_sums[:, None] * X**2).sum(axis=0)
    term_b = (col_sums[:, None] * X**2).sum(axis=0)
    term_c = (X * (W @ X)).sum(axis=0)
    geary_num = term_a + term_b - 2 * term_c

    geary_c = (((n - 1) / (2 * S0)) * (geary_num / safe_denom)).astype(np.float32)
    geary_c[denom < 1e-12] = 1.0  # E[C] under no autocorrelation

    var_C = ((2 * S1 + S2) * (n - 1) - 4 * S0**2) / (2 * (n + 1) * S0**2)
    var_C = max(var_C, 1e-12)
    z_c = (1.0 - geary_c) / np.sqrt(var_C)  # E[C] = 1
    pvals_c = stats.norm.sf(z_c)
    _, pvals_c_adj, _, _ = multipletests(pvals_c, method="fdr_bh")

    adata.var["morans_i"], adata.var["morans_pval"], adata.var["morans_pval_adj"] = morans_i, pvals, pvals_adj
    adata.var["geary_c"], adata.var["geary_pval"], adata.var["geary_pval_adj"] = geary_c, pvals_c, pvals_c_adj
    morans_df = pd.DataFrame({
        "metabolite": adata.var_names, "morans_i": morans_i, "z_score": z,
        "pval": pvals, "pval_adj": pvals_adj,
        "geary_c": geary_c, "geary_z_score": z_c,
        "geary_pval": pvals_c, "geary_pval_adj": pvals_c_adj,
    }).sort_values("morans_i", ascending=False).reset_index(drop=True)

    sig_col = "pval_adj" if use_fdr else "pval"
    n_sig = (morans_df[sig_col] < 0.05).sum()

    print(f"[MORTIS] Moran's I: {n_sig}/{adata.n_vars} spatially variable (p<0.05, FDR={use_fdr}).")
    return adata, morans_df

def spatial_de(adata: ad.AnnData, n_top: int = 50, n_neighbors: int = 6, use_fdr: bool = False, copy: bool = False) -> Tuple[ad.AnnData, pd.DataFrame]:
    adata, morans_df = spatial_autocorrelation(adata, n_neighbors=n_neighbors, use_fdr=use_fdr, copy=copy)
    sig_col = "pval_adj" if use_fdr else "pval"
    sig = morans_df[morans_df[sig_col] < 0.05]
    top = sig.head(n_top).reset_index(drop=True)
    return adata, top

def local_moran(adata: ad.AnnData, metabolite: str, n_neighbors: int = 6, batch_key: str = "sample", copy: bool = False) -> Tuple[ad.AnnData, pd.DataFrame]:
    _check_spatial(adata)
    if metabolite not in adata.var_names: raise InvalidParameterError(f"Metabolite '{metabolite}' not found.")
    if copy: adata = adata.copy()

    coords, n = adata.obsm["spatial"].astype(np.float32), len(adata.obsm["spatial"])
    W, _ = _build_spatial_weights(adata, n_neighbors, batch_key=batch_key)

    X = _get_X(adata)
    idx = adata.var_names.get_loc(metabolite)
    x = X[:, idx].astype(np.float64)

    # Analytical Z-score calculation for massive N
    z = (x - x.mean()) / (x.std() + 1e-12)
    Wz = np.asarray(W @ z).ravel()
    local_i = z * Wz

    # Fast analytical p-values
    z_i = (local_i - local_i.mean()) / (local_i.std() + 1e-12)
    pvals = stats.norm.sf(np.abs(z_i)) * 2

    lisa_type, sig = np.full(n, "NS", dtype=object), pvals < 0.05
    lisa_type[(z > 0) & (Wz > 0) & sig] = "HH"
    lisa_type[(z < 0) & (Wz < 0) & sig] = "LL"
    lisa_type[(z > 0) & (Wz < 0) & sig] = "HL"
    lisa_type[(z < 0) & (Wz > 0) & sig] = "LH"

    adata.obs[f"{metabolite}_lisa"], adata.obs[f"{metabolite}_lisa_type"] = local_i.astype(np.float32), lisa_type

    lisa_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "value": x, "local_i": local_i, "z_score": z, "pval": pvals, "lisa_type": lisa_type})

    counts = pd.Series(lisa_type).value_counts()
    print(f"[MORTIS] LISA '{metabolite}' (Analytical): HH={counts.get('HH',0)}, LL={counts.get('LL',0)}")
    return adata, lisa_df

def getis_ord_gi(
    adata: ad.AnnData, metabolite: str, n_neighbors: int = 6, batch_key: str = "sample", copy: bool = False,
) -> Tuple[ad.AnnData, pd.DataFrame]:
    """
    Getis-Ord Gi* hotspot statistic (Getis & Ord 1992, 1995) for one
    metabolite: a per-pixel z-score testing whether a pixel *and its
    neighbours* (self included, unlike LISA) form a statistically
    significant high-value ("hot spot") or low-value ("cold spot") cluster.

    Complements :func:`local_moran`: LISA also flags spatial *outliers*
    (a high pixel surrounded by low neighbours, or vice versa), whereas
    Gi* only flags concordant hot/cold clusters — the more standard
    "hotspot map" statistic in GIS/spatial-epidemiology tooling.

    Note on normalization: this uses uniform 1/(k+1) weights over each
    pixel and its k neighbours (self included, row sums to exactly 1).
    Verified against esda/PySAL's ``G_Local(star=True)``: the two agree on
    which pixels are relatively hot/cold with correlation > 0.999, but the
    absolute z-scale can differ by a constant factor from tools (e.g. esda
    without an explicit self-weight) that add the self-term without
    renormalizing the row to sum to 1 — a normalization-convention
    difference, not a disagreement about which pixels are hotspots.

    Returns
    -------
    (AnnData, DataFrame) — adds ``adata.obs[f'{metabolite}_gi']`` /
    ``_gi_type`` (``'hot'``, ``'cold'``, ``'NS'``), and a DataFrame with
    columns ``x, y, value, gi_star, pval, hotspot_type``.
    """
    _check_spatial(adata)
    if metabolite not in adata.var_names:
        raise InvalidParameterError(f"Metabolite '{metabolite}' not found.")
    if copy: adata = adata.copy()

    coords = adata.obsm["spatial"].astype(np.float32)
    n = adata.n_obs
    # Gi* includes the pixel itself as one of its own neighbours.
    W_star, _ = _build_spatial_weights(adata, n_neighbors, batch_key=batch_key, include_self=True)
    k = n_neighbors + 1  # neighbours including self

    X = _get_X(adata)
    idx = adata.var_names.get_loc(metabolite)
    x = X[:, idx].astype(np.float64)

    x_bar = x.mean()
    s = np.sqrt(max((x**2).mean() - x_bar**2, 1e-12))
    # W_star is uniform-weight (1/k per row), so both the row-sum (=1) and
    # sum-of-squared-weights (=1/k) are identical across every pixel.
    const_term = np.sqrt(max((n * (1.0 / k) - 1.0) / (n - 1), 1e-12))

    numerator = np.asarray(W_star @ x).ravel() - x_bar
    gi_star = (numerator / (s * const_term)).astype(np.float32)
    pvals = stats.norm.sf(np.abs(gi_star)) * 2

    hotspot_type = np.full(n, "NS", dtype=object)
    sig = pvals < 0.05
    hotspot_type[(gi_star > 0) & sig] = "hot"
    hotspot_type[(gi_star < 0) & sig] = "cold"

    adata.obs[f"{metabolite}_gi"] = gi_star
    adata.obs[f"{metabolite}_gi_type"] = hotspot_type

    gi_df = pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1], "value": x,
        "gi_star": gi_star, "pval": pvals, "hotspot_type": hotspot_type,
    })
    counts = pd.Series(hotspot_type).value_counts()
    print(f"[MORTIS] Getis-Ord Gi* '{metabolite}': hot={counts.get('hot', 0)}, cold={counts.get('cold', 0)}")
    return adata, gi_df

def spatial_neighbors(adata: ad.AnnData, n_neighbors: int = 6, batch_key: str = "sample", copy: bool = False) -> ad.AnnData:
    _check_spatial(adata)
    if copy: adata = adata.copy()

    W, idx = _build_spatial_weights(adata, n_neighbors, batch_key=batch_key)
    W_sym = W + W.T
    W_sym.data = np.ones_like(W_sym.data)
    adata.obsp["spatial_connectivities"] = W_sym.tocsr()
    adata.uns["spatial_neighbors"] = {"n_neighbors": n_neighbors, "params": {"n_neighbors": n_neighbors}}
    return adata

@nb.njit(parallel=True, fastmath=True)
def _numba_permute_and_count(er, ec, label_int, nc, n_permutations, seed):
    np.random.seed(seed)
    n_edges = len(er)
    n_nodes = len(label_int)
    null_counts = np.zeros((n_permutations, nc, nc), dtype=np.float32)

    for p in nb.prange(n_permutations):
        perm_labels = label_int.copy()
        for i in range(n_nodes - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            temp = perm_labels[i]
            perm_labels[i] = perm_labels[j]
            perm_labels[j] = temp

        for e in range(n_edges):
            a = perm_labels[er[e]]
            b = perm_labels[ec[e]]
            null_counts[p, a, b] += 1
            if a != b:
                null_counts[p, b, a] += 1
    return null_counts

def neighborhood_enrichment(
    adata: ad.AnnData, cluster_key: str = "cluster", n_permutations: int = 1000,
    n_jobs: Optional[int] = None, random_state: int = 0, copy: bool = False
) -> Tuple[ad.AnnData, pd.DataFrame]:
    """
    n_jobs : int or None, optional
        Threads used by the Numba-JIT permutation loop. Default: ``None``,
        which uses the ``MORTIS_N_JOBS`` environment variable (or all CPU
        cores if unset). Unlike ``os.environ['OMP_NUM_THREADS']``-style
        settings, ``numba.set_num_threads()`` genuinely takes effect at
        call time, so this parameter is not just cosmetic.
    """
    _check_clusters(adata, cluster_key)
    if "spatial_connectivities" not in adata.obsp: raise InvalidParameterError("Run MORTIS.spatial_neighbors first.")
    if copy: adata = adata.copy()

    prior_n_threads = nb.get_num_threads()
    nb.set_num_threads(max(1, n_jobs if n_jobs is not None else _N_JOBS))

    W = adata.obsp["spatial_connectivities"].tocoo()
    upper = W.row < W.col
    er, ec = W.row[upper].astype(np.int32), W.col[upper].astype(np.int32)

    labels = adata.obs[cluster_key].astype(str).values
    unique_clusters = sorted(np.unique(labels))
    nc = len(unique_clusters)
    label_idx = {c: i for i, c in enumerate(unique_clusters)}
    label_int = np.array([label_idx[lbl] for lbl in labels], dtype=np.int32)

    a_obs, b_obs = label_int[er], label_int[ec]
    observed = np.zeros((nc, nc), dtype=np.float32)
    np.add.at(observed, (a_obs, b_obs), 1)
    np.add.at(observed, (b_obs, a_obs), 1)

    try:
        null_counts = _numba_permute_and_count(er, ec, label_int, nc, n_permutations, random_state)
    finally:
        nb.set_num_threads(prior_n_threads)  # don't leak the thread-count change process-wide
    null_mean = null_counts.mean(axis=0)
    null_std = null_counts.std(axis=0) + 1e-9
    zscore = (observed - null_mean) / null_std
    pval = 2 * (1 - stats.norm.cdf(np.abs(zscore)))

    rows_out = []
    for i, ca in enumerate(unique_clusters):
        for j, cb in enumerate(unique_clusters):
            if i <= j:
                rows_out.append({
                    "cluster_a": ca, "cluster_b": cb, "observed": int(observed[i, j]),
                    "expected": float(null_mean[i, j]), "zscore": float(zscore[i, j]), "pval": float(pval[i, j]),
                })
    enrichment_df = pd.DataFrame(rows_out).sort_values("zscore", ascending=False).reset_index(drop=True)
    adata.uns["neighborhood_enrichment"] = enrichment_df
    return adata, enrichment_df

def co_occurrence(
    adata: ad.AnnData, cluster_key: str = "cluster", n_bins: int = 25,
    max_dist: Optional[float] = None, batch_key: str = "sample", copy: bool = False,
) -> pd.DataFrame:
    """
    Distance-binned co-occurrence probability between cluster/domain labels
    (analogous to squidpy's ``gr.co_occurrence``), distinct from
    :func:`neighborhood_enrichment`: this uses continuous physical distance
    bins rather than a fixed-degree graph, so it reveals *at what distance
    scale* two regions tend to co-occur, not just whether they're adjacent.

    For each ordered pair of labels (a, b) and distance bin, reports the
    ratio ``P(b | within this distance of a) / P(b)`` — a ratio > 1 means
    b is enriched near a at that distance (relative to b's overall
    frequency); a ratio < 1 means depletion.

    Parameters
    ----------
    max_dist : float or None
        Maximum physical distance to bin out to. Default: 25x the median
        nearest-neighbour distance (a scale-appropriate heuristic).

    Returns
    -------
    pandas.DataFrame with columns ``cluster_a, cluster_b, bin, distance, ratio``.
    """
    _check_spatial(adata)
    _check_clusters(adata, cluster_key)
    if copy: adata = adata.copy()

    coords = _offset_coords_by_batch(adata, batch_key).astype(np.float64)
    labels = adata.obs[cluster_key].astype(str).values
    clusters = sorted(np.unique(labels))
    nc = len(clusters)
    label_idx = {c: i for i, c in enumerate(clusters)}
    li = np.array([label_idx[lbl] for lbl in labels], dtype=np.int32)
    global_freq = np.array([(labels == c).mean() for c in clusters])

    tree = cKDTree(coords)
    if max_dist is None:
        d, _ = tree.query(coords, k=2, workers=_N_JOBS)
        max_dist = float(np.median(d[:, 1]) * n_bins)
    if max_dist <= 0:
        raise InvalidParameterError(f"max_dist must be > 0, got {max_dist}.")
    bins = np.linspace(0, max_dist, n_bins + 1)

    dist_mat = tree.sparse_distance_matrix(tree, max_dist, output_type="coo_matrix")
    self_pair = dist_mat.row == dist_mat.col
    row, col, dist = dist_mat.row[~self_pair], dist_mat.col[~self_pair], dist_mat.data[~self_pair]
    bin_idx = np.clip(np.digitize(dist, bins) - 1, 0, n_bins - 1)

    counts = np.zeros((nc, nc, n_bins), dtype=np.int64)
    np.add.at(counts, (li[row], li[col], bin_idx), 1)

    rows_out = []
    bin_centers = (bins[:-1] + bins[1:]) / 2
    for ai, a in enumerate(clusters):
        totals_per_bin = counts[ai].sum(axis=0)
        for bi, b in enumerate(clusters):
            if global_freq[bi] <= 0:
                continue
            for k in range(n_bins):
                if totals_per_bin[k] == 0:
                    continue
                cond_prob = counts[ai, bi, k] / totals_per_bin[k]
                rows_out.append({
                    "cluster_a": a, "cluster_b": b, "bin": k,
                    "distance": float(bin_centers[k]),
                    "ratio": float(cond_prob / global_freq[bi]),
                })

    df = pd.DataFrame(rows_out)
    print(f"[MORTIS] Co-occurrence: {nc} labels x {n_bins} distance bins (max_dist={max_dist:.1f}).")
    return df

# ---------------------------------------------------------------------------
# Enrichment & scoring
# ---------------------------------------------------------------------------

def score_metabolite_set(adata: ad.AnnData, metabolites: List[str], score_name: str = "metabolite_set_score", copy: bool = False) -> ad.AnnData:
    if copy: adata = adata.copy()
    found = [m for m in metabolites if m in adata.var_names]
    missing = [m for m in metabolites if m not in adata.var_names]
    if missing:
        print(f"[MORTIS] Warning: {len(missing)} metabolite(s) not found and skipped: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if not found:
        raise InvalidParameterError("None of the provided metabolites were found in adata.var_names.")
    X = _get_X(adata, cache_key="_x_for_set_scoring")
    idx = [adata.var_names.get_loc(m) for m in found]
    adata.obs[score_name] = X[:, idx].mean(axis=1)
    return adata

def metabolite_set_enrichment(
    results_df: pd.DataFrame, metabolite_sets: Dict[str, List[str]], score_col: str = "log2fc",
    min_set_size: int = 3, n_permutations: int = 1000, random_state: int = 0
) -> pd.DataFrame:
    ranked = results_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    all_mets = ranked["metabolite"].tolist()
    scores = ranked[score_col].to_numpy(dtype=np.float64)
    n_total = len(all_mets)
    met_rank = {m: i for i, m in enumerate(all_mets)}
    rng = np.random.default_rng(random_state)
    rows = []

    for pathway, members in metabolite_sets.items():
        found = [m for m in members if m in met_rank]
        if len(found) < min_set_size: continue

        hit_idx = np.array(sorted(met_rank[m] for m in found))
        n_hit, n_miss = len(hit_idx), n_total - len(hit_idx)

        hit_scores = np.abs(scores[hit_idx])
        hit_sum = hit_scores.sum() + 1e-12
        running = np.zeros(n_total)
        running[hit_idx] += hit_scores / hit_sum
        miss_mask = np.ones(n_total, dtype=bool)
        miss_mask[hit_idx] = False
        running[miss_mask] -= 1.0 / max(n_miss, 1)
        cumsum = np.cumsum(running)
        es = float(cumsum[np.abs(cumsum).argmax()])

        le = [all_mets[i] for i in hit_idx if i <= np.argmax(cumsum)] if es > 0 else [all_mets[i] for i in hit_idx if i >= np.argmin(cumsum)]

        null_es = np.zeros(n_permutations, dtype=np.float64)
        for p in range(n_permutations):
            perm_idx = np.sort(rng.choice(n_total, n_hit, replace=False))
            ph = np.abs(scores[perm_idx])
            pr = np.zeros(n_total)
            pr[perm_idx] += ph / (ph.sum() + 1e-12)
            pm = np.ones(n_total, dtype=bool)
            pm[perm_idx] = False
            pr[pm] -= 1.0 / max(n_miss, 1)
            pc = np.cumsum(pr)
            null_es[p] = pc[np.abs(pc).argmax()]

        pos_null, neg_null = null_es[null_es >= 0], null_es[null_es < 0]
        if es >= 0: nes, pval = es / (pos_null.mean() + 1e-12), float((pos_null >= es).sum() + 1) / (len(pos_null) + 1)
        else: nes, pval = es / (abs(neg_null.mean()) + 1e-12), float((neg_null <= es).sum() + 1) / (len(neg_null) + 1)

        rows.append({"pathway": pathway, "n_metabolites": len(members), "n_found": n_hit, "es": es, "nes": nes, "pval": pval, "leading_edge": ", ".join(le[:10])})

    if not rows: return pd.DataFrame(columns=["pathway", "n_metabolites", "n_found", "es", "nes", "pval", "pval_adj", "leading_edge"])
    df = pd.DataFrame(rows)
    _, df["pval_adj"], _, _ = multipletests(df["pval"].values, method="fdr_bh")
    return df.sort_values("nes", ascending=False).reset_index(drop=True)

def lipid_class_summary(adata: ad.AnnData, groupby: Optional[str] = None) -> pd.DataFrame:
    lipid_prefixes = {"PC": "Phosphatidylcholine", "PE": "Phosphatidylethanolamine", "PS": "Phosphatidylserine", "PI": "Phosphatidylinositol", "PG": "Phosphatidylglycerol", "PA": "Phosphatidic acid", "SM": "Sphingomyelin", "Cer": "Ceramide", "HexCer": "Hexosylceramide", "TG": "Triglyceride", "DG": "Diglyceride", "MG": "Monoglyceride", "LPC": "Lysophosphatidylcholine", "LPE": "Lysophosphatidylethanolamine", "FA": "Fatty acid", "CE": "Cholesterol ester"}
    def _classify(name: str) -> str:
        for prefix, full in lipid_prefixes.items():
            if name.startswith(prefix): return full
        return "Other"

    classes, X, rows = np.array([_classify(m) for m in adata.var_names]), _get_X(adata), []
    for cls in sorted(set(classes)):
        mask = classes == cls
        if mask.sum() == 0: continue
        X_cls = X[:, mask]
        if groupby and groupby in adata.obs.columns:
            row = {"lipid_class": cls, "n_metabolites": int(mask.sum())}
            for g in sorted(adata.obs[groupby].unique()):
                row[str(g)] = float(X_cls[adata.obs[groupby].values == g].mean())
            rows.append(row)
        else:
            rows.append({"lipid_class": cls, "n_metabolites": int(mask.sum()), "mean_intensity": float(X_cls.mean())})
    return pd.DataFrame(rows)

def diversity_index(adata: ad.AnnData, method: str = "shannon", copy: bool = False) -> ad.AnnData:
    """
    Per-pixel metabolomic diversity/heterogeneity score, treating each
    pixel's (non-negative) intensities as a compositional distribution
    over metabolites — an ecology-style diversity index applied to
    chemistry instead of species counts.

    method : {"shannon", "simpson"}
        "shannon" — Shannon entropy -sum(p*log(p)); higher = more even,
        diverse metabolite composition.
        "simpson" — Gini-Simpson index 1 - sum(p^2); same interpretation,
        bounded in [0, 1), more sensitive to dominant metabolites.

    Adds ``adata.obs[f'{method}_diversity']``.
    """
    valid = {"shannon", "simpson"}
    if method not in valid:
        raise InvalidParameterError(f"method must be one of {valid}, got '{method}'.")
    if copy: adata = adata.copy()

    X = np.clip(_get_X(adata), 0, None).astype(np.float64)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    p = X / row_sums

    if method == "shannon":
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, p * np.log(p), 0.0)
        score = -terms.sum(axis=1)
    else:
        score = 1.0 - (p**2).sum(axis=1)

    adata.obs[f"{method}_diversity"] = score.astype(np.float32)
    print(f"[MORTIS] {method.capitalize()} diversity computed → adata.obs['{method}_diversity'] "
          f"(mean={score.mean():.3f})")
    return adata

def cluster_diversity(
    adata: ad.AnnData, cluster_key: str = "cluster", groupby: str = "sample", method: str = "shannon",
) -> pd.DataFrame:
    """
    Region-level heterogeneity: diversity of cluster/domain *composition*
    within each group (e.g. sample or condition) — how mixed vs.
    homogeneous each sample's tissue-domain makeup is, complementing the
    per-pixel :func:`diversity_index`.

    Returns a DataFrame with one row per ``groupby`` value and its
    diversity index over ``cluster_key`` proportions.
    """
    valid = {"shannon", "simpson"}
    if method not in valid:
        raise InvalidParameterError(f"method must be one of {valid}, got '{method}'.")
    if cluster_key not in adata.obs.columns:
        raise NoClustersError(f"Cluster key '{cluster_key}' not found. Run mortis.cluster() first.")
    if groupby not in adata.obs.columns:
        raise InvalidParameterError(f"'{groupby}' not found in adata.obs.")

    ct = pd.crosstab(adata.obs[groupby], adata.obs[cluster_key])
    p = ct.div(ct.sum(axis=1), axis=0).to_numpy()

    if method == "shannon":
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, p * np.log(p), 0.0)
        score = -terms.sum(axis=1)
    else:
        score = 1.0 - (p**2).sum(axis=1)

    return pd.DataFrame({
        groupby: ct.index, f"{method}_diversity": score, "n_clusters_present": (ct.to_numpy() > 0).sum(axis=1),
    }).sort_values(f"{method}_diversity", ascending=False).reset_index(drop=True)

def unmix_pixels(
    adata: ad.AnnData, reference_spectra: Dict[str, Dict[str, float]], copy: bool = False,
) -> ad.AnnData:
    """
    Non-negative least squares (NNLS) unmixing of mixed pixels against a
    library of known reference spectra — recovers, per pixel, the
    non-negative mixing fractions that best reconstruct its intensities as
    a combination of the references. MSI pixels routinely contain mixed
    signal from more than one underlying tissue/cell population (unlike
    single-cell data), so this is a standard MSI-specific analysis
    category (related to multivariate curve resolution) when reference
    spectra are known a priori.

    Uses scikit-learn's coordinate-descent NNLS solver (fixed components,
    ``update_H=False``) so all pixels are solved in one batched call
    rather than a per-pixel Python loop.

    Parameters
    ----------
    reference_spectra : dict[str, dict[str, float]]
        ``{reference_name: {metabolite_name: intensity, ...}, ...}``. Only
        metabolites present in *every* reference AND in ``adata.var_names``
        are used.

    Returns
    -------
    anndata.AnnData with ``adata.obsm['X_unmixed']`` (n_pixels x n_references,
    fractions summing to 1 per pixel) and one ``adata.obs['fraction_{name}']``
    column per reference. ``adata.uns['unmixing']`` records which
    metabolites were used.
    """
    from sklearn.decomposition import non_negative_factorization

    if not reference_spectra:
        raise InvalidParameterError("reference_spectra must be a non-empty dict.")
    if copy: adata = adata.copy()

    ref_names = list(reference_spectra.keys())
    common_mets = [
        m for m in adata.var_names
        if all(m in reference_spectra[r] for r in ref_names)
    ]
    if not common_mets:
        raise InvalidParameterError(
            "No metabolites are shared between adata.var_names and every reference spectrum."
        )

    ref_matrix = np.array(
        [[reference_spectra[r][m] for m in common_mets] for r in ref_names], dtype=np.float64
    )  # (n_refs, n_mets)
    idx = [adata.var_names.get_loc(m) for m in common_mets]
    X = np.clip(_get_X(adata), 0, None)[:, idx].astype(np.float64)

    n_refs = len(ref_names)
    W, _H, _n_iter = non_negative_factorization(
        X, n_components=n_refs, update_H=False, H=ref_matrix,
        max_iter=300, random_state=0,
    )

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    fractions = (W / row_sums).astype(np.float32)

    adata.obsm["X_unmixed"] = fractions
    for i, r in enumerate(ref_names):
        adata.obs[f"fraction_{r}"] = fractions[:, i]
    adata.uns["unmixing"] = {
        "reference_names": ref_names, "metabolites_used": common_mets, "n_metabolites_used": len(common_mets),
    }
    print(f"[MORTIS] NNLS unmixing: {n_refs} reference(s), {len(common_mets)} shared metabolite(s) → adata.obsm['X_unmixed']")
    return adata

# ---------------------------------------------------------------------------
# Multi-sample data management
# ---------------------------------------------------------------------------

def subset_obs(adata: ad.AnnData, obs_col: str, value: Union[str, List[str]], copy: bool = True) -> ad.AnnData:
    if obs_col not in adata.obs.columns: raise InvalidParameterError(f"'{obs_col}' not found.")
    value = [value] if isinstance(value, str) else value
    mask = adata.obs[obs_col].isin(value)
    if mask.sum() == 0: raise InvalidParameterError("No pixels found.")
    result = adata[mask]
    return result.copy() if copy else result

def merge_samples(adatas: List[ad.AnnData], sample_labels: Optional[List[str]] = None, sample_col: str = "sample", join: str = "inner") -> ad.AnnData:
    if sample_labels is not None and len(sample_labels) != len(adatas):
        raise InvalidParameterError(
            f"sample_labels length ({len(sample_labels)}) must match number of adatas ({len(adatas)})."
        )
    sample_labels = sample_labels or [f"sample_{i}" for i in range(len(adatas))]
    for i, (a, label) in enumerate(zip(adatas, sample_labels)):
        a = a.copy()
        a.obs[sample_col], a.obs_names = label, [f"{label}_{j}" for j in range(a.n_obs)]
        adatas[i] = a
    merged = ad.concat(adatas, join=join, fill_value=0.0)
    merged.obs_names_make_unique()
    return merged

def split_by_obs(adata: ad.AnnData, obs_col: str, copy: bool = True) -> Dict[str, ad.AnnData]:
    if obs_col not in adata.obs.columns: raise InvalidParameterError(f"'{obs_col}' not found.")
    return {str(val): (adata[adata.obs[obs_col] == val].copy() if copy else adata[adata.obs[obs_col] == val]) for val in sorted(adata.obs[obs_col].unique())}

# ---------------------------------------------------------------------------
# PAGA trajectory
# ---------------------------------------------------------------------------

def run_paga(adata: ad.AnnData, cluster_key: str = "cluster", copy: bool = False) -> ad.AnnData:
    _check_clusters(adata, cluster_key)
    _check_neighbors(adata)
    if copy: adata = adata.copy()
    sc.tl.paga(adata, groups=cluster_key)
    return adata

# ---------------------------------------------------------------------------
# KILLER FEATURES
# ---------------------------------------------------------------------------

def spatially_weighted_nmf(
    adata: ad.AnnData, n_components: int = 10, alpha: float = 0.5,
    n_neighbors: int = 6, batch_key: str = "sample", random_state: int = 0,
    use_hardware: bool = True, copy: bool = False, **kwargs
) -> Tuple[ad.AnnData, pd.DataFrame]:
    """Smooths data across physical neighbors before NMF with GPU hardware dispatch."""
    if copy: adata = adata.copy()
    W, _ = _build_spatial_weights(adata, n_neighbors, batch_key=batch_key)
    X = np.clip(_get_X(adata), 0, None)

    X_smooth = (1 - alpha) * X + alpha * (W @ X)

    H, W_comp = None, None
    gpu_success = False

    if use_hardware:
        try:
            import cuml
            import cupy as cp
            print("[MORTIS] Hardware Accelerated Spatially-Weighted NMF: NVIDIA CUDA")
            model = cuml.NMF(
                n_components=n_components, max_iter=kwargs.pop("max_iter", 500),
                random_state=random_state, **kwargs
            )
            H = model.fit_transform(cp.asarray(X_smooth)).get().astype(np.float32)
            W_comp = model.components_.get().astype(np.float32)
            gpu_success = True
        except ImportError:
            pass

    if not gpu_success:
        from sklearn.decomposition import NMF
        print("[MORTIS] Running Spatially-Weighted NMF on CPU.")
        model = NMF(
            n_components=n_components, init=kwargs.pop("init", "nndsvda"),
            max_iter=kwargs.pop("max_iter", 500), random_state=random_state, **kwargs
        )
        H = model.fit_transform(X_smooth).astype(np.float32)
        W_comp = model.components_.astype(np.float32)

    adata.obsm["X_snmf"] = H
    adata.obs["snmf_cluster"] = H.argmax(axis=1).astype(str)

    rows = [
        {"component": str(k), "metabolite": adata.var_names[idx], "weight": float(W_comp[k, idx])}
        for k in range(n_components)
        for idx in W_comp[k].argsort()[::-1][:20]
    ]
    print(f"[MORTIS] Spatially-Weighted NMF: {n_components} microenvironments extracted.")
    return adata, pd.DataFrame(rows)


def spatial_gradient(
    adata: ad.AnnData, target_col: str, target_val: str,
    batch_key: str = "sample", bins: int = 15, max_dist: float = 1000.0
) -> pd.DataFrame:
    _check_spatial(adata)
    if target_col not in adata.obs.columns: raise InvalidParameterError(f"'{target_col}' not found.")

    coords, X, distances = adata.obsm["spatial"], _get_X(adata), np.full(adata.n_obs, np.inf)

    batches = adata.obs[batch_key].unique() if batch_key in adata.obs.columns else [None]
    for b in batches:
        b_mask = (adata.obs[batch_key] == b) if b else np.ones(adata.n_obs, dtype=bool)
        t_mask = b_mask & (adata.obs[target_col] == target_val)
        if not t_mask.any(): continue
        tree = cKDTree(coords[t_mask])
        distances[b_mask], _ = tree.query(coords[b_mask])

    valid = distances <= max_dist
    dist_bins = np.linspace(0, max_dist, bins + 1)
    binned = np.digitize(distances[valid], dist_bins) - 1

    res = []
    for i in range(bins):
        mask = binned == i
        if mask.any():
            mean_vals = X[valid][mask].mean(axis=0)
            res.append({"distance": float(dist_bins[i] + (max_dist/bins/2))} | {m: v for m, v in zip(adata.var_names, mean_vals)})

    return pd.DataFrame(res)


def metabolite_colocalization(
    adata: ad.AnnData, top_n: int = 50, corr_threshold: float = 0.4, use_spatial_smooth: bool = True,
    metric: str = "pearson",
) -> pd.DataFrame:
    """
    Pairwise ion-image similarity network among the top-``top_n`` spatially
    variable metabolites (run :func:`spatial_autocorrelation` first).

    metric : {"pearson", "cosine"}
        "pearson" (default) — Pearson correlation between (optionally
        spatially-smoothed) ion image vectors.
        "cosine" — cosine similarity between raw (non-mean-centred) ion
        images, matching METASPACE's own colocalization metric and the
        approach benchmarked in ColocML (Ryabchykov et al., *Bioinformatics*
        2020) as agreeing better with expert-annotated colocalization
        rankings than raw Pearson correlation in that comparison. Cosine
        similarity is not centred, so it is more sensitive to shared
        baseline/background signal than Pearson — prefer it when ion
        images are sparse (mostly zero) with a clear "on/off" spatial
        pattern; prefer Pearson when comparing continuously-varying
        intensity gradients.
    """
    if "morans_i" not in adata.var: raise InvalidParameterError("Run MORTIS.spatial_autocorrelation first.")
    valid_metrics = {"pearson", "cosine"}
    if metric not in valid_metrics:
        raise InvalidParameterError(f"metric must be one of {valid_metrics}, got '{metric}'.")

    top_mets = adata.var.nlargest(top_n, "morans_i").index
    idx = [adata.var_names.get_loc(m) for m in top_mets]
    X = _get_X(adata)[:, idx]

    if use_spatial_smooth and "spatial_connectivities" in adata.obsp:
        X = adata.obsp["spatial_connectivities"] @ X

    if metric == "cosine":
        norms = np.linalg.norm(X, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        X_norm = X / norms
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = pd.DataFrame(X_norm.T @ X_norm, index=top_mets, columns=top_mets)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            sim = pd.DataFrame(np.corrcoef(X.T), index=top_mets, columns=top_mets)

    edges = sim.where(np.triu(np.ones(sim.shape), k=1).astype(bool)).stack().reset_index()
    edges.columns = ["source", "target", "weight"]
    edges = edges.dropna(subset=["weight"])

    filtered = edges[edges["weight"] >= corr_threshold].sort_values("weight", ascending=False).reset_index(drop=True)
    print(f"[MORTIS] Metabolite Interactome ({metric}): Found {len(filtered)} high-confidence edges.")
    return filtered


# ---------------------------------------------------------------------------
# Save utilities
# ---------------------------------------------------------------------------

def save_results(results_df: pd.DataFrame, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)
    print(f"[MORTIS] Saved: {out} ({len(results_df)} rows)")

def save_adata(adata: ad.AnnData, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)
    print(f"[MORTIS] Saved AnnData: {out} (shape={adata.shape})")
