# Performance & Threading

<span class="mortis-badge developer">For anyone tuning MORTIS on a shared server or large dataset</span>

## Setting `MORTIS_N_JOBS`

```bash
MORTIS_N_JOBS=4 python my_analysis.py
```

Set this **before starting Python**, it's read once, at import time.
It genuinely changes wall time for:

- **`mt.run_pca()`**: via [`threadpoolctl`](https://github.com/joblib/threadpoolctl),
  which controls the already-loaded BLAS backend directly. Measured
  ~3-5x difference between 1 and 16 threads on a 30,000-pixel x ~2,000-metabolite
  PCA.
- **`mt.neighborhood_enrichment()`**'s permutation test: via
  `numba.set_num_threads()`. Measured ~3-10x between 1 and 16 threads,
  confirmed **order-independent** (i.e. not a compilation artifact, see
  below for why that check matters).

## Two things worth knowing before you conclude threading "isn't working"

### 1. Environment-variable-based thread limits silently do nothing here

If you've used `os.environ['OMP_NUM_THREADS'] = "4"` (or the MKL/OpenBLAS
equivalents) to control thread count in other tools, you might expect
that to work inside MORTIS too. It doesn't, and this isn't a MORTIS
quirk, it's how OpenBLAS/MKL work: they read that variable **once**,
the first time their internal thread pool initializes (typically at
NumPy/SciPy import, or the first BLAS-using call anywhere in the
process). Setting it later, deep inside a function call, is silently
ignored.

This is exactly why `MORTIS_N_JOBS` is implemented via `threadpoolctl`
(which talks to the already-loaded library directly, bypassing the
env-var-at-import-time limitation) rather than environment variables.
An earlier version of this package's internal thread control *did* use
the environment-variable approach, and it measured a **0x** difference
between `MORTIS_N_JOBS=1` and `MORTIS_N_JOBS=16` on a real 30,000-pixel
PCA, the fix was verified with the same benchmark afterward, showing
the expected ~3-5x.

### 2. The first `run_neighbors`/`run_umap` call in a process is slow for an unrelated reason

Both dispatch to [`pynndescent`](https://pynndescent.readthedocs.io/)
(a Numba-JIT-compiled approximate nearest-neighbour library) for
anything but tiny datasets. On a 30,000-pixel dataset, **the first
call** pays a one-time ~15-17 second Numba compilation cost,
*regardless of thread count*, and **every subsequent call in the same
process** is ~1.3-1.5 seconds.

If you're timing a single one-shot script, don't mistake this for a
performance problem or try to fix it via `MORTIS_N_JOBS`. There is
currently no way to avoid this one-time cost short of keeping a
process alive across multiple calls (a long-running notebook kernel or
service, rather than a fresh `python script.py` invocation each time).

!!! example "How this was actually verified (not just assumed)"
    An intermediate version of the threading fix in this package
    benchmarked "1 thread, then 16 threads" back-to-back in the same
    process and attributed the entire ~12x difference to thread
    scaling. Re-running with the order reversed (16 threads first, then
    1) showed **both orders** taking ~15-17s on the first call and
    ~1.3-1.5s on every call after, proving the effect was JIT
    compilation, not threading. This is a useful general lesson for
    benchmarking anything Numba-based: always test with the run order
    reversed before trusting a "before vs. after" number.

## What's not (yet) covered by `MORTIS_N_JOBS`

- GPU dispatch (`use_hardware=True` on `run_pca`/`run_umap`/`cluster_nmf`/
  `spatially_weighted_nmf`/`run_harmony`), if `cuml`/`cupy` (NVIDIA
  CUDA) are importable, these functions transparently use the GPU
  instead; this is orthogonal to `MORTIS_N_JOBS`, which only applies to
  the CPU path.
- `cKDTree`-based spatial-neighbour queries (used internally by
  `spatial_neighbors`, `spatial_domains`, `getis_ord_gi`, `co_occurrence`,
  `batch_mixing_score`) respect `MORTIS_N_JOBS` for the number of worker
  threads in the query itself.

## Memory tips

- `mt.subset_obs()` to work on one sample/patient/condition at a time
  instead of holding a whole cohort in memory.
- For very large datasets, `adata.X = scipy.sparse.csr_matrix(adata.X)`
  before heavy preprocessing steps.
- Delete intermediate variables you no longer need:
  `del adata_raw; import gc; gc.collect()`.
- For very large `.h5ad` files (>10 GB), consider backed mode:
  `anndata.read_h5ad(path, backed="r")`.

