# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Full documentation site (MkDocs Material, deployed via GitHub Actions
  to GitHub Pages): zero-Python-experience installation/quickstart,
  a complete parameter-level API reference for every function, four
  real-data tutorials with real generated plots, a plot gallery, and
  concept pages (normalization, batch correction verification, spatial
  statistics decision guide, performance/threading). Raw patient data
  is never committed — only derived plots (`docs/assets/img/*.png`,
  ~5MB total) and code.
- `.github/workflows/docs.yml` — builds and deploys the docs site on
  every push to `docs/`/`mkdocs.yml`.

### Changed
- `README.md` shrunk from ~600 lines to a short landing page pointing
  at the documentation site, rather than duplicating everything inline.
- Renamed the internal module `mortis/preprocess.py` to `preprocess**ing**.py`.
  It shared its name with the top-level `mortis.preprocess()` pipeline
  function, so `__init__.py`'s `from .preprocess import preprocess`
  silently rebound the `mortis.preprocess` *attribute* from "the
  submodule" to "the function" — meaning `import mortis.preprocess as x`
  gave you the function, not the module, a genuinely confusing Python
  footgun (confirmed by direct testing while investigating an unrelated
  threading issue). Doesn't affect the documented flat `mt.run_pca()`-style
  API at all; only affects anyone importing the submodule directly.
- Renamed `subset_sample()` → `subset_obs()` for consistency with its
  sibling `split_by_obs()` and to match its actual (already generic)
  behavior: despite the name, its first parameter was always a generic
  `obs_col` (any `adata.obs` column — condition, cluster, patient, etc.),
  not specifically "sample". The old name implied narrower behavior than
  the function actually had.

### Fixed
- **`MORTIS_N_JOBS` never actually worked.** `_configure_runtime_threads()`
  set `os.environ['OMP_NUM_THREADS']`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`
  at runtime, deep inside `run_pca`/`run_neighbors`/`run_umap` — but
  OpenBLAS/MKL read those variables once, the first time their thread
  pool initialises (typically at NumPy/SciPy import), so setting them
  this late had **zero measured effect** (verified: `MORTIS_N_JOBS=1` vs
  `=16` gave identical wall time on a 30k x 1952 PCA). Replaced with
  [`threadpoolctl`](https://github.com/joblib/threadpoolctl), which
  controls the already-loaded BLAS library directly and is verified to
  give a real ~3-5x difference between 1 and 16 threads. Also wired up
  `numba.set_num_threads()` (a separate, independently-verified-working
  mechanism) for `neighborhood_enrichment`'s permutation loop (~3-10x,
  confirmed order-independent so not a JIT-warmup artifact) and for
  `run_neighbors`/`run_umap`'s underlying `pynndescent` calls.
- Found and corrected a self-inflicted **confounded benchmark** while
  investigating the above: an intermediate version of this fix claimed
  "10-13x speedup" for `run_neighbors` from Numba thread count, based on
  timing 1-thread-then-16-thread back-to-back in the same process. Testing
  with the order reversed (16 first, then 1) showed *both* orders taking
  ~15-17s on the first call and ~1.3-1.5s on every call after, regardless
  of thread count — the entire effect was Numba JIT-compiling
  `pynndescent`'s kernels once per process, not thread scaling. Documented
  in the README's new "Performance & Threading" section so users aren't
  misled by the same artifact when timing their own one-shot scripts.
- Removed two dead `n_jobs: Optional[int] = None` parameters (from
  `compare_groups` and `multi_group_test`) that were accepted but never
  referenced anywhere in the function body — confirmed via full-codebase
  search that no test, script, or doc ever passed `n_jobs=` to them. Also
  removed the unused `joblib` dependency (never imported anywhere;
  `compare_groups`/`multi_group_test` are already fully vectorised via
  scipy's `axis=0` batch tests, not joblib-parallelised as an old
  benchmark-script comment incorrectly claimed).
- `neighborhood_enrichment`'s `n_jobs` parameter (which *is* meaningful,
  since it has genuine Numba-JIT parallelism) is now actually wired to
  `numba.set_num_threads()` instead of being silently ignored, and
  restores the prior thread count afterward (verified not to leak).
- `filter_drugs()`/`list_drug_matches()` defaulted `db_path="drugbank.db"`,
  a cwd-relative path — the DrugBank database was never actually bundled
  with the installable package, so a fresh `pip install` user had no way
  to use this feature without separately obtaining the file. `drugbank.db`
  is now shipped as package data (`mortis/data/drugbank.db`, declared via
  `[tool.setuptools.package-data]`, verified present inside the built
  wheel) and resolved automatically when `db_path=None` (the new default).
- Two remaining `[SpatPy]`-prefixed print statements (`filter_by_score`,
  `list_drug_matches`) corrected to `[MORTIS]`.

## [0.5.0] - 2026-07-03

This release repairs a mid-refactor break (the package was renamed from
`spatpy` to `mortis` internally, but the test suite, README, and packaging
metadata were never updated to match) and closes several real-data gaps
found while running the package against facility MSI exports.

### Added
- `median_normalize()` — median-intensity row normalization, an alternative
  to `tic_normalize()` shown in the literature to be more robust to a few
  high-intensity ions dominating the pixel total. `preprocess()` gained a
  `normalize_method="tic"|"median"` parameter.
- `spatial_domains()` — spatially-aware clustering ("niche"/domain
  detection): smooths the PCA embedding across each pixel's physical
  neighbours before Leiden clustering, so results are contiguous tissue
  regions rather than scattered chemical clusters.
- `run_harmony()` — Harmony (Korsunsky et al. 2019) batch correction of the
  PCA embedding, complementing the existing ComBat-based `correct_batches()`.
  (`harmonypy` was already a declared dependency but unused until now.)
- `load_annotation_scores()` — merges a separate facility-exported feature
  table (columns like `Compound` / `Identification Score`) into
  `adata.var['score']`, so `filter_by_score()` actually works on raw
  `.xlsx`/`.csv` tissue exports instead of only on pre-scored `.h5ad` files.
- New test coverage for previously-untested public API surface:
  `correct_batches`, `spatially_weighted_nmf`, `spatial_gradient`,
  `metabolite_colocalization`, `spatial_domains`, `median_normalize`,
  `load_annotation_scores`, and the `audit` module.
- `LICENSE`, `CITATION.cff`, `.gitignore`.

### Fixed
- **Test suite was completely uncollectable** (`ModuleNotFoundError:
  No module named 'src.spatpy'`) — all 8 test modules still imported from
  the pre-rename `src.spatpy` package. Fixed to import from `mortis`.
- `spatial_neighbors()` raised `NameError` instead of `MissingSpatialError`
  when spatial coordinates were missing (`MissingSpatialError` was never
  imported at module scope in `analysis.py`).
- `analysis.py` imported `numba` at module scope with no corresponding
  entry in `pyproject.toml` dependencies — any clean install would fail at
  `import mortis`. Added `numba>=0.58.0` to dependencies.
- `read_metabolomics_data()` hard-required the optional `python_calamine`
  package for `.xlsx` files (undeclared dependency, not installed by
  default) — now falls back to `openpyxl` (already a hard dependency) when
  `python_calamine` isn't available. Same fallback added for `pyarrow` on
  CSV reads.
- `cluster()` and `spatial_domains()` now validate `resolution > 0` instead
  of silently passing negative values to Leiden.
- `multi_group_test()` now validates `method` instead of silently falling
  through to ANOVA for unrecognized method names.
- `merge_samples()` now raises `InvalidParameterError` on a
  `sample_labels`/`adatas` length mismatch instead of silently truncating
  via `zip()`.
- `score_metabolite_set()` now warns (matching the convention used
  elsewhere) when some requested metabolites aren't found, instead of
  failing silently.
- Several `InvalidParameterError`/`NoEmbeddingError` messages were
  inconsistent with what they claimed to check (e.g. `run_neighbors()`
  didn't actually verify a PCA embedding existed before building the graph).
- `plot_spatial()`, `plot_umap()`, `plot_embedding_grid()`, and `plot_qc()`
  only recognized the marker-size kwarg as `s=` internally, but the
  benchmark scripts and prior README documented it as `spot_size=` — passing
  `spot_size=` crashed with `PathCollection.set() got an unexpected keyword
  argument 'spot_size'` because it fell through unpopped to
  `ax.scatter(**kwargs)`. Both spellings are now accepted everywhere.
- `run_benchmark.py` (the primary, most up-to-date example/integration
  script) had its UMAP computation step commented out, but Part 5 still
  unconditionally called `plot_umap()`, so the script always crashed before
  reaching most of its plotting section — meaning `results_benchmark/` had
  never actually contained any plot output, only the CSVs from the parts
  that ran before the crash. Re-enabled the UMAP step; the script now runs
  to completion end-to-end against the real `Responder`/`Non Responder`
  cohorts (verified: 24/24 steps, all plots + H5AD + audit receipt written).
- `benchmark_full_pipeline.py` still imported the pre-rename `spatpy`
  package (`import spatpy`, `spatpy.read_metabolomics_data(...)`, etc.) —
  fixed to `import mortis as mt`; verified it now runs end-to-end against
  the bundled `.h5ad`/`.xlsx` example files (single-sample, multi-sample
  with `run_harmony`, and a 30k-pixel scale benchmark).
- Removed dead `core.py` (empty, unreferenced file) and stale
  `spatpy.egg-info/` left over from the pre-rename package name.
- `image.py`'s `load_image`/`align_image`/`extract_image_features`/
  `plot_image_overlay` were never exported from the top-level `mortis`
  namespace (only reachable as `mortis.image.*`), inconsistent with the
  rest of the flat API. Now exported as `mortis.load_image` etc.
- Exception classes (`MissingROIError`, `InvalidParameterError`, ...) were
  not importable from the top-level `mortis` namespace despite the README
  documenting `mortis.MissingROIError`-style usage. Now exported.

### Changed
- PyPI distribution name changed to `mortis-spatial` (the plain name
  `mortis` is already taken on PyPI by an unrelated package). The
  importable module name is unchanged: `import mortis`.
- `pyproject.toml` classifiers, keywords, and optional dependency groups
  (`fast-io`, `image-network`) expanded for clearer install guidance.
- `README.md` rewritten to match the current `mortis` API (it previously
  documented a `spatpy` API — wrong import name, wrong exception names,
  wrong function signatures — left over from before the rename).

## [0.4.0] and earlier

Predates this changelog. See git history / `pypi_upload_guide.md` for
context; feature set at this point included background filtering,
annotation-score and DrugBank-based metabolite filtering, TIC
normalization, PCA/UMAP/Leiden clustering, NMF, marker discovery,
differential expression, Moran's I / LISA spatial statistics, neighborhood
enrichment, metabolite set scoring/enrichment, spatially-weighted NMF,
spatial gradient profiling, metabolite colocalization networks, PAGA, and
histology image overlay.
