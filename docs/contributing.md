# Contributing

## Development setup

```bash
git clone https://github.com/FarisHrvat/mortis.git
cd mortis
micromamba create -n spatpy_env python=3.11 -c conda-forge -y
micromamba activate spatpy_env
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
pytest tests/ -v
ruff check .
```

Both must pass — CI runs the same checks (`.github/workflows/test.yml`)
across Python 3.10–3.12 on Ubuntu and macOS.

## What the test suite covers

- **Unit tests** (`tests/test_*.py`) — every public function's normal
  behavior, edge cases, and error conditions.
- **Correctness verification** (`tests/test_correctness_vs_reference.py`) —
  the custom spatial statistics (Moran's I, Geary's C, Getis-Ord Gi*)
  cross-checked against [esda/PySAL](https://pysal.org/esda/), an
  independent published implementation. The "thin wrapper" functions
  (PCA, Leiden, ComBat, Harmony, Wilcoxon/Kruskal + FDR, NMF,
  silhouette/ARI/AMI) verified to produce numerically identical results
  to calling scanpy/sklearn/scipy/statsmodels/harmonypy directly. This
  file requires the optional `esda`/`libpysal` test dependencies
  (installed automatically via `pip install -e ".[dev]"`) and is skipped
  automatically if they're unavailable.

## Building the docs locally

```bash
pip install mkdocs-material mkdocs-glightbox
mkdocs serve
```

Open `http://127.0.0.1:8000`. The site rebuilds automatically as you
edit files under `docs/`.

!!! warning "Never commit raw patient data"
    `.gitignore` already excludes `*.h5ad`, the large example `.xlsx`
    files, and the `Responder`/`Non Responder` folders — these are
    private clinical data and must never be pushed to the repository,
    even though it's currently private. Only **derived** plots/results
    (e.g. `docs/assets/img/*.png`) generated from that data are
    committed.

## Code style

- `ruff` handles linting; there's no separate formatter configuration —
  match the existing terse style (`if x: y` single-line guard clauses
  are used deliberately throughout and are excluded from `E701` in
  `pyproject.toml`).
- Every public function needs a docstring explaining parameters,
  return value, and exceptions raised (see any existing function in
  `src/mortis/` for the expected format).
- New analysis functions should include: a test file entry, an entry in
  `src/mortis/__init__.py`'s imports and `__all__`, and a section in
  the relevant `docs/api/*.md` page.

## Releasing

See [`pypi_upload_guide.md`](https://github.com/FarisHrvat/mortis/blob/main/pypi_upload_guide.md)
in the repository root for the full release process (GitHub Actions
trusted-publishing workflow, or the manual `twine` fallback).
