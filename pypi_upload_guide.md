# MORTIS: PyPI Upload Guide

This guide explains how to package and publish `mortis-spatial` (the
`mortis` package) to PyPI so that anyone can `pip install mortis-spatial`.

There are two ways to do it: the **recommended** automated path via GitHub
Actions (no secrets to manage), and the **manual** `twine` path (useful for
the very first release, or if you're not using GitHub Actions).

---

## Recommended: automated publish via GitHub Actions (Trusted Publishing)

This repo already ships `.github/workflows/publish.yml`, which builds and
publishes to PyPI whenever you push a tag matching `v*` (e.g. `v0.5.0`),
using PyPI's **Trusted Publishing** (OIDC) — no API tokens or secrets to
create or rotate.

**One-time setup** (before your first release):

1. Push this repository to GitHub.
2. On PyPI, go to **pypi.org → Your account → Publishing** (or, for a brand
   new project name, [pypi.org/manage/account/publishing](https://pypi.org/manage/account/publishing/))
   and register a new "pending publisher" with:
   - PyPI project name: `mortis-spatial`
   - Owner: your GitHub username/org
   - Repository name: this repo's name
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. That's it — no token to copy anywhere.

**Every release after that:**

```bash
# bump the version in pyproject.toml first, then:
git tag v0.5.1
git push origin v0.5.1
```

The `publish.yml` workflow builds the sdist/wheel and publishes them
automatically. Watch the Actions tab for status.

---

## Manual path (twine)

Useful for a one-off release without setting up CI, or for uploading to
TestPyPI first.

### Step 1: Prepare

```bash
pip install build twine
```

Check `pyproject.toml`:
- `name = "mortis-spatial"` (the PyPI distribution name — the import name
  stays `mortis`, e.g. `pip install mortis-spatial` then `import mortis`)
- `version` — must be incremented for every upload (PyPI never lets you
  re-upload the same version, even if you delete the release)
- `dependencies` — kept in sync with actual `import` statements in
  `src/mortis/*.py` (this was a real bug fixed in v0.5.0 — `numba` was
  imported but undeclared)

### Step 2: Build

```bash
rm -rf dist/ build/ src/*.egg-info/
python -m build
```

Produces `dist/mortis_spatial-X.Y.Z-py3-none-any.whl` and
`dist/mortis_spatial-X.Y.Z.tar.gz`. Sanity-check before uploading:

```bash
twine check dist/*
```

### Step 3: Test upload to TestPyPI (recommended)

```bash
python -m twine upload --repository testpypi dist/*
```
- Username: `__token__`
- Password: your TestPyPI API token (starts with `pypi-`)

Verify in a fresh environment:
```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mortis-spatial
python -c "import mortis; print(mortis.__version__)"
```
(`--extra-index-url` is needed because TestPyPI doesn't mirror every
dependency, like `scanpy`/`anndata`.)

### Step 4: Upload to PyPI

```bash
python -m twine upload dist/*
```
- Username: `__token__`
- Password: your PyPI API token

> **Once a version is uploaded to PyPI it can never be replaced or
> deleted-and-reused.** Bump the version and re-upload if you find a bug.

### Step 5: Verify the public release

```bash
pip install mortis-spatial
python -c "import mortis; print(mortis.__version__)"
```

---

## Before your first public release, also consider

- Running the full test + lint suite one more time: `pytest tests/ -v && ruff check .`
- Filling in `CITATION.cff` with your real name/ORCID and the repository URL.
- Updating the CI badge URL at the top of `README.md` once the repo has a
  GitHub remote.
- If you plan to submit to [JOSS](https://joss.theoj.org/) for a citable
  DOI, JOSS expects roughly 6 months of public development history with
  releases/issues before review, plus a `paper.md`/`paper.bib` with a
  "Statement of Need" section — plan the submission date accordingly rather
  than submitting immediately after the first public push.
