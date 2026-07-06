# Installation

<span class="mortis-badge beginner">New to Python? Read this whole page first</span>

## 1. Install a Python environment manager

MORTIS depends on a lot of scientific packages (`scanpy`, `numpy`, `scipy`,
`scikit-learn`, ...) that need to be installed together in a matching,
compatible set. The easiest and most reliable way to do that is
[**micromamba**](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html),
a fast, free tool that creates an isolated Python environment just for
this project (so it can never conflict with anything else on your
computer).

=== "macOS / Linux"

    Open **Terminal** and run:

    ```bash
    curl -Ls https://micro.mamba.pm/api/micromamba/osx-64/latest | tar -xvj bin/micromamba
    ```

    Or, if you already have [Homebrew](https://brew.sh):

    ```bash
    brew install micromamba
    ```

=== "Windows"

    Open **PowerShell** and run:

    ```powershell
    Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1 -UseBasicParsing).Content)
    ```

You only need to do this once, ever.

## 2. Create an environment for MORTIS

An "environment" is just an isolated folder containing its own Python and
its own packages, so different projects on your machine never interfere
with each other. Create one now:

```bash
micromamba create -n spatpy_env python=3.11 -c conda-forge -y
```

`-n spatpy_env` names it; you can call it anything. `python=3.11` picks
the Python version (3.9–3.12 all work).

Every time you want to use MORTIS (in a new terminal window), you first
need to **activate** this environment:

```bash
micromamba activate spatpy_env
```

!!! tip "How do I know it worked?"
    Your terminal prompt should now show `(spatpy_env)` at the start of
    the line. If you close the terminal and open a new one, you'll need
    to run `micromamba activate spatpy_env` again — that's normal.

## 3. Install MORTIS

With the environment activated:

```bash
pip install mortis-spatial
```

!!! note "Why `mortis-spatial` but `import mortis`?"
    The package you install (the **distribution name** on PyPI) is
    `mortis-spatial`, but once installed, you `import mortis` in Python
    (the **import name**). This is a common pattern — for example
    `pip install beautifulsoup4` gives you `import bs4`. It happens
    because the short name `mortis` was already taken on PyPI by an
    unrelated package.

That's it. Verify it worked:

```bash
python -c "import mortis; print(mortis.__version__)"
```

If that prints a version number (e.g. `0.5.0`) instead of an error,
you're ready for the **[Quickstart](quickstart.md)**.

## Optional extras

Most users don't need these, but they're available:

```bash
# Faster CSV/XLSX reading (falls back to the default reader automatically
# if not installed, so this is a pure speed optimization)
pip install "mortis-spatial[fast-io]"

# Metabolite colocalization network plots (mt.plot_colocalization_network)
# and non-overlapping volcano-plot labels
pip install "mortis-spatial[image-network]"
```

## Installing from source (for contributors)

```bash
git clone https://github.com/FarisHrvat/mortis.git
cd mortis
pip install -e ".[dev]"
pytest tests/ -v
```

See [Contributing](../contributing.md) for the full development workflow.

## Citation

If you use MORTIS in your research, please cite it — see
[`CITATION.cff`](https://github.com/FarisHrvat/mortis/blob/main/CITATION.cff)
in the repository for the machine-readable citation record. If you use
`mt.filter_drugs()`/`mt.list_drug_matches()`, please also cite DrugBank
(Wishart et al. 2018); if you use `mt.run_harmony()`, please also cite
Korsunsky et al. 2019 — both are linked from the
[API reference](../api/preprocessing.md).
