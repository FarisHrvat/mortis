# Quickstart

<span class="mortis-badge beginner">Assumes zero prior Python experience</span>

This page gets you from a raw instrument export file to your first
figure, explaining every single command as we go. If you haven't
installed MORTIS yet, do that first: [Installation](installation.md).

## Step 0: what you'll need

- A terminal window with your environment activated
  (`micromamba activate spatpy_env` — see [Installation](installation.md)).
- One or more spatial metabolomics files: `.h5ad`, `.csv`, or `.xlsx`
  (with `x`, `y`, and metabolite intensity columns).

## Step 1: start Python and load your data

Type `python` in your terminal and press Enter — you're now inside an
interactive Python session (you'll see a `>>>` prompt).

```python
import mortis as mt
```

This line loads the MORTIS package and gives it the short nickname `mt`
so you don't have to type `mortis` every time. Every command from now on
starts with `mt.`.

```python
adata = mt.read_metabolomics_data("my_sample.xlsx")
```

This reads one file into a variable called `adata` (short for
"annotated data" — the standard object type used throughout MORTIS,
scanpy, and the wider single-cell/spatial Python ecosystem). Think of
`adata` as a smart spreadsheet: rows are pixels, columns are
metabolites, plus extra metadata (spatial coordinates, sample info,
computed results) attached to it as your analysis progresses.

!!! tip "Working with a whole folder instead of one file?"
    If you have multiple samples, and each sample is split into a
    `..._tissue.xlsx` file and a `..._background.xlsx` file (a common
    export format), point MORTIS at the folder instead and it will find
    and pair them up automatically:
    ```python
    adatas = mt.load_from_folder("./my_data_folder")
    ```
    This gives you back a **list** of `adata` objects (one per sample),
    so later steps use `adatas[0]`, `adatas[1]`, etc., or a loop.

Check what you loaded:

```python
print(adata)
```

You'll see something like:
```
AnnData object with n_obs × n_vars = 7943 × 2334
    obs: 'x', 'y'
    obsm: 'spatial'
```
`n_obs` is the number of pixels, `n_vars` is the number of metabolites.

## Step 2: remove background noise

Raw MSI data includes signal from *outside* the tissue (background/matrix
noise). If your file has separate tissue and background regions marked
(via `load_from_folder`'s auto-pairing, or drawn manually — see
[ROI Selection](../api/roi.md)), remove metabolites that aren't
meaningfully above background:

```python
clean, stats = mt.filter_background([adata], cutoff=1.5, mode="sample")
adata = clean[0]
```

`cutoff=1.5` means "keep a metabolite only if it's at least 1.5x
stronger in tissue than in background." Higher = stricter. `stats[0]`
holds the numbers behind this decision — visualize them:

```python
mt.plot_qc(stats[0], adata, sample_name="My Sample", save="qc.pdf")
```

This saves a 4-panel quality-control report to `qc.pdf` in your current
folder. Open it to see exactly what was kept/removed and why.

## Step 3: keep only confidently-identified metabolites

Instrument software (METASPACE, SCiLS, etc.) assigns each metabolite a
confidence score. Keep only the trustworthy ones:

```python
adata = mt.filter_by_score(adata, min_score=0.3)
```

!!! note "Don't have a `score` column yet?"
    Some facility exports keep annotation scores in a *separate* file
    (columns like `Compound`, `Identification Score`). Merge it in
    first:
    ```python
    adata = mt.load_annotation_scores(adata, "feature_table.xlsx")
    adata = mt.filter_by_score(adata, min_score=0.3)
    ```

## Step 4: preprocess (one line does normalization + PCA + graph)

```python
adata = mt.preprocess(adata)
```

Behind this single line: total-ion-current normalization, a
log-transform (compresses very large values so a few super-bright
metabolites don't dominate everything), PCA (compresses thousands of
metabolite columns down to the ~50 dimensions that matter most), and a
nearest-neighbour graph (needed for the next step). See
[Preprocessing](../api/preprocessing.md) if you want to run these steps
individually or change the defaults.

## Step 5: find tissue regions (clustering)

```python
adata = mt.cluster(adata, resolution=0.5)
```

This groups pixels with similar chemistry into numbered clusters
(`"0"`, `"1"`, `"2"`, ...), stored in `adata.obs["cluster"]`. Higher
`resolution` = more, smaller clusters.

## Step 6: make your first figure

```python
mt.plot_spatial(adata, color="cluster", save="my_first_figure.pdf")
```

Open `my_first_figure.pdf` — you should see your tissue shape, colored
by cluster. That's a complete raw-file-to-figure pipeline in 6 steps.

## What's next

- Run the **[Complete Workflow](complete-workflow.md)** to add
  differential expression, marker metabolites, and spatial statistics.
- Browse **[real examples with real plots](../tutorials/index.md)** on
  actual facility data.
- Look up any function's exact parameters in the
  **[API Reference](../api/index.md)**.

!!! tip "Save your work before you close the terminal"
    Nothing is saved automatically. Before you're done for the day:
    ```python
    mt.save_adata(adata, "my_processed_sample.h5ad")
    ```
    Next time, reload it instantly instead of re-running everything:
    ```python
    import anndata as ad
    adata = ad.read_h5ad("my_processed_sample.h5ad")
    ```
