<h1 align="center">
  <img src="https://raw.githubusercontent.com/FarisHrvat/mortis/main/web/assets/logo.svg" width="46" alt=""><br>
  MORTIS
</h1>

<p align="center">
  <b>Cohort-scale analysis for spatial metabolomics.</b><br>
  Patient-level statistics, differential spatial organization, and figures you can submit.
</p>

<p align="center">
  <a href="https://farishrvat.github.io/mortis/"><b>Read the documentation</b></a>
</p>

<p align="center">
  <a href="https://github.com/FarisHrvat/mortis/actions/workflows/test.yml"><img src="https://github.com/FarisHrvat/mortis/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <img src="https://img.shields.io/badge/tested-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg" alt="Linux, macOS, Windows">
  <a href="https://github.com/FarisHrvat/mortis/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT licence"></a>
  <img src="https://img.shields.io/badge/version-0.1.0-orange.svg" alt="version 0.1.0">
</p>

---

## What it does

Imaging mass spectrometry tells you *where* a metabolite is. Most analyses then
throw that away and ask only *how much*, which is the question bulk
metabolomics already answered, more cheaply.

MORTIS asks both, and asks them at the level where the statistics actually hold:

- **Patient-level testing.** A section has 30,000 pixels and one patient. Test
  the pixels and you inflate your sample size by four orders of magnitude. On
  simulated null data that turns 0 real findings into 183 significant ones.
  `pseudobulk()` comes first here, and it is not optional.
- **Differential spatial *organization*.** A metabolite can sit at identical
  abundance in two groups and be arranged completely differently, diffuse in
  one, pooled into foci in the other. That finding is invisible to every
  abundance test and to bulk metabolomics entirely.
- **Cohorts, not sections.** Compare two drugs, or the same patients before and
  after treatment, and ask whether a signature persists, reorganises, or flips.
- **Figures and receipts.** Vector PDF, SVG and EPS whose text stays editable,
  plus PNG, JPEG and TIFF at whatever DPI the journal asks for. Fonts, sizes,
  colours and DPI are all yours to set. Every figure can carry a sealed
  manifest a reviewer checks your re-run against, without you sending them a
  single byte of patient data.

## What it is not

It is not an acquisition or peak-picking tool. It starts from a feature table
or an `.h5ad`, so extraction, alignment and annotation happen upstream in
SCiLS, METASPACE, Cardinal or the vendor software.

It does not identify compounds. It takes the names your annotation pipeline
gave you, and it cannot tell a confident match from a shaky one beyond the
score your instrument software already wrote.

It is not built for a single section. Most of what it adds is about comparing
groups of patients, and on one section a good deal of it will refuse to run
rather than give you a p-value that counts pixels as replicates.

It does not read raw vendor formats or imzML. It starts from a peak-picked
table: `.csv`, `.tsv`, `.txt`, `.xlsx`, `.parquet`, `.rds` or `.h5ad`. The
delimiter and the decimal mark are worked out from the file, so a
semicolon-and-comma export out of a European Excel reads without editing.
For anything else, read it with whatever library does and hand the frame to
`mt.from_dataframe()`.

## Install

```bash
pip install mortis-spatial
```

Leiden clustering needs two GPL packages, which are not installed by default
because MORTIS is MIT and the choice of pulling GPL code into your environment
should be yours:

```bash
pip install "mortis-spatial[cluster]"
```

Everything else works without them, and `spatial_domains_kmeans()` finds
tissue domains if you would rather not add GPL code at all.

```python
import mortis as mt

adata = mt.preprocess(mt.read_metabolomics_data("section.h5ad"))

# collapse pixels to patients, then test, in that order
pb  = mt.pseudobulk(adata, sample_key="patient")
ab  = mt.differential_abundance(pb, "response", "R", "NR")

# and ask the question only imaging can answer
org = mt.spatial_organization(adata, sample_key="section")
do  = mt.differential_spatial_organization(org, "response", "R", "NR")

mt.compare_abundance_and_organization(ab, do)   # which axis actually moved?
```

The [documentation](https://farishrvat.github.io/mortis/) has the guided tour,
every function with its parameters, the figure gallery, and a validation run on
a public METASPACE study using nothing but this package.

## Or without writing any Python

```bash
mortis template > analysis.yaml     # a commented starting point
mortis run analysis.yaml            # results, figures and a manifest
```

The config that produced a result is a better methods section than one written
from memory, so every run writes it back out beside the results.

**In a container**, when you would rather not install anything:

```bash
docker build -t mortis .
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/work" mortis run /work/analysis.yaml
```

**On a cluster**, [`hpc/`](https://github.com/FarisHrvat/mortis/blob/main/hpc) has Slurm and PBS templates, an Apptainer
definition for sites that will not permit `pip install`, and a conda
environment for the ones that will. The scripts derive thread limits from the
scheduler's allocation and set them before Python starts, which is the
difference between using your cores and oversubscribing a shared node.

## Does it work?

`validation/run_validation.py` downloads a public imaging study, rebuilds the
pixel matrices from the ion images, and runs the whole pipeline, no simulation,
no private data, no other package:

```bash
python validation/run_validation.py
```

24 sections, 250,514 pixels. The positive control (two different plant species)
separates on 4 ions at FDR < 0.05. The deliberately underpowered control returns
nothing, which is the right answer rather than a disappointing one.

## Development

```bash
git clone https://github.com/FarisHrvat/mortis.git && cd mortis
pip install -e ".[dev]"
pytest && ruff check .
```

572 tests, run against Python 3.10, 3.11 and 3.12 on Linux, macOS and
Windows. See
[CONTRIBUTING.md](https://github.com/FarisHrvat/mortis/blob/main/CONTRIBUTING.md).

## Citation

MORTIS is **under review for publication**. A citation will appear here, and in
[`CITATION.cff`](https://github.com/FarisHrvat/mortis/blob/main/CITATION.cff), as soon as the paper is out. Until then, cite the
repository and version.

`filter_drugs()` matches against a drug-name list built from Wikidata, which
is CC0. `run_harmony()` implements Korsunsky et al. 2019, and
`annotate_pathways()` calls MetaboAnalyst and KEGG. Each is linked from the
function's own documentation, and should be cited alongside MORTIS if you use
it.


## Licence

MIT, see [LICENSE](https://github.com/FarisHrvat/mortis/blob/main/LICENSE).

