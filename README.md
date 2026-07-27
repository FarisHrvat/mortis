<h1 align="center">
  <img src="web/assets/logo.svg" width="46" alt=""><br>
  MORTIS
</h1>

<p align="center">
  <b>Cohort-scale analysis for spatial metabolomics.</b><br>
  Patient-level statistics, differential spatial organization, and figures you can submit.
</p>

<p align="center">
  <a href="https://farishrvat.github.io/mortis/"><b>📖 Read the documentation →</b></a>
</p>

<p align="center">
  <a href="https://github.com/FarisHrvat/mortis/actions/workflows/test.yml"><img src="https://github.com/FarisHrvat/mortis/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <img src="https://img.shields.io/badge/tested-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg" alt="Linux, macOS, Windows">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT licence"></a>
  <img src="https://img.shields.io/badge/version-0.1.0-orange.svg" alt="version 0.1.0">
</p>

---

## What it does

Imaging mass spectrometry tells you *where* a metabolite is. Most analyses then
throw that away and ask only *how much* — which is the question bulk
metabolomics already answered, more cheaply.

MORTIS asks both, and asks them at the level where the statistics actually hold:

- **Patient-level testing.** A section has 30,000 pixels and one patient. Test
  the pixels and you inflate your sample size by four orders of magnitude. On
  simulated null data that turns 0 real findings into 183 significant ones.
  `pseudobulk()` comes first here, and it is not optional.
- **Differential spatial *organization*.** A metabolite can sit at identical
  abundance in two groups and be arranged completely differently — diffuse in
  one, pooled into foci in the other. That finding is invisible to every
  abundance test and to bulk metabolomics entirely.
- **Cohorts, not sections.** Compare two drugs, or the same patients before and
  after treatment, and ask whether a signature persists, reorganises, or flips.
- **Figures and receipts.** Vector PDFs whose text stays editable, plus a sealed
  manifest a reviewer can check your re-run against without you sending them a
  single byte of patient data.

## Install

```bash
pip install mortis-spatial
```

```python
import mortis as mt

adata = mt.preprocess(mt.read_metabolomics_data("section.h5ad"))

# collapse pixels to patients, then test — in that order
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

**On a cluster** — [`hpc/`](hpc/) has Slurm and PBS templates, an Apptainer
definition for sites that will not permit `pip install`, and a conda
environment for the ones that will. The scripts derive thread limits from the
scheduler's allocation and set them before Python starts, which is the
difference between using your cores and oversubscribing a shared node.

## Does it work?

`validation/run_validation.py` downloads a public imaging study, rebuilds the
pixel matrices from the ion images, and runs the whole pipeline — no simulation,
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

520 tests, run against Python 3.10, 3.11 and 3.12 on Linux and macOS. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

MORTIS is **under review for publication**. A citation will appear here, and in
[`CITATION.cff`](CITATION.cff), as soon as the paper is out. Until then, cite the
repository and version.

If you use `filter_drugs()` please also cite DrugBank (Wishart et al. 2018);
`run_harmony()`, Korsunsky et al. 2019; `annotate_pathways()`, MetaboAnalyst and
KEGG. Each is linked from the function's own documentation.


## Licence

MIT — see [LICENSE](LICENSE).
