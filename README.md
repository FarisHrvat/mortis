# MORTIS

**High-performance downstream analysis for spatial metabolomics (Imaging Mass Spectrometry)**

[![Tests](https://github.com/FarisHrvat/mortis/actions/workflows/test.yml/badge.svg)](https://github.com/FarisHrvat/mortis/actions/workflows/test.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> This repository is currently private. The Actions badge above only
> renders for people with repo access until it's made public.

MORTIS is an end-to-end Python package for downstream analysis of
spatial metabolomics (MALDI-MSI, DESI, and similar imaging mass
spectrometry) data — from raw instrument export to publication-ready
figures, built on `AnnData`/`scanpy`.

## :book: Full documentation

**[farishrvat.github.io/mortis](https://farishrvat.github.io/mortis/)**

Everything lives there, in far more depth than a README can hold:
installation for readers with zero Python experience, a complete
parameter-by-parameter API reference for every function, four
real-data tutorials (with real generated plots — the raw patient data
itself is never published, only derived plots/results), a full plot
gallery, and the reasoning behind every non-obvious design decision
(normalization choices, batch-correction verification, thread-tuning
gotchas, etc.).

## Install

```bash
pip install mortis-spatial
```

```python
import mortis as mt

adatas = mt.load_from_folder("./data")
clean, stats = mt.filter_background(adatas, cutoff=1.5)
adata = mt.preprocess(mt.filter_by_score(clean[0], min_score=0.3))
adata = mt.cluster(adata, resolution=0.5)
mt.plot_spatial(adata, color="cluster", save="clusters.pdf")
```

See the [full documentation](https://farishrvat.github.io/mortis/) for
everything else — installation details, the complete API, tutorials,
and the plot gallery.

## Development

```bash
git clone https://github.com/FarisHrvat/mortis.git && cd mortis
pip install -e ".[dev]"
pytest tests/ -v && ruff check .
```

See [Contributing](https://farishrvat.github.io/mortis/contributing/)
for the full workflow, including how the docs site itself is built.

## Citation

See [`CITATION.cff`](CITATION.cff). If you use `filter_drugs()`/
`list_drug_matches()`, please also cite DrugBank (Wishart et al. 2018);
if you use `run_harmony()`, please also cite Korsunsky et al. 2019 —
both are linked from the [API reference](https://farishrvat.github.io/mortis/api/preprocessing/).
