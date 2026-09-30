# Contributing

Thanks for looking. This is a scientific package, which means a bug here does
not crash. It prints a number that is wrong, and somebody puts that number in a
paper. Most of what follows exists because of that.

## Setting up

```bash
git clone https://github.com/FarisHrvat/mortis.git && cd mortis
python -m venv .venv && source .venv/bin/activate   # or conda/micromamba
pip install -e ".[dev]"
```

Python 3.10 or newer. CI runs 3.10 through 3.14 on Linux, Windows and Apple
Silicon macOS, so those are the versions that are actually promised.

Worth knowing: on 3.12 and newer, pip resolves **pandas 3 and anndata 0.13**,
which behave differently from what 3.10 and 3.11 get. If a test passes locally
and fails in CI, that is the first thing to check.

## Before you open a pull request

```bash
pytest          # 610 tests, about 25 seconds
ruff check .
```

Both have to pass. CI runs exactly these.

## Tests

The rule: **a test for a bug fix has to fail on the code before the fix.**

If it passes both before and after, it is not testing what you think it is. When
the four defects in `test_reproducibility.py` were fixed, the new tests were run
against the old code first, 9 of 18 failed, which is how anyone knows they mean
something.

Some practical consequences:

- **Check the artefact, not the setting.** "PDF text stays editable" is a
  property of the bytes in the file, so the test looks for `/FontFile2` in the
  PDF and round-trips it through `pdftotext`. Asserting that an rcParam was set
  proves nothing about the file a co-author opens.
- **Look at figures.** Three real bugs: clipped axis labels, a title landing on
  the panel labels, and a legend sitting on top of the bars, passed every
  assertion and were found by rendering a PNG and looking at it.
- **Network tests are opt-in.** `tests/test_pathway.py` runs offline against
  captured payloads. The live-service tests need `MORTIS_TEST_NETWORK=1`, so
  the suite neither flakes nor hammers somebody else's free API.

## Statistics

One non-negotiable, because it is the whole reason this package exists:
**pixels are not replicates.**

Anything that compares groups of samples goes through `pseudobulk()` first.
A section has tens of thousands of pixels and one patient; testing the pixels
inflates *n* by four orders of magnitude. On simulated null data that is the
difference between 0 findings and 183 of them.

If you add a test that compares groups, it needs a null-data case showing it
does not invent results.

## Writing

Comments explain **why**, not what. The code already says what.

```python
# Bad, restates the line below it
# Set the number of threads
nb.set_num_threads(n)

# Good, says the thing you cannot see
# Numba fixes its ceiling at import from the core count, so asking for more
# than the machine has raises rather than just using what is available.
nb.set_num_threads(int(np.clip(requested, 1, nb.config.NUMBA_NUM_THREADS)))
```

Please avoid marketing voice. "Seamless integration", "intelligently detects"
and "ultra-fast" have all been removed from this codebase once already and none
of them told a reader anything actionable. Plain sentences, and a joke now and
then is fine, the package is named after rigor mortis.

Document limits where they exist. `classify_compounds` says out loud that it
leaves ~55% of an untargeted panel unclassified, and
`compare_abundance_and_organization` says it needs six sections per arm. A
number you are slightly embarrassed by is worth more than a claim nobody
checked.

## Data

**Never commit patient data.** `test_data/` is gitignored and stays that way.
Derived figures are fine; the arrays that made them are not.

`validation/` runs on public METASPACE data. If you extend it, keep it that
way, so that anybody can run it.

## Docs

The site lives in `web/`. `web/api.json` is generated from the installed
package, so do not hand-edit it:

```bash
python web/build_api.py
```

CI regenerates it on every deploy, and fails if an exported function is missing
from a group in `web/build_api.py`, which is how new functions avoid quietly
going undocumented.

## Releasing

1. Update `CHANGELOG.md`, real sentences, not a list of commit subjects.
2. Bump `version` in `pyproject.toml` and `CITATION.cff`.
3. Tag it. The publish workflow does the rest.

## Anything else

Open an issue. A failing snippet and the output you expected is plenty, no
template to fill in.

