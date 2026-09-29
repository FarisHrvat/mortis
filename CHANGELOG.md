# Changelog

Notable changes, in [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
order, versioned with [SemVer](https://semver.org/).

## [0.1.0], unreleased

The first version worth putting a number on. Everything before this was me
finding out what the package needed to be; the version counter is starting here
because that is honest, even though the code has been through a lot more than
"0.1" usually implies.

It works, it is tested on real public data, and it is under review for
publication. What it is not yet is released.

### The four things that were quietly wrong

Every one of these produced numbers. Wrong ones. Each was reproduced before it
was fixed, and each now has a test that fails on the old code.

- **Spatial coordinates collapsed when sections were combined.** Batches were
  pushed apart by `i * 1e6` in float32, and float32 runs out of precision at
  around 8.4 million, past which neighbouring pixels round onto each other. On
  a real 52-section cohort, **35 sections lost roughly half their distinct pixel
  positions**, which silently corrupted Moran's I, Geary's C, Gi\*, LISA,
  co-occurrence, spatial domains and spatially-weighted NMF for all of them.
  Now float64, with the offset derived from the actual coordinate range instead
  of a hard-coded constant that assumed your microns were small.
- **A cache that never noticed the data had changed.** `_get_X` memoised a dense
  copy into `.uns` and never invalidated it, so anything that replaced `.X`,
  `scale()`, `correct_batches()`, sparse normalisation, left every later call
  reading pre-correction data. Wiping `.X` to all zeros still returned the
  original statistics, cheerfully. It also kept up to four full copies of the
  matrix (~2.4 GB on a 100k × 2000 dataset) and wrote them into every saved
  `.h5ad`. Deleted. Rebuilding costs about half a second.
- **Permutation tests were not reproducible.** `np.random.seed` inside a
  `@njit(parallel=True)` function seeds exactly one worker thread and leaves the
  rest to their own devices, so results depended on how many cores you had. Same
  seed, 1 vs 4 threads, |Δz| up to 0.79, and not even repeatable twice in the
  same process. Each permutation now draws its own independent seed stream.
- **`compare_groups` was testing pixels as if they were patients.** On simulated
  data with six patients and *no group difference at all*, it called **183 of
  200** metabolites significant. It still exists for genuine within-section
  comparisons, but it now warns and points at the sample-level path.

### Added

- **`mortis.stats`**: the sample-level statistics that make everything else
  legitimate. `pseudobulk()`, `differential_abundance()` (Cliff's δ first,
  Mann-Whitney and BH-FDR as supporting detail, optional bootstrap interval),
  `paired_differential_abundance()` for before/after designs, and
  `cliffs_delta()` on its own.
- **`mortis.organization`**: differential spatial *organization*. Summarises
  how each metabolite is arranged per section (Moran's I, normalised entropy,
  Gi\* hotspot fraction, Gini), then tests those the same way abundance is
  tested. `compare_abundance_and_organization()` labels which axis moved; the
  "organization only" class is the one nothing else can find.

  Benchmarked honestly: with 3 planted differences among 30 metabolites, effect
  size alone over-called at every cohort size, while effect size plus FDR gave
  exactly 3 true and 0 false **from six sections per group upward, and nothing
  below it**. Six per arm is the floor, it is documented, and it is pinned by a
  test.
- **`mortis.compare`**: `cross_cohort_profile()` and `track_flow()`: do two
  drugs move the same metabolites, and does a signature persist, reorganise or
  flip over time.
- **`mortis.annotate`**: layered chemical-class assignment, class-level
  enrichment, and directional pathway over-representation. On a real
  2,231-compound untargeted panel the built-in name rules leave about 55%
  unclassified; a `reference=` mapping from HMDB or LIPID MAPS takes that to
  28%. Both numbers are in the docstring rather than hidden, because a
  classifier that quietly places two-thirds of a panel and says nothing about
  the rest is the exact failure this module exists to avoid.
- **`mortis.pathway`**: compound names → HMDB/KEGG identifiers → pathways →
  enrichment, in one call, cached on disk. The original plan was to hand
  enrichment to MetaboAnalyst; it turns out to document exactly one REST
  endpoint and enrichment is not it. Doing the statistics locally is better
  anyway, because the background set decides the answer and a web service
  cannot know which compounds *your* instrument saw.
- **`mortis.reproducibility`**: `export_manifest()` and `verify_manifest()`.
  A sealed JSON recording the environment, every recorded step, and SHA-256
  fingerprints of the inputs and every result table. Hand it to a reviewer:
  they can confirm your analysis reproduces without you sending them any data,
  because checksums only go one way. "Available on reasonable request" verifies
  nothing; this verifies something.
- **`mortis.viz`**: publication figures. PDF text stays *text* (embedded
  TrueType with a character map), so a co-author can retype a label in
  Illustrator instead of emailing you about it. Only dense scatter interiors get
  rasterised. Every figure carries its parameters and a hash in the PDF
  metadata, so `pdfinfo` will tell you which run made it long after you have
  forgotten. `theme="light"/"dark"` renders on a transparent ground for slides
  and web, and `ion_cmap()` replaces viridis as the ion-image default, still
  perceptually ordered, just less obviously the work of a plotting library.
- **`validation/run_validation.py`**: the package run end-to-end against a
  public METASPACE study, no simulation and nothing else in the pipeline.
- **A documentation site** in `web/`, with the API reference generated from the
  installed package at build time so it cannot drift from the code.

### Changed

- **The bundled drug list is built from Wikidata instead of DrugBank.**
  DrugBank releases its data under CC BY-NC, which does not permit shipping it
  inside a wheel that anyone, including commercial users, can install from
  PyPI. The vocabulary is now assembled from Wikidata, which is CC0: an entry
  counts as a drug when Wikidata gives it a DrugBank or ATC identifier, or
  files it under medication or pharmaceutical product. That comes to 20,181
  names and 43,697 synonyms, against 17,430 and 45,731 before, and every drug
  in a 19-compound spot check is still found. `tools/build_drug_vocabulary.py`
  rebuilds it. `list_drug_matches()` returns `is_drug` in place of
  `in_drugbank`, and adds an `endogenous` column, since taurine, cholesterol
  and most amino acids carry drug identifiers and `remove_all=True` would
  otherwise delete them from a metabolomics panel without comment.
- **`leidenalg` and `python-igraph` moved to a `[cluster]` extra.** Both are
  GPL while MORTIS is MIT, so installing them makes the whole environment GPL.
  That is a decision for whoever installs it, not something `pip install
  mortis-spatial` should make on their behalf. `cluster()` and
  `spatial_domains()` raise with the install command when they are missing;
  nothing else in the package touches them.
- **The Leiden backend is named explicitly.** scanpy is switching its default
  from `leidenalg` to `igraph`, and the two do not give the same partition, so
  an unpinned call would have quietly changed everybody's clusters on a scanpy
  upgrade. Minimum scanpy is now 1.10, which is where the argument appeared.
- **`correct_batches()` raises when ComBat fails** instead of mean-centring
  each batch and printing a line about it. Substituting a weaker method
  returns data corrected by something other than what was asked for, and other
  than what the methods section will say.
- **Warnings go through `warnings.warn`.** Three modules printed them to
  stdout, where they could not be filtered, caught or redirected, and were
  invisible to `pytest.warns`.
- **`mortis.audit` removed.** `export_manifest()` records everything it did,
  plus a verification step it never had. Two receipt systems in one package
  was one too many.
- **`merge_samples()` no longer writes into the list it was given.** It
  replaced the caller's elements with relabelled copies.

- **Spatial statistics stream over metabolite tiles.** The direct formulation
  held several full pixels × metabolites matrices at once. Since the reduction
  is over pixels and every output is one scalar per metabolite, metabolites can
  be processed in tiles, exactly the same arithmetic, reassociated. Combined
  with deriving `W @ (X − mean)` from `W @ X` algebraically, on a 95,751 × 2,231
  dataset: **1.23 s → 0.62 s, working set 4.30 GB → 0.99 GB, bit-identical
  checksums.**

  Tile size is picked from free RAM rather than cache size, because
  cache-sized tiles measured *slowest*: SciPy walks the whole sparse structure
  of the weights matrix once per tile regardless of how many columns come along
  for the ride, so amortising that beats locality. This was genuinely
  counter-intuitive and the measurement is in the code.
- Sparse and disk-backed (`backed="r"`) inputs are now covered by tests proving
  they give identical answers to dense ones.
- `metabolite_colocalization()` gained `metric="cosine_median"`, the measure
  that actually won the ColocML benchmark. Opt-in, not default, because it
  rasterises by coordinate span and scattered coordinates would ask for an
  enormous empty grid, now guarded, after it hung the test suite once.
- The ColocML citation was wrong. It is Ovchinnikova, Stuart, Rakhlin,
  Nikolenko & Alexandrov, *Bioinformatics* 2020;36(10):3215-3224, not
  "Ryabchykov et al.", which is a paper about something else entirely.
- `local_moran()` gained a docstring, including an explicit warning that its
  p-values are approximate rather than Anselin's conditional permutation. Fine
  for ranking pixels and drawing a LISA map; not something to report as
  calibrated inference, and it now says so instead of calling them "fast
  analytical p-values".
- Comments and docstrings across the older modules were rewritten out of
  marketing voice. "Seamless integration", "Intelligently detects",
  "Ultra-fast index creation utilizing Python List Comprehensions" and a
  section header reading `KILLER FEATURES` are all gone. None of it told a
  reader anything they could use.
- **Minimum Python is now 3.10**, not 3.9. 3.9 was declared and never tested;
  the CI matrix has always started at 3.10. Verified rather than assumed.
- `test_data/` is explicitly gitignored. The previous global `*.h5ad` rule
  covered two fixtures and missed the patient CSV entirely.

### Fixed

- Progress messages crashed on a Windows console using a legacy code page.
  `->`, `>=` and similar characters cannot be encoded in cp1252, so a
  `print()` containing one raised `UnicodeEncodeError` instead of reporting
  progress. The package source is ASCII now, bar a plus-minus in the
  stereodescriptor regex and a micrometre sign in one axis label, and a test
  keeps it that way.
- Every public function has a docstring. Nineteen of them had none, so `help()`
  and IDE tooltips came back blank even though the documentation site covered
  them.
- `neighborhood_enrichment(n_jobs=...)` crashed with `ValueError: The number of
  threads must be between 1 and N` when asked for more threads than the machine
  has. Numba fixes its ceiling at import; asking for more is a wish, not an
  error, so the request is clamped.
- Compatible with pandas 3 and anndata 0.13, both of which arrive by default on
  Python 3.12. Two test assumptions broke there, pandas 3 string columns do not
  support 2-D fancy indexing, and anndata 0.13 lists `.X` under a `None` key in
  `layers`, while the package itself was already correct. Error messages that
  list available layers now filter that `None` out, because showing it to
  someone hunting for a metric name helps nobody.
- `save_figure()` used `Path.with_suffix("")`, which eats everything after the
  last dot: `two_axis.dark` quietly became `two_axis`, and `figure_v1.2` would
  have lost its version.
- Two runs of the same analysis produced different files. Result tables sorted
  on effect size with no tiebreaker, so equally-ranked metabolites came out in
  whatever order the sort happened to leave them; and figure PDFs carried the
  wall-clock time. Ties now break on the compound name, and `SOURCE_DATE_EPOCH`
  is honoured, so `diff` is a usable way to ask whether anything changed. The
  public-data validation reproduces byte-for-byte apart from the manifest,
  which records when the run happened on purpose.
- Error messages said what was wrong but not what to do about it.
  `'sample' not found in adata.obs.` is technically accurate and practically
  useless; it now lists the columns that do exist and, when the name looks like
  a typo, guesses which one you meant. Several also told you to call
  `MORTIS.preprocess()`, which is not how the package is imported.
- `plot_abundance_vs_organization` and `plot_signature_comparison` drew one dot
  per coordinate, and on a small cohort Cliff's delta takes so few distinct
  values that a whole panel collapses onto a handful of points. The figure
  showed twelve dots while the legend said 160. Marker area now scales with how
  many metabolites share a position, and the figure says so.
- Group labels on `plot_organization_heatmap` were rotated, so on a two-section
  arm the text was taller than its own band and the group names printed over
  each other.

