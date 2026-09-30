# Security

## Reporting

Use GitHub's [private vulnerability reporting][pvr] on this repository, or email
mortis-spatial@proton.me. Please do not open a public issue for anything that
looks exploitable.

I maintain this alongside a PhD, so expect a first reply within about a week
rather than within a day.

[pvr]: https://github.com/FarisHrvat/mortis/security/advisories/new

## Supported versions

The latest release on PyPI. There are no long-term support branches.

## What is worth reporting

MORTIS reads files and makes network calls, so the interesting surface is:

- **`.h5ad` and pickle deserialisation.** Reading an untrusted `.h5ad` runs
  whatever h5py and anndata will run. Treat files from strangers the way you
  would treat a pickle.
- **`mortis.pathway`**, which calls MetaboAnalyst and KEGG over HTTPS and caches
  the responses on disk. A malicious response that escapes the cache directory
  or poisons a later run would be a real finding.
- **The bundled drug vocabulary** (`mortis/data/drug_names.db`), which is a
  SQLite file read with a fixed query. Anything that turns it into arbitrary
  execution matters.
- **The CLI**, which reads a YAML config and writes files where the config says.
  A config that escapes its output directory counts.

## What is not a vulnerability

- Wrong numbers. Those are bugs and they matter more than most security issues
  here, but please open a normal issue so it can be discussed in the open.
- A dependency advisory that MORTIS does not actually reach. Say which call path
  gets there and it becomes a real report.
- Denial of service from feeding it a file too large for your machine. That is
  the documented cost of the analysis, not an attack.
