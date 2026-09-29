# Filtering

## `filter_by_score`

```python
mt.filter_by_score(
    adata, min_score: float = 0.3, score_col: str = "score", copy: bool = False,
) -> anndata.AnnData
```

Keep only metabolites whose annotation confidence score meets a minimum
threshold. Scores typically come from METASPACE/SCiLS and range 0 (no
confidence) - 2 (high confidence / library match). If `adata.var['score']`
isn't already populated (common for raw `.xlsx`/`.csv` exports), run
[`load_annotation_scores()`](io.md#load_annotation_scores) first.

| Parameter | Default | Description |
|---|---|---|
| `min_score` | `0.3` | Recommended: `0.3` (moderate) · `0.5` (high) · `0.8` (very high / library-match only) |
| `score_col` | `"score"` | Column in `adata.var` to filter on |

**Returns:** `AnnData` with low-confidence metabolites removed;
`adata.uns['filter_score']` records the parameters used.

**Raises:** [`InvalidParameterError`](exceptions.md#invalidparametererror)
if `score_col` doesn't exist or `min_score` is outside `[0, 2]`.

```python
adata = mt.filter_by_score(adata, min_score=0.5)
```

---

## `filter_drugs`

```python
mt.filter_drugs(
    adata, db_path: str | None = None, remove_all: bool = True,
    drug_names: list[str] | None = None, copy: bool = False,
) -> anndata.AnnData
```

Remove metabolites that match the bundled drug vocabulary
(`mortis/data/drug_names.db`), which is resolved automatically when
`db_path=None`. No setup is needed. Pass a path to use your own file, for
instance a DrugBank export if you hold a licence for one.

The vocabulary holds roughly 20,000 drug names and 44,000 synonyms, built from
[Wikidata](https://www.wikidata.org): an entry counts as a drug when Wikidata
gives it a DrugBank or ATC identifier, or files it under medication or
pharmaceutical product. Wikidata is CC0, so the file ships inside the wheel.
Rebuild it with `python tools/build_drug_vocabulary.py`.

!!! warning "Many ordinary metabolites are also sold as drugs"
    Taurine, glycine, carnitine, cholesterol and most amino acids all carry
    drug identifiers, so `remove_all=True` deletes them from your panel along
    with the xenobiotics. `filter_drugs()` warns when it is about to do this,
    and [`list_drug_matches()`](#list_drug_matches) flags them in an
    `endogenous` column. Look at that list before you filter.

!!! warning "Optional QC step, not a mandatory pipeline stage"
    MSI is also widely used to visualise drug and xenobiotic distribution as
    the analyte of interest, in in-situ pharmacokinetic studies. If that is
    your use case, skip this and use
    [`list_drug_matches()`](#list_drug_matches) or
    [`score_metabolite_set()`](analysis-multisample.md#score_metabolite_set)
    to *find* drug ions instead of removing them.

| Parameter | Default | Description |
|---|---|---|
| `remove_all` | `True` | `True` removes every match. `False` removes only `drug_names` |
| `drug_names` | `None` | Names to remove, matched case-insensitively against names and synonyms |

**Returns:** `AnnData` with drug metabolites removed;
`adata.uns['filter_drugs']` records what was removed.

**Raises:** `FileNotFoundError` if `db_path` doesn't exist;
[`InvalidParameterError`](exceptions.md#invalidparametererror) if
`remove_all=False` with no `drug_names`.

```python
# Remove everything that matches the vocabulary
adata = mt.filter_drugs(adata)

# Remove only specific drugs
adata = mt.filter_drugs(adata, remove_all=False,
                         drug_names=["Aspirin", "Ibuprofen"])
```

---

## `list_drug_matches`

```python
mt.list_drug_matches(adata, db_path: str | None = None) -> pandas.DataFrame
```

Preview which metabolites [`filter_drugs()`](#filter_drugs) *would*
remove, without removing them, inspect before committing, or use this
as the entry point for a drug-distribution study.

**Returns:** `DataFrame` with columns `metabolite`, `is_drug` and
`endogenous`, the last marking compounds the body makes for itself.

```python
matches = mt.list_drug_matches(adata)
print(matches[~matches.endogenous])   # the genuinely xenobiotic ones
```

