# Filtering

## `filter_by_score`

```python
mt.filter_by_score(
    adata, min_score: float = 0.3, score_col: str = "score", copy: bool = False,
) -> anndata.AnnData
```

Keep only metabolites whose annotation confidence score meets a minimum
threshold. Scores typically come from METASPACE/SCiLS and range 0 (no
confidence) – 2 (high confidence / library match). If `adata.var['score']`
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

Remove metabolites matching the [DrugBank](https://doi.org/10.1093/nar/gkx1037)
database — bundled with the package (`mortis/data/drugbank.db`) and
resolved automatically when `db_path=None` (the default). No setup
needed; pass an explicit path only if you want to use a different or
updated DrugBank export.

!!! warning "Optional QC step, not a mandatory pipeline stage"
    MSI is also widely used to **visualize** drug/xenobiotic
    distribution as the analyte of interest (in situ pharmacokinetic
    studies). If that's your use case, skip this and use
    [`list_drug_matches()`](#list_drug_matches) or
    [`score_metabolite_set()`](analysis-multisample.md#score_metabolite_set)
    to *find* drug ions instead of removing them.

| Parameter | Default | Description |
|---|---|---|
| `remove_all` | `True` | `True` — remove every DrugBank match. `False` — only remove `drug_names` |
| `drug_names` | `None` | Specific names to remove (case-insensitive, matched against DrugBank names + synonyms) |

**Returns:** `AnnData` with drug metabolites removed;
`adata.uns['filter_drugs']` records what was removed.

**Raises:** `FileNotFoundError` if `db_path` doesn't exist;
[`InvalidParameterError`](exceptions.md#invalidparametererror) if
`remove_all=False` with no `drug_names`.

```python
# Remove all DrugBank matches
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
remove, without removing them — inspect before committing, or use this
as the entry point for a drug-distribution study.

**Returns:** `DataFrame` with columns `metabolite`, `in_drugbank`.

```python
matches = mt.list_drug_matches(adata)
print(f"Found {len(matches)} drug metabolites in your data")
```
