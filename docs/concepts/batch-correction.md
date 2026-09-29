# Batch Correction & Verifying It Worked

When you combine multiple samples (patients, acquisition days, slides)
into one analysis, technical variation between them (instrument drift,
matrix application differences, day-to-day calibration) can dominate
over the biological signal you actually care about. Batch correction
tries to remove that technical variation while preserving real
biological differences.

## Two methods, two philosophies

| | [`mt.correct_batches()`](../api/preprocessing.md#correct_batches) (ComBat) | [`mt.run_harmony()`](../api/preprocessing.md#run_harmony) (Harmony) |
|---|---|---|
| Corrects | The expression matrix itself | The PCA embedding |
| Output | Overwrites `adata.X`/`adata.layers['log1p']` | New key: `adata.obsm['X_pca_harmony']` |
| Speed with many batches | Slower | Faster |
| Protect a biological covariate | `covariates=["condition"]` | Not applicable (Harmony's own iterative clustering already tends to preserve strong biological signal, but isn't given an explicit covariate to protect) |

```python
# ComBat, corrects the matrix, can protect a covariate from being "corrected away"
adata = mt.correct_batches(adata, batch_key="sample", covariates=["condition"])

# Harmony, corrects the embedding; point downstream steps at the new key
adata = mt.run_harmony(adata, batch_key="sample")
adata = mt.run_neighbors(adata, use_rep="X_pca_harmony")
adata = mt.run_umap(adata)
```

## Don't just assume it worked, measure it

This is the step people skip. Use
[`mt.batch_mixing_score()`](../api/analysis-validation.md#batch_mixing_score)
(LISI) **before and after** correction, on the same embedding type:

```python
adata = mt.batch_mixing_score(adata, batch_key="sample", use_rep="X_pca")
before = adata.obs["lisi_score"].mean()

adata = mt.run_harmony(adata, batch_key="sample")
adata = mt.batch_mixing_score(adata, batch_key="sample", use_rep="X_pca_harmony")
after = adata.obs["lisi_score"].mean()

print(f"LISI before: {before:.2f}, after: {after:.2f} (max possible = number of batches)")
```

### A real result, not a cherry-picked one

On the 14-sample Responder/Non-Responder cohort used in the
[Cohort Comparison tutorial](../tutorials/04-cohort-comparison.md):

```
mean LISI before harmony: 2.13   (out of 14 possible)
mean LISI after harmony:  2.69
```

That's a real, modest improvement (~26% relative increase in local
batch mixing), **not** a dramatic "problem solved" number. This is the
honest range you should expect on real clinical MSI data with strong
patient-to-patient chemistry differences; treat a LISI score that jumps
to near the maximum with more suspicion than reassurance (it can mean
biological signal was over-corrected away, not just batch effect
removed). Always sanity-check that a biological signal you know should
be there (e.g. a expected marker's spatial pattern) survives batch
correction.

## When batch correction isn't obviously needed

If you're analyzing a single sample, or samples that were all acquired
in one session on one slide with one calibration, batch correction may
add noise rather than remove it. Check `batch_mixing_score` on the raw
embedding first, if mixing is already reasonably high, correction may
not be worth the risk of over-correcting real biology away.

