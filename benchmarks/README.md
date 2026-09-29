# Benchmarks

Numbers measured on an Apple M3 Max, Python 3.12, against esda 2.10 and
libpysal 4.15. Reproduce with `moran_vs_esda.py`; every run is its own process
because peak RSS is per process.

## Moran's I on a wide panel

59,536 pixels, one section, k = 6.

| metabolites | | peak RSS | wall time |
|---|---|---|---|
| 200 | MORTIS | 0.57 GB | **1.0 s** |
| 200 | esda + libpysal | 0.60 GB | 20.4 s |
| 2000 | MORTIS | 2.83 GB | **1.3 s** |
| 2000 | esda + libpysal | 2.20 GB | 197.0 s |

MORTIS is about 20x faster at 200 metabolites and about 150x at 2000, because
it does one sparse matrix product for the whole panel rather than looping a
per-variable call. Memory is comparable, and at 2000 metabolites MORTIS is
actually about 30% higher: the tiling caps the intermediates, not the input
matrix, and it keeps the four summary metrics per section.

Neither figure supports a claim that MORTIS is the lightest tool available.
The speed difference is the real result.

## Whole workflow, ten sections of real tissue

335,009 pixels, 47 shared compounds, load through to a differential
organisation table.

| | peak RSS | wall time |
|---|---|---|
| MORTIS | 1.96 GB | 45.2 s |
| scanpy + esda + libpysal + scipy + statsmodels | 1.89 GB | 31.3 s |

On a narrow panel the hand-assembled version is slightly lighter and faster.
MORTIS validates its inputs, carries the sample metadata through, and records
provenance, and that costs something. What it buys is 23 statements instead of
108 across five libraries, and no opportunity to wire the FDR correction to the
wrong axis.

Note `preprocess(do_neighbors=False)`. The kNN graph in expression space is the
most expensive step in preprocessing and is only needed for clustering and
UMAP; the spatial statistics build their own graph from pixel coordinates.
Leaving it on costs 0.5 GB and 23 s here for nothing.

## Precision

Against esda's float64 implementation, 59,536 pixels, coordinates jittered so
the k-NN graph is unambiguous:

| Moran's I | MORTIS | esda | relative difference |
|---|---|---|---|
| strong structure | 0.9595555663 | 0.9595605870 | 5.2e-06 |
| moderate | 0.5720750690 | 0.5720661614 | 1.6e-05 |
| weak | 0.0983663499 | 0.0983637819 | 2.6e-05 |
| none | 0.0082541052 | 0.0082637231 | 1.2e-03 |

Five significant figures, which comes from MORTIS accumulating in float32
where esda uses float64. That is a deliberate trade for memory. The relative
error only becomes large where Moran's I is already indistinguishable from
zero, which is a case no result rests on.

On an exact pixel lattice the two disagree by about 6e-03, and that is not a
precision difference. Most pixels on a raster have several neighbours at
identical distance, so "the 6 nearest" is ambiguous and two implementations
pick different sets. Both answers are correct for the graph each built. On
irregular coordinates the two agree to 5e-08 and `numpy.allclose` passes.
