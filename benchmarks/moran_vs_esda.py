"""
Moran's I on a wide panel: MORTIS against esda, on identical input.

Measures peak resident memory and wall time for both, and checks the numbers
agree. Run it as:

    python benchmarks/moran_vs_esda.py mortis 2000
    python benchmarks/moran_vs_esda.py esda 2000

Each run is a separate process, because peak RSS is per process and importing
both libraries into one would charge each of them for the other's memory.
"""

import json
import resource
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd

PIXELS = 60_000


def build(n_metabolites: int, jitter: bool):
    """A square raster of pixels with random intensities.

    ``jitter`` moves every pixel a little. On an exact lattice most pixels
    have several neighbours at the same distance, so two k-NN implementations
    can pick different ones and get different, equally correct answers. The
    jitter removes that ambiguity when the point is to compare the arithmetic.
    """
    side = int(np.sqrt(PIXELS))
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.c_[xs.ravel(), ys.ravel()].astype(np.float64)
    if jitter:
        coords += np.random.default_rng(7).normal(0, 0.05, coords.shape)

    rng = np.random.default_rng(0)
    X = np.abs(rng.normal(100, 25, (len(coords), n_metabolites))).astype(np.float32)
    obs = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1], "section": "s1"},
        index=[f"p{i}" for i in range(len(coords))],
    )
    adata = ad.AnnData(X=X, obs=obs,
                       var=pd.DataFrame(index=[f"m{i}" for i in range(n_metabolites)]))
    adata.obsm["spatial"] = coords.astype(np.float32)
    return adata, coords


def main() -> None:
    tool = sys.argv[1]
    n_metabolites = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    jitter = "--lattice" not in sys.argv

    adata, coords = build(n_metabolites, jitter)
    start = time.perf_counter()

    if tool == "mortis":
        import mortis as mt
        org = mt.spatial_organization(adata, sample_key="section", metrics=("morans_i",))
        values = np.asarray(org.X)[0]
    elif tool == "esda":
        from esda.moran import Moran
        from libpysal.weights import KNN
        weights = KNN.from_array(coords, k=6)
        weights.transform = "r"
        matrix = np.asarray(adata.X)
        values = np.array([Moran(matrix[:, j], weights, permutations=0).I
                           for j in range(n_metabolites)])
    else:
        raise SystemExit(f"unknown tool {tool!r}, pick 'mortis' or 'esda'")

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = peak / 1024**3 if sys.platform == "darwin" else peak / 1024**2
    print(json.dumps({
        "tool": tool,
        "metabolites": n_metabolites,
        "pixels": int(adata.n_obs),
        "lattice": not jitter,
        "peak_gb": round(peak, 2),
        "seconds": round(time.perf_counter() - start, 1),
        "first_five": [round(float(v), 8) for v in values[:5]],
    }))


if __name__ == "__main__":
    main()
