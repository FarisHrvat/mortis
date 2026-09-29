# Running MORTIS on a cluster

Everything here is a working starting point, not a framework. Copy the script
that matches your scheduler, change the parts marked `EDIT`, submit.

```
hpc/
  slurm_single.sh    one cohort, one node, start here
  slurm_array.sh     many cohorts in parallel, one array task each
  pbs_single.sh      the same, for PBS/Torque
  apptainer.def      a container for clusters that will not let you pip install
  environment.yml    a conda environment, when they will
```

## The two things that actually bite

**Threads.** Schedulers hand you a CPU allocation; libraries help themselves to
the whole machine. On a shared node that means a dozen jobs each spawning 128
threads, all of them slower than if they had behaved. Every script here sets
the thread limits from what the scheduler granted:

```bash
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MORTIS_N_JOBS=$SLURM_CPUS_PER_TASK
```

These must be set **before** Python starts. BLAS reads them once when its
thread pool initialises, which happens on `import numpy`, so exporting them
later has no effect at all. MORTIS learned this the hard way and the finding is
written up in `docs/concepts/performance.md`.

**Memory.** Ask for less than you think and the job dies four hours in. The
spatial statistics stream over metabolite tiles, so peak memory is roughly:

```
pixels × metabolites × 4 bytes × 2      +  ~2 GB for the interpreter and libraries
```

A 500,000-pixel cohort with 2,000 metabolites needs about 10 GB. The tile size
adapts to free RAM, so giving the job more genuinely makes it faster rather than
just safer.

## No network on the compute nodes

Common, and it only matters for `annotate_pathways()`, which fetches compound
identifiers and KEGG pathways. Warm the cache from the login node first:

```bash
python -c "import mortis as mt; mt.fetch_kegg_pathway_sets()"
```

It writes to `~/.cache/mortis`, which is usually on shared storage, and the
compute node then never reaches for the network. Everything else in the package
works offline.

## Checking it worked

Every run writes a manifest. Verifying it is the fastest way to find out whether
a job that *said* it succeeded actually produced the same numbers as last time:

```bash
mortis verify results/manifest.json --data results/pseudobulk.h5ad
```

Worth doing after a cluster migration or a module upgrade, when the environment
has moved underneath you and nothing announces it.

