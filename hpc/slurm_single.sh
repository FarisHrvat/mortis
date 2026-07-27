#!/bin/bash
#SBATCH --job-name=mortis
#SBATCH --output=logs/mortis-%j.out
#SBATCH --error=logs/mortis-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=compute          # EDIT: your partition

# One cohort, one node. Submit with:
#     mkdir -p logs && sbatch hpc/slurm_single.sh analysis.yaml

set -euo pipefail

CONFIG="${1:-analysis.yaml}"

# Thread limits BEFORE python starts. BLAS reads these once, when its pool
# initialises on `import numpy` — set them afterwards and they are ignored,
# silently, while your job quietly oversubscribes the node.
CPUS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$CPUS"
export MKL_NUM_THREADS="$CPUS"
export OPENBLAS_NUM_THREADS="$CPUS"
export NUMEXPR_NUM_THREADS="$CPUS"
export NUMBA_NUM_THREADS="$CPUS"
export MORTIS_N_JOBS="$CPUS"
export MPLBACKEND=Agg               # no display on a compute node

# EDIT: however your site provides Python.
# module load python/3.11
source "${MORTIS_VENV:-$HOME/venvs/mortis}/bin/activate"

echo "job ${SLURM_JOB_ID:-local} on $(hostname), ${CPUS} cpus"
mortis info

# Fail before the queue time is spent, not after.
if [[ ! -f "$CONFIG" ]]; then
    echo "No config at '$CONFIG'. Make one with: mortis template > analysis.yaml" >&2
    exit 1
fi

time mortis run "$CONFIG"

# A job that exits 0 has not necessarily produced the right numbers.
RESULTS=$(python - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["output"]["dir"])
PY
)
if [[ -f "$RESULTS/manifest.json" ]]; then
    mortis verify "$RESULTS/manifest.json" --data "$RESULTS/pseudobulk.h5ad" || true
fi
