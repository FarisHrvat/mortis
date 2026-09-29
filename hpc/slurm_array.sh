#!/bin/bash
#SBATCH --job-name=mortis-array
#SBATCH --output=logs/mortis-%A_%a.out
#SBATCH --error=logs/mortis-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=compute          # EDIT
#SBATCH --array=0-9%4                # EDIT: 10 configs, at most 4 at once

# One array task per config, for a batch of cohorts, or the same cohort under
# several parameter choices. Put the configs in a directory and submit:
#
#     ls configs/*.yaml | wc -l          # set --array to N-1
#     mkdir -p logs && sbatch hpc/slurm_array.sh configs
#
# The %4 throttle exists to be a good neighbour. Ten jobs each taking every
# core on a shared filesystem is how you get an email from the sysadmin.

set -euo pipefail

CONFIG_DIR="${1:-configs}"
mapfile -t CONFIGS < <(find "$CONFIG_DIR" -name '*.yaml' | sort)

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "No .yaml configs in '$CONFIG_DIR'." >&2
    exit 1
fi
if [[ ${SLURM_ARRAY_TASK_ID:-0} -ge ${#CONFIGS[@]} ]]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} has no config (only ${#CONFIGS[@]} found), nothing to do."
    exit 0
fi

CONFIG="${CONFIGS[${SLURM_ARRAY_TASK_ID:-0}]}"

CPUS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$CPUS" MKL_NUM_THREADS="$CPUS" OPENBLAS_NUM_THREADS="$CPUS"
export NUMEXPR_NUM_THREADS="$CPUS" NUMBA_NUM_THREADS="$CPUS" MORTIS_N_JOBS="$CPUS"
export MPLBACKEND=Agg

# module load python/3.11
source "${MORTIS_VENV:-$HOME/venvs/mortis}/bin/activate"

echo "array task ${SLURM_ARRAY_TASK_ID:-0} -> $CONFIG"
time mortis run "$CONFIG"

