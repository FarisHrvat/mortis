#!/bin/bash
#PBS -N mortis
#PBS -l select=1:ncpus=16:mem=32gb
#PBS -l walltime=04:00:00
#PBS -j oe

# The PBS/Torque equivalent of slurm_single.sh. Submit with:
#     qsub -v CONFIG=analysis.yaml hpc/pbs_single.sh

set -euo pipefail
cd "${PBS_O_WORKDIR:-$PWD}"

CONFIG="${CONFIG:-analysis.yaml}"

# PBS does not export a core count as tidily as Slurm does; NCPUS is the usual
# one, with the node file as a fallback.
CPUS="${NCPUS:-$(wc -l < "${PBS_NODEFILE:-/dev/null}" 2>/dev/null || echo 1)}"
export OMP_NUM_THREADS="$CPUS" MKL_NUM_THREADS="$CPUS" OPENBLAS_NUM_THREADS="$CPUS"
export NUMEXPR_NUM_THREADS="$CPUS" NUMBA_NUM_THREADS="$CPUS" MORTIS_N_JOBS="$CPUS"
export MPLBACKEND=Agg

# module load python/3.11
source "${MORTIS_VENV:-$HOME/venvs/mortis}/bin/activate"

echo "job ${PBS_JOBID:-local} on $(hostname), ${CPUS} cpus"
mortis info
time mortis run "$CONFIG"
