#!/usr/bin/env bash
#SBATCH --job-name=cdmcmp
#SBATCH --array=1-900%30
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=24:00:00
#SBATCH --output=log/slurm_task_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
Rscript code/run_task.R --task_id="${SLURM_ARRAY_TASK_ID}" --iteration=3000 --burnin=2000
