#!/usr/bin/env bash
#SBATCH --job-name=eacdm-bigbic
#SBATCH --array=1-600%30
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=24:00:00
#SBATCH --output=log/slurm_task_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
Rscript code/run_replicate_candidates.R \
  --task_id="${SLURM_ARRAY_TASK_ID}" \
  --iteration="${ITERATION:-3000}" --burnin="${BURNIN:-2000}" \
  --fit_k1_values="${FIT_K1_VALUES:-1,2,3,4,5}" \
  --fit_k2_values="${FIT_K2_VALUES:-1,2,3,4,5}" \
  --seed="${BIC_SEED:-960000}"
