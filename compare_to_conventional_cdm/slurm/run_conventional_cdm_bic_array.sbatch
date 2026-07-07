#!/bin/bash
#SBATCH --job-name=cdm_bic
#SBATCH --array=1-500
#SBATCH --output=log/conventional_cdm_bic_%A_%a.out
#SBATCH --error=log/conventional_cdm_bic_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhuang@fredhutch.org

set -euo pipefail
ml R/4.4.2-gfbf-2024a
cd "${SLURM_SUBMIT_DIR}"

Rscript code/run_conventional_cdm_bic.R \
  --task_id="${SLURM_ARRAY_TASK_ID}" \
  --data=data/simulation_data.rds \
  --result_dir=result \
  --setnum=100 \
  --k_values=2,3,4,5,6 \
  --iteration="${ITERATION:-3000}" \
  --burnin="${BURNIN:-2000}" \
  --seed=1000
