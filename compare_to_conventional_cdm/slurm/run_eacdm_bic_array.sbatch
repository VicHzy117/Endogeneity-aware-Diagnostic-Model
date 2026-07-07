#!/bin/bash
#SBATCH --job-name=eacdm_bic
#SBATCH --array=1-900
#SBATCH --output=log/eacdm_bic_%A_%a.out
#SBATCH --error=log/eacdm_bic_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhuang@fredhutch.org

set -euo pipefail
ml R/4.4.2-gfbf-2024a
cd "${SLURM_SUBMIT_DIR}"

Rscript code/run_eacdm_bic.R \
  --task_id="${SLURM_ARRAY_TASK_ID}" \
  --data=data/simulation_data.rds \
  --result_dir=result \
  --setnum=100 \
  --k1_values=2,3,4 \
  --k2_values=2,3,4 \
  --iteration="${ITERATION:-3000}" \
  --burnin="${BURNIN:-2000}" \
  --seed=2000
