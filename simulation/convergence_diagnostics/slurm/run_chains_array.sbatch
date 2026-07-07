#!/bin/bash
#SBATCH --job-name=eacdm_conv
#SBATCH --array=1-216
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhuang@fredhutch.org

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

if command -v ml >/dev/null 2>&1; then
  ml R/4.4.2-gfbf-2024a
elif command -v module >/dev/null 2>&1; then
  module load R/4.4.2-gfbf-2024a
fi

Rscript code/run_chain.R \
  --task_id="${SLURM_ARRAY_TASK_ID}" \
  --data_dir=data/generated \
  --output_dir=output \
  --iteration="${ITERATION:-3000}" \
  --burnin="${BURNIN:-2000}" \
  --seed="${CHAIN_SEED:-20260610}" \
  --chains=4

