#!/bin/bash
#SBATCH --job-name=eacdm_conv_data
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhuang@fredhutch.org

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

if command -v ml >/dev/null 2>&1; then
  ml R/4.4.2-gfbf-2024a
elif command -v module >/dev/null 2>&1; then
  module load R/4.4.2-gfbf-2024a
fi

Rscript data/generate_diagnostic_data.R \
  --out_dir=data/generated \
  --seed="${DATA_SEED:-20260517}" \
  --replicate_ids="${REPLICATE_IDS:-1,50,100}"

