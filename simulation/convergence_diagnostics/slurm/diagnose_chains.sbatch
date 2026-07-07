#!/bin/bash
#SBATCH --job-name=eacdm_conv_diag
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhuang@fredhutch.org

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

if command -v ml >/dev/null 2>&1; then
  ml R/4.4.2-gfbf-2024a
elif command -v module >/dev/null 2>&1; then
  module load R/4.4.2-gfbf-2024a
fi

Rscript code/diagnose_chains.R \
  --data_dir=data/generated \
  --output_dir=output \
  --burnin="${BURNIN:-2000}" \
  --chains=4 \
  --trace_scenarios="${TRACE_SCENARIOS:-13,11,6}" \
  --trace_replicate="${TRACE_REPLICATE:-50}"

