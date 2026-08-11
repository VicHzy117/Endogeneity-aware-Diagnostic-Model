#!/usr/bin/env bash
#SBATCH --job-name=eacdm-conv-data
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=log/generate_data_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
if [[ ! -f data/generated/manifest.csv ]]; then Rscript data/generate_diagnostic_data.R; fi
Rscript code/validate_data.R
