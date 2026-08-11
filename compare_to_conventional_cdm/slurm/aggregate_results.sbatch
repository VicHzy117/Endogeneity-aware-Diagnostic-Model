#!/usr/bin/env bash
#SBATCH --job-name=cdmcmp-summary
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=log/aggregate_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
Rscript code/audit_results.R 3000 2000
Rscript code/aggregate_results.R
