#!/usr/bin/env bash
#SBATCH --job-name=eacdm-conv-summary
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=log/diagnostics_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
Rscript code/audit_chains.R 3000 2000
Rscript code/diagnose_chains.R --burnin=2000
Rscript code/plot_running_rhat.R
