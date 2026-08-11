#!/usr/bin/env bash
#SBATCH --job-name=cdmcmp-data
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=log/generate_data_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
if [[ ! -f data/simulation_data.rds ]]; then Rscript data/generate_simulation_data.R --out_dir=data; fi
Rscript code/validate_data.R
