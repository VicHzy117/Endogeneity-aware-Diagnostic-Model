#!/bin/bash
#SBATCH --job-name=sim_aggregate
#SBATCH --output=log/aggregate_%j.out
#SBATCH --error=log/aggregate_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhuang@fredhutch.org

set -euo pipefail
ml R/4.4.2-gfbf-2024a
cd "${SLURM_SUBMIT_DIR}"

Rscript code/aggregate_and_plot.R \
  --data=data/simulation_data.rds \
  --result_dir=result \
  --plot_best=TRUE
