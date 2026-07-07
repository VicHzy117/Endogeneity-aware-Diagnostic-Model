#!/bin/bash
#SBATCH --job-name=sim_generate
#SBATCH --output=log/generate_data_%j.out
#SBATCH --error=log/generate_data_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhuang@fredhutch.org

set -euo pipefail
ml R/4.4.2-gfbf-2024a
cd "${SLURM_SUBMIT_DIR}"

Rscript data/generate_simulation_data.R \
  --out_dir=data \
  --setnum=100 \
  --n=1000 \
  --seed=237
