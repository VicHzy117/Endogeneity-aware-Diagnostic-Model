#!/bin/bash
#SBATCH --job-name=eacdm_bic
#SBATCH --array=1-15000
#SBATCH --output=log/bic_%A_%a.log
#SBATCH --error=log/bic_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhuang@fredhutch.org

ml R/4.4.2-gfbf-2024a 2>/dev/null || module load R/4.4.2-gfbf-2024a

Rscript code/run_bic_one_fit.R \
  --task_id="${SLURM_ARRAY_TASK_ID}" \
  --data_dir=../parameter_recovery/data/generated \
  --result_dir=result \
  --iteration="${ITERATION:-3000}" \
  --burnin="${BURNIN:-2000}" \
  --seed="${BIC_SEED:-930000}" \
  --fit_k1_values="${FIT_K1_VALUES:-1,2,3,4,5}" \
  --fit_k2_values="${FIT_K2_VALUES:-1,2,3,4,5}"
