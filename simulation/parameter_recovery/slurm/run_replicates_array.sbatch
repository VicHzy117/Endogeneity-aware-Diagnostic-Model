#!/bin/bash
#SBATCH --job-name=eacdm_exactq_pr
#SBATCH --array=1-1800%100
#SBATCH --output=log/run_%A_%a.log
#SBATCH --error=log/run_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhuang@fredhutch.org

ml R/4.4.2-gfbf-2024a 2>/dev/null || module load R/4.4.2-gfbf-2024a

cd "${SLURM_SUBMIT_DIR:?Submit this job from simulation/parameter_recovery}"

Rscript code/run_one_replicate.R \
  --task_id="${SLURM_ARRAY_TASK_ID}" \
  --data_dir=data/generated \
  --result_dir=result \
  --iteration="${ITERATION:-3000}" \
  --burnin="${BURNIN:-2000}" \
  --seed="${FIT_SEED:-910000}"
