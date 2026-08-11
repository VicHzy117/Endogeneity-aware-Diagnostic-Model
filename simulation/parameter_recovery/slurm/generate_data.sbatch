#!/bin/bash
#SBATCH --job-name=eacdm_data
#SBATCH --output=log/generate_data_%j.log
#SBATCH --error=log/generate_data_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhuang@fredhutch.org

ml R/4.4.2-gfbf-2024a 2>/dev/null || module load R/4.4.2-gfbf-2024a

cd "${SLURM_SUBMIT_DIR:?Submit this job from simulation/parameter_recovery}"

Rscript data/generate_simulation_data.R \
  --out_dir=data/generated \
  --setnum="${SETNUM:-100}" \
  --seed="${DATA_SEED:-20260517}"

Rscript code/validate_data.R --data_dir=data/generated
