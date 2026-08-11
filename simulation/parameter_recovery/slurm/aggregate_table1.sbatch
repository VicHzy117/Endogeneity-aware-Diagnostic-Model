#!/bin/bash
#SBATCH --job-name=eacdm_exactq_agg
#SBATCH --output=log/aggregate_%j.log
#SBATCH --error=log/aggregate_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhuang@fredhutch.org

ml R/4.4.2-gfbf-2024a 2>/dev/null || module load R/4.4.2-gfbf-2024a

cd "${SLURM_SUBMIT_DIR:?Submit this job from simulation/parameter_recovery}"

Rscript code/audit_results.R --data_dir=data/generated --result_dir=result
Rscript code/aggregate_table1.R --result_dir=result
