#!/bin/bash
#SBATCH --job-name=eacdm_bic_agg
#SBATCH --output=log/aggregate_bic_%j.log
#SBATCH --error=log/aggregate_bic_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhuang@fredhutch.org

ml R/4.4.2-gfbf-2024a 2>/dev/null || module load R/4.4.2-gfbf-2024a

Rscript code/aggregate_bic_selection.R --result_dir=result
