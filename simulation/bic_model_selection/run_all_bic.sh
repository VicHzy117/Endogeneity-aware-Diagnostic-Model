#!/bin/bash
set -euo pipefail

bic_job=$(sbatch slurm/run_bic_array.sbatch | awk '{print $4}')
agg_job=$(sbatch --dependency=afterok:${bic_job} slurm/aggregate_bic_selection.sbatch | awk '{print $4}')

echo "Submitted BIC array ${bic_job} and dependent aggregation job ${agg_job}."

