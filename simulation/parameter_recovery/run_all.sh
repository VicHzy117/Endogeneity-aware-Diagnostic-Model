#!/bin/bash
set -euo pipefail

gen_job=$(sbatch slurm/generate_data.sbatch | awk '{print $4}')
fit_job=$(sbatch --dependency=afterok:${gen_job} slurm/run_replicates_array.sbatch | awk '{print $4}')
agg_job=$(sbatch --dependency=afterok:${fit_job} slurm/aggregate_table1.sbatch | awk '{print $4}')

echo "Submitted data job ${gen_job}, fit array ${fit_job}, and aggregation job ${agg_job}."
