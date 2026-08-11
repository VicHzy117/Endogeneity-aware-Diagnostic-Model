#!/usr/bin/env bash
set -euo pipefail

mkdir -p log result/eacdm result/summary result/plots
if [[ ! -f data/simulation_data.rds ]]; then
  Rscript data/generate_simulation_data.R --out_dir=data
fi
Rscript code/validate_data.R
Rscript code/test_samplers.R

if command -v sbatch >/dev/null 2>&1; then
  data_job=$(sbatch slurm/generate_data.sbatch | awk '{print $4}')
  fit_job=$(sbatch --dependency=afterok:${data_job} slurm/run_tasks_array.sbatch | awk '{print $4}')
  summary_job=$(sbatch --dependency=afterany:${fit_job} slurm/aggregate_results.sbatch | awk '{print $4}')
  echo "Submitted Slurm jobs: data=${data_job}, fits=${fit_job}, summary=${summary_job}."
else
  echo "Slurm was not found; using the single-server parallel runner."
  echo "Starting 900 revised EACDM fits with ${N_WORKERS:-30} workers."
  echo "The 500 original conventional-CDM results are bundled as a fixed baseline and are not rerun."
  bash run_local_parallel.sh
fi
