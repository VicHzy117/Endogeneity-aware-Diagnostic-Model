#!/usr/bin/env bash
set -euo pipefail

mkdir -p log output/chains output/diagnostics output/traceplots output/running_rhat
if [[ ! -f data/generated/manifest.csv ]]; then
  Rscript data/generate_diagnostic_data.R
fi
Rscript code/validate_data.R
Rscript code/test_sampler.R

if command -v sbatch >/dev/null 2>&1; then
  data_job=$(sbatch slurm/generate_data.sbatch | awk '{print $4}')
  chain_job=$(sbatch --dependency=afterok:${data_job} slurm/run_chains_array.sbatch | awk '{print $4}')
  diagnostic_job=$(sbatch --dependency=afterany:${chain_job} slurm/diagnose_chains.sbatch | awk '{print $4}')
  echo "Submitted Slurm jobs: data=${data_job}, chains=${chain_job}, diagnostics=${diagnostic_job}."
else
  echo "Slurm was not found; using the single-server parallel runner."
  echo "Starting 216 chain tasks with ${N_WORKERS:-30} workers."
  bash run_local_parallel.sh
fi
