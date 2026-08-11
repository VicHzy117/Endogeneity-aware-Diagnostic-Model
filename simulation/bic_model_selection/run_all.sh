#!/usr/bin/env bash
set -euo pipefail

mkdir -p log result/bic_fits result/summary result/plots
if [[ ! -f data/generated/manifest.csv ]]; then
  echo "Generating the six n=1000 model-selection datasets from fixed seeds."
  Rscript data/generate_simulation_data.R \
    --out_dir=data/generated \
    --setnum=100 \
    --n_values=1000 \
    --j_values=24,36 \
    --k_values=2,3,4 \
    --seed=20260517
fi
Rscript code/validate_data.R
Rscript code/test_samplers.R

if command -v sbatch >/dev/null 2>&1; then
  fit_job=$(sbatch slurm/run_replicate_array.sbatch | awk '{print $4}')
  summary_job=$(sbatch --dependency=afterany:${fit_job} slurm/aggregate_results.sbatch | awk '{print $4}')
  echo "Submitted Slurm jobs: fits=${fit_job}, summary=${summary_job}."
else
  echo "Slurm was not found; using the single-server parallel runner."
  exec bash run_local_parallel.sh
fi
