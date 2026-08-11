#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Generated datasets are intentionally not stored in Git.  On a fresh clone,
# reproduce them from the fixed seeds before submitting the fits.
if [[ ! -f data/generated/manifest.csv ]]; then
  export REGENERATE_DATA=true
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Slurm was not found; switching to the single-server parallel runner."
  exec bash run_local_parallel.sh
fi

Rscript code/test_sampler.R

if [[ "${REGENERATE_DATA:-false}" == "true" ]]; then
  gen_job=$(sbatch slurm/generate_data.sbatch | awk '{print $4}')
  fit_job=$(sbatch --dependency=afterok:${gen_job} slurm/run_replicates_array.sbatch | awk '{print $4}')
  echo "Submitted optional data-generation job ${gen_job}."
else
  if [[ ! -f data/generated/manifest.csv ]]; then
    echo "Missing data/generated/manifest.csv." >&2
    echo "Copy the validated old datasets here or rerun with REGENERATE_DATA=true." >&2
    exit 1
  fi
  Rscript code/validate_data.R --data_dir=data/generated
  fit_job=$(sbatch --array="1-1800%${MAX_CONCURRENT:-100}" slurm/run_replicates_array.sbatch | awk '{print $4}')
fi

agg_job=$(sbatch --dependency=afterok:${fit_job} slurm/aggregate_table1.sbatch | awk '{print $4}')
echo "Submitted exact-Q fit array ${fit_job} and aggregation job ${agg_job}."
