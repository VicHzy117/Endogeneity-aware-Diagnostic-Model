#!/bin/bash
set -euo pipefail

set +e
Rscript code/audit_results.R
audit_status=$?
set -e
if [[ ${audit_status} -eq 0 ]]; then
  echo "Nothing to rerun."
  exit 0
fi
if [[ ${audit_status} -ne 2 ]]; then
  echo "Audit failed unexpectedly." >&2
  exit "${audit_status}"
fi

array_spec=$(<result/summary/rerun_array_spec.txt)
if [[ -z "${array_spec}" ]]; then
  echo "Audit reported missing tasks but produced an empty array specification." >&2
  exit 1
fi
sbatch --array="${array_spec}%${MAX_CONCURRENT:-100}" slurm/run_replicates_array.sbatch
