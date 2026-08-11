#!/usr/bin/env bash
set -euo pipefail

N_WORKERS=${N_WORKERS:-30}
ITERATION=${ITERATION:-3000}
BURNIN=${BURNIN:-2000}
mkdir -p log result/summary
Rscript code/audit_results.R "${ITERATION}" "${BURNIN}" || true
missing_file=result/summary/missing_task_ids.txt
if [[ ! -s "${missing_file}" ]]; then
  echo "No missing or invalid results."
  exit 0
fi
xargs -P "${N_WORKERS}" -n 1 bash -c '
  task_id="$1"
  log_id=$(printf "%04d" "${task_id}")
  Rscript code/run_task.R --task_id="${task_id}" \
    --iteration="'"${ITERATION}"'" --burnin="'"${BURNIN}"'" \
    >"log/task_${log_id}.log" 2>&1
' _ < "${missing_file}"
Rscript code/audit_results.R "${ITERATION}" "${BURNIN}"
Rscript code/aggregate_results.R
