#!/usr/bin/env bash
set -euo pipefail

N_WORKERS=${N_WORKERS:-30}
ITERATION=${ITERATION:-3000}
BURNIN=${BURNIN:-2000}
mkdir -p log result/eacdm result/summary result/plots
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

seq 1 900 | xargs -P "${N_WORKERS}" -n 1 bash -c '
  task_id="$1"
  log_id=$(printf "%04d" "${task_id}")
  Rscript code/run_task.R --task_id="${task_id}" \
    --iteration="'"${ITERATION}"'" --burnin="'"${BURNIN}"'" \
    >"log/task_${log_id}.log" 2>&1
' _

Rscript code/audit_results.R "${ITERATION}" "${BURNIN}"
Rscript code/aggregate_results.R
echo "Model comparison complete. See result/summary and result/plots."
