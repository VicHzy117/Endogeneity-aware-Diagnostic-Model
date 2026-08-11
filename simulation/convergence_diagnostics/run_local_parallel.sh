#!/usr/bin/env bash
set -euo pipefail

N_WORKERS=${N_WORKERS:-30}
ITERATION=${ITERATION:-3000}
BURNIN=${BURNIN:-2000}
mkdir -p log

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

seq 1 216 | xargs -P "${N_WORKERS}" -n 1 bash -c '
  task_id="$1"
  log_id=$(printf "%03d" "${task_id}")
  Rscript code/run_chain.R \
    --task_id="${task_id}" \
    --iteration="'"${ITERATION}"'" \
    --burnin="'"${BURNIN}"'" \
    >"log/chain_${log_id}.log" 2>&1
' _

Rscript code/audit_chains.R "${ITERATION}" "${BURNIN}"
Rscript code/diagnose_chains.R --burnin="${BURNIN}"
Rscript code/plot_running_rhat.R
echo "Convergence analysis complete. See output/diagnostics, output/traceplots, and output/running_rhat."
