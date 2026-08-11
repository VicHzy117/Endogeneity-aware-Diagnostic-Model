#!/usr/bin/env bash
set -euo pipefail

N_WORKERS=${N_WORKERS:-30}
ITERATION=${ITERATION:-3000}
BURNIN=${BURNIN:-2000}
mkdir -p log output/diagnostics
Rscript code/audit_chains.R "${ITERATION}" "${BURNIN}" || true

missing_file=output/diagnostics/missing_task_ids.txt
if [[ ! -s "${missing_file}" ]]; then
  echo "No missing or invalid chains."
  exit 0
fi

xargs -P "${N_WORKERS}" -n 1 bash -c '
  task_id="$1"
  log_id=$(printf "%03d" "${task_id}")
  Rscript code/run_chain.R --task_id="${task_id}" \
    --iteration="'"${ITERATION}"'" --burnin="'"${BURNIN}"'" \
    >"log/chain_${log_id}.log" 2>&1
' _ < "${missing_file}"

Rscript code/audit_chains.R "${ITERATION}" "${BURNIN}"
Rscript code/diagnose_chains.R --burnin="${BURNIN}"
Rscript code/plot_running_rhat.R
