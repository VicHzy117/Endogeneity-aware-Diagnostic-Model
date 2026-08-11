#!/usr/bin/env bash
set -euo pipefail

N_WORKERS=${N_WORKERS:-30}
ITERATION=${ITERATION:-3000}
BURNIN=${BURNIN:-2000}
FIT_K1_VALUES=${FIT_K1_VALUES:-1,2,3,4,5}
FIT_K2_VALUES=${FIT_K2_VALUES:-1,2,3,4,5}
BIC_SEED=${BIC_SEED:-960000}
mkdir -p log result/summary

Rscript code/audit_results.R \
  --iteration="${ITERATION}" --burnin="${BURNIN}" \
  --fit_k1_values="${FIT_K1_VALUES}" --fit_k2_values="${FIT_K2_VALUES}" || true
missing_file=result/summary/missing_task_ids.txt
if [[ ! -s "${missing_file}" ]]; then
  echo "No missing or invalid replicate tasks."
  Rscript code/aggregate_results.R
  exit 0
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export EACDM_ITERATION=${ITERATION} EACDM_BURNIN=${BURNIN}
export EACDM_FIT_K1_VALUES=${FIT_K1_VALUES} EACDM_FIT_K2_VALUES=${FIT_K2_VALUES}
export EACDM_BIC_SEED=${BIC_SEED}

xargs -P "${N_WORKERS}" -n 1 bash -c '
  task_id="$1"
  log_id=$(printf "%03d" "${task_id}")
  Rscript code/run_replicate_candidates.R \
    --task_id="${task_id}" --iteration="${EACDM_ITERATION}" \
    --burnin="${EACDM_BURNIN}" \
    --fit_k1_values="${EACDM_FIT_K1_VALUES}" \
    --fit_k2_values="${EACDM_FIT_K2_VALUES}" \
    --seed="${EACDM_BIC_SEED}" \
    >"log/local_task_${log_id}.log" 2>&1
' _ < "${missing_file}"

Rscript code/audit_results.R \
  --iteration="${ITERATION}" --burnin="${BURNIN}" \
  --fit_k1_values="${FIT_K1_VALUES}" --fit_k2_values="${FIT_K2_VALUES}"
Rscript code/aggregate_results.R
