#!/usr/bin/env bash
set -euo pipefail

N_WORKERS=${N_WORKERS:-30}
ITERATION=${ITERATION:-3000}
BURNIN=${BURNIN:-2000}
FIT_K1_VALUES=${FIT_K1_VALUES:-1,2,3,4,5}
FIT_K2_VALUES=${FIT_K2_VALUES:-1,2,3,4,5}
BIC_SEED=${BIC_SEED:-960000}
mkdir -p log result/bic_fits result/summary result/plots
echo $$ > log/local_runner.pid
trap 'rm -f log/local_runner.pid' EXIT

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export EACDM_ITERATION=${ITERATION}
export EACDM_BURNIN=${BURNIN}
export EACDM_FIT_K1_VALUES=${FIT_K1_VALUES}
export EACDM_FIT_K2_VALUES=${FIT_K2_VALUES}
export EACDM_BIC_SEED=${BIC_SEED}

echo "Starting 600 replicate tasks with ${N_WORKERS} workers."
echo "Each task fits 25 candidate dimension pairs and resumes candidate by candidate."
seq 1 600 | xargs -P "${N_WORKERS}" -n 1 bash -c '
  task_id="$1"
  log_id=$(printf "%03d" "${task_id}")
  Rscript code/run_replicate_candidates.R \
    --task_id="${task_id}" \
    --iteration="${EACDM_ITERATION}" \
    --burnin="${EACDM_BURNIN}" \
    --fit_k1_values="${EACDM_FIT_K1_VALUES}" \
    --fit_k2_values="${EACDM_FIT_K2_VALUES}" \
    --seed="${EACDM_BIC_SEED}" \
    >"log/local_task_${log_id}.log" 2>&1
' _

Rscript code/audit_results.R \
  --iteration="${ITERATION}" --burnin="${BURNIN}" \
  --fit_k1_values="${FIT_K1_VALUES}" --fit_k2_values="${FIT_K2_VALUES}"
Rscript code/aggregate_results.R
echo "Big model-selection simulation complete. See result/summary and result/plots."
