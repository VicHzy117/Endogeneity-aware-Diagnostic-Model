#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
data_dir="${DATA_DIR:-data/generated}"
result_dir="${RESULT_DIR:-result}"
log_dir="${LOG_DIR:-log}"
mkdir -p "${log_dir}" "${result_dir}/fits" "${result_dir}/summary"

if [[ ! -f "${data_dir}/manifest.csv" ]]; then
  export REGENERATE_DATA=true
fi

if [[ "${REGENERATE_DATA:-false}" == "true" ]]; then
  echo "Regenerating all simulation data because REGENERATE_DATA=true."
  Rscript data/generate_simulation_data.R \
    --out_dir="${data_dir}" \
    --setnum="${SETNUM:-100}" \
    --seed="${DATA_SEED:-20260517}"
fi

Rscript code/validate_data.R --data_dir="${data_dir}"
Rscript code/test_sampler.R

cpu_count=$(command -v nproc >/dev/null 2>&1 && nproc || echo 1)
if [[ -n "${N_WORKERS:-}" ]]; then
  workers="${N_WORKERS}"
else
  workers=$((cpu_count > 1 ? cpu_count - 1 : 1))
  (( workers > 4 )) && workers=4
fi
if ! [[ "${workers}" =~ ^[1-9][0-9]*$ ]]; then
  echo "N_WORKERS must be a positive integer; got '${workers}'." >&2
  exit 1
fi

iteration="${ITERATION:-3000}"
burnin="${BURNIN:-2000}"
fit_seed="${FIT_SEED:-910000}"
task_start="${TASK_START:-1}"
task_end="${TASK_END:-1800}"
if ! [[ "${task_start}" =~ ^[1-9][0-9]*$ && "${task_end}" =~ ^[1-9][0-9]*$ ]] ||
   (( task_start > task_end || task_end > 1800 )); then
  echo "TASK_START and TASK_END must satisfy 1 <= start <= end <= 1800." >&2
  exit 1
fi
export EACDM_ITERATION="${iteration}" EACDM_BURNIN="${burnin}" EACDM_FIT_SEED="${fit_seed}"
export EACDM_DATA_DIR="${data_dir}" EACDM_RESULT_DIR="${result_dir}" EACDM_LOG_DIR="${log_dir}"

echo "Starting tasks ${task_start}-${task_end} with ${workers} parallel workers."
echo "MCMC: ${iteration} iterations, ${burnin} burn-in."
echo "Existing valid result files are skipped, so rerunning this command resumes the job."

set +e
seq "${task_start}" "${task_end}" | xargs -P "${workers}" -n 1 bash -c '
  task_id="$1"
  log_id=$(printf "%04d" "${task_id}")
  Rscript code/run_one_replicate.R \
    --task_id="${task_id}" \
    --data_dir="${EACDM_DATA_DIR}" \
    --result_dir="${EACDM_RESULT_DIR}" \
    --iteration="${EACDM_ITERATION}" \
    --burnin="${EACDM_BURNIN}" \
    --seed="${EACDM_FIT_SEED}" \
    >"${EACDM_LOG_DIR}/local_task_${log_id}.log" 2>&1
' _
fit_status=$?
set -e

if (( task_start != 1 || task_end != 1800 )); then
  echo "Partial task range completed; skipping the 1800-fit audit and aggregation."
  exit "${fit_status}"
fi

set +e
Rscript code/audit_results.R --data_dir="${data_dir}" --result_dir="${result_dir}"
audit_status=$?
set -e

if [[ ${audit_status} -ne 0 ]]; then
  echo "Some fits are missing or invalid. Review log/local_task_*.log." >&2
  echo "After fixing the cause, run 'bash run_all.sh' again to resume." >&2
  exit_code="${fit_status}"
  [[ "${exit_code}" -eq 0 ]] && exit_code=2
  exit "${exit_code}"
fi

Rscript code/aggregate_table1.R --result_dir="${result_dir}"
echo "All 1800 fits and final summaries are complete."
