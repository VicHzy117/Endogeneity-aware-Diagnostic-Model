#!/usr/bin/env bash
set -euo pipefail

candidate_count=$(find result/bic_fits -name 'fit_K1_*_K2_*.rds' -type f 2>/dev/null | wc -l | tr -d ' ')
complete_replicates=$(find result/bic_fits -mindepth 2 -maxdepth 2 -type d 2>/dev/null | while read -r directory; do
  count=$(find "${directory}" -maxdepth 1 -name 'fit_K1_*_K2_*.rds' -type f | wc -l | tr -d ' ')
  if [[ "${count}" -eq 25 ]]; then echo 1; fi
done | wc -l | tr -d ' ')
running=$(ps -eo args 2>/dev/null | grep -c '[R]script code/run_replicate_candidates.R' || true)

echo "Candidate fits: ${candidate_count}/15000"
echo "Complete replicate tasks: ${complete_replicates}/600"
echo "Currently running R workers: ${running}"
if [[ -f log/local_runner.pid ]]; then echo "Local runner PID: $(cat log/local_runner.pid)"; fi
