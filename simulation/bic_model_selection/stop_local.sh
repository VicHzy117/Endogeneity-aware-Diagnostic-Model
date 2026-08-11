#!/usr/bin/env bash
set -euo pipefail

if [[ -f log/local_runner.pid ]]; then
  runner_pid=$(cat log/local_runner.pid)
  if kill -0 "${runner_pid}" 2>/dev/null; then
    kill -TERM "${runner_pid}" || true
    echo "Sent TERM to local runner PID ${runner_pid}."
  fi
fi
pkill -TERM -f '[R]script code/run_replicate_candidates.R' 2>/dev/null || true
pkill -TERM -f '[x]args -P .*run_replicate_candidates.R' 2>/dev/null || true
echo "Sent TERM to all package-scoped candidate workers. Completed .rds files remain resumable."
