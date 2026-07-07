#!/bin/bash
set -euo pipefail

data_job=$(sbatch slurm/generate_data.sbatch | awk '{print $4}')
chain_job=$(sbatch --dependency=afterok:${data_job} slurm/run_chains_array.sbatch | awk '{print $4}')
diagnostic_job=$(sbatch --dependency=afterany:${chain_job} slurm/diagnose_chains.sbatch | awk '{print $4}')

echo "Submitted data job ${data_job}."
echo "Submitted 216-chain array ${chain_job} after data generation."
echo "Submitted convergence diagnostics ${diagnostic_job} after all chain tasks finish."
