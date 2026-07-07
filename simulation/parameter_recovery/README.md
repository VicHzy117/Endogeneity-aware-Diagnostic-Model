# EACDM Table 1 Simulation Package

This folder reruns the Section 4.1/4.2 simulation for the new EACDM model and
extends the true dimensions to `K1 = K2 = 2, 3, 4`.

## Design

- `n = 500, 1000, 2000`
- `J1 = J2 = 24, 36`, so total `J = 48, 72`
- `K1 = K2 = 2, 3, 4`
- 100 replicates per scenario
- 3 ordinal response categories encoded as `0, 1, 2`
- MCMC defaults: 3000 iterations, first 2000 discarded as burn-in, matching
  the conventional-CDM comparison
- Metrics: `ARI(Q)`, `RMSE(B)`, and `RMSE(eta)`, summarized by median and IQR

For `K = 3`, the structural coefficient matrix `eta` follows the values printed
in Section 4.1, including one binary covariate row. For `K = 2` and `K = 4`, the
same pattern is truncated or extended in the generator.

Seeds are deterministic:

- Data seed default: `20260517`
- Dataset seed: `seed + n * 100000 + J_block * 1000 + K * 100 + replicate_id`
- Fit seed: `910000 + scenario_id * 100000 + replicate_id`

## Folder Layout

```text
eacdm_simulation_github/simulation/parameter_recovery/
  ../../src/
    eacdm_model.R
    eacdm_mcmc.cpp
  data/
    generate_simulation_data.R
    generated/                 # created by the data-generation job
  code/
    run_one_replicate.R
    aggregate_table1.R
  slurm/
    generate_data.sbatch
    run_replicates_array.sbatch
    aggregate_table1.sbatch
  log/
  result/
    fits/
    summary/
  run_all.sh
```

The fitting script loads the shared EACDM implementation from `../../src/`.

## Run On Server

From this folder:

```bash
sbatch slurm/generate_data.sbatch
sbatch slurm/run_replicates_array.sbatch
sbatch slurm/aggregate_table1.sbatch
```

Or submit the dependent pipeline:

```bash
bash run_all.sh
```

Override MCMC length when needed:

```bash
ITERATION=5000 BURNIN=4000 sbatch slurm/run_replicates_array.sbatch
```

## Outputs

Each fit writes one result file:

```text
result/fits/scenario_01_n500_J24_K2/replicate_001.rds
```

Aggregation writes:

```text
result/summary/all_replicate_metrics.csv
result/summary/table1_extended.csv
result/summary/table1_extended.md
result/summary/table1_extended_summary.rds
```
