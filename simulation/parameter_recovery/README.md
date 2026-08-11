# EACDM exact-Q parameter recovery

This self-contained folder reruns the manuscript's parameter-recovery
simulation with the revised exact-Q EACDM algorithm. It does not modify or
source the original implementation.

## What changed

The implementation targets the revised posterior:

- `Q` is part of the measurement likelihood through the effective loading
  `Delta = Q * B_A` (elementwise product).
- If `q[j,k] = 0`, then `Delta[j,k] = 0` exactly. There is no inactive
  auxiliary coefficient in the likelihood.
- Each `q[j,k]` is updated with the collapsed marginal likelihood from the
  positive half-normal slab, followed immediately by an active loading draw
  when the indicator is one.
- The exogenous profile probabilities `pi2` have a Dirichlet update.
- The posterior-averaged complete-data BIC includes
  `log(pi2[alpha2_i])`, which was absent from the previous implementation.
- Measurement recovery is reported as `RMSE(Delta)`, not RMSE of inactive
  auxiliary coefficients.

See `ALGORITHM_NOTES.md` for the formula-to-code mapping.

## Simulation design

- `n = 500, 1000, 2000`
- `J1 = J2 = 24, 36`, hence total `J = 48, 72`
- `K1 = K2 = 2, 3, 4`
- 100 independent replicates for every combination
- 18 scenarios and 1,800 fits in total
- three ordinal categories coded `0, 1, 2`
- 3,000 MCMC iterations with 2,000 burn-in iterations
- deterministic data and fit seeds
- summaries: median and IQR of `ARI(Q)`, `RMSE(Delta)`, and `RMSE(eta)`

The true structural matrices for `K = 2, 3, 4` are implemented in
`data/generate_simulation_data.R` and match the supplementary material.

## Data generation

The revised estimation changes the likelihood restriction and posterior
sampler, not the data-generating distribution. The earlier responses can
therefore be reused if available. To keep the Git repository small, generated
RDS files are not tracked. On a fresh clone, `run_all.sh` recreates all 18
scenario files from the fixed seeds and validates them before fitting.

## Folder layout

```text
parameter_recovery/
  README.md
  ALGORITHM_NOTES.md
  run_all.sh
  rerun_missing.sh
  code/
    new_model_mcmc.cpp
    new_model_main.R
    run_one_replicate.R
    aggregate_table1.R
    validate_data.R
    audit_results.R
    test_sampler.R
  data/
    generate_simulation_data.R
    generated/                 # created locally; excluded from Git
  slurm/
    run_replicates_array.sbatch
    aggregate_table1.sbatch
    generate_data.sbatch       # optional only
  log/
  result/
    fits/
    summary/
```

## Run on the server

From the repository root, enter this folder and run:

```bash
cd simulation/parameter_recovery
bash run_all.sh
```

`run_all.sh` automatically detects the server type:

- when `sbatch` is available, it validates the data, runs sampler tests,
  generates missing data, submits the 1,800-task Slurm array, and submits
  aggregation with an `afterok` dependency;
- when Slurm is unavailable, it runs the same 1,800 fits through
  `run_local_parallel.sh` with up to four local workers by default.

Slurm array concurrency defaults to 100 and can be changed at submission:

```bash
MAX_CONCURRENT=50 bash run_all.sh
```

On a single server, choose the local worker count according to available CPU
and memory:

```bash
N_WORKERS=30 bash run_all.sh
```

For a long single-server run that should continue after disconnecting SSH:

```bash
nohup env N_WORKERS=30 bash run_all.sh > log/local_master.log 2>&1 &
```

Monitor progress with:

```bash
tail -f log/local_master.log
find result/fits -name 'replicate_*.rds' | wc -l
```

The scripts load `R/4.4.2-gfbf-2024a` and require the R packages `Rcpp` and
`RcppArmadillo`. Edit the module line in the three files under `slurm/` if the
server uses another R module name.

Optional longer chains can be requested without editing code:

```bash
ITERATION=5000 BURNIN=4000 bash run_all.sh
```

To overwrite and regenerate data intentionally:

```bash
REGENERATE_DATA=true bash run_all.sh
```

## Resume after failed array tasks

Completed result files are skipped automatically. To audit all expected files
and submit only missing or invalid task IDs:

```bash
bash rerun_missing.sh
```

After reruns finish, submit the audit and aggregation job again:

```bash
sbatch slurm/aggregate_table1.sbatch
```

Aggregation stops unless all 1,800 results are present, unique, finite, and
satisfy the exact-Q invariant.

## Outputs

Each fit is saved as:

```text
result/fits/scenario_01_n500_J24_K2/replicate_001.rds
```

Final summaries are written to:

```text
result/summary/all_replicate_metrics.csv
result/summary/table1_extended.csv
result/summary/table1_extended.md
result/summary/table1_extended_summary.rds
```

## Local checks

```bash
Rscript code/test_sampler.R
Rscript code/run_one_replicate.R \
  --task_id=1 --iteration=100 --burnin=50 \
  --data_dir=data/generated --result_dir=/tmp/eacdm_exactq_smoke \
  --overwrite=true
Rscript code/aggregate_table1.R \
  --result_dir=/tmp/eacdm_exactq_smoke --allow_incomplete=true
```
