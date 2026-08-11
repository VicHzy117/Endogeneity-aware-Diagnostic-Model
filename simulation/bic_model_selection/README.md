# EACDM exact-Q big model-selection simulation

This is a standalone server package. It reproduces the original
`simulation_github_package/bic_model_selection` design using the revised EACDM
sampler, the exact-Q response likelihood, and the corrected BIC.

## Design

- `n = 1000`.
- True dimensions: `K1 = K2 = 2, 3, 4`.
- Item blocks: `J1 = J2 = 24, 36`, so total `J = 48, 72`.
- 100 replicates for each of the six true scenarios.
- Candidate dimensions: `K1,K2 = 1,2,3,4,5` (25 fits per dataset).
- MCMC: 3,000 iterations with 2,000 burn-in iterations.
- Total: 600 datasets and 15,000 candidate fits.
- All items have three ordinal categories, encoded `0,1,2`.

The six required `n=1000` scenario files are generated with
`Delta = Q * beta`. They are excluded from Git; on a fresh clone,
`run_all.sh` recreates them from the fixed seeds before validation and fitting.

## Efficient task layout

The server launches 600 replicate tasks. Each R process compiles the C++ code
once and then fits all 25 candidate dimension pairs sequentially. Every
candidate is saved immediately in its own RDS file, so interruption and rerun
resume at the candidate level.

## Single 32-core server

From inside this folder:

```bash
nohup env \
  N_WORKERS=30 \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  bash run_all.sh > log/local_master.log 2>&1 &
```

Check progress:

```bash
bash progress.sh
tail -f log/local_master.log
```

The master log will stay quiet while workers are running. Individual logs are
`log/local_task_001.log` through `log/local_task_600.log`.
On a 30-worker machine, plan for roughly 12--24 hours; the actual time depends
mainly on CPU speed and the candidates with `K1` or `K2` equal to 5.

Stop safely:

```bash
bash stop_local.sh
```

Resume all unfinished or invalid replicate tasks:

```bash
nohup env N_WORKERS=30 bash rerun_missing.sh > log/rerun_missing.log 2>&1 &
```

## Slurm

If `sbatch` is available, `bash run_all.sh` automatically submits a 600-element
array capped at 30 simultaneous tasks, followed by an aggregation job.

## Outputs

Candidate fits are written as:

```text
result/bic_fits/scenario_02_n1000_J24_trueK2/replicate_001/fit_K1_1_K2_1.rds
```

Final summaries include:

- `result/summary/bic_all_fits.csv`
- `result/summary/bic_best_by_replicate.csv`
- `result/summary/bic_selection_summary.csv`
- `result/summary/bic_selected_model_counts.csv`
- `result/summary/bic_model_selection_table.tex`
- `result/plots/big_model_selection_heatmaps.pdf`

The repository also includes the lightweight manuscript subset for total
`J=72` under `result/j72_only/`. After all candidate fits finish, regenerate
that subset with:

```bash
Rscript code/aggregate_j72_only.R
```

Run `bash progress.sh` before downloading results. Completion should report
`15000/15000` candidate fits and `600/600` complete replicate tasks.
