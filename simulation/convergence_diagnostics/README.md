# Exact-Q convergence diagnostics

This is a standalone rerun of the paper's convergence-diagnostic design using the revised exact-Q sampler. It does not modify or source the old simulation repository.

## Design

- 18 parameter-recovery scenarios: `n = 500, 1000, 2000`; `J1 = J2 = 24, 36`; `K1 = K2 = 2, 3, 4`.
- Prespecified replicates 1, 50, and 100 from each scenario.
- Four independent chains per dataset: 54 datasets x 4 = 216 chain jobs.
- 3,000 MCMC iterations with 2,000 burn-in iterations.
- All items have `M_j = 3`, encoded as `0, 1, 2`.
- Each chain is aligned to the true Q-matrix separately within the two blocks before diagnostics.
- Final diagnostics use rank-normalized split-Rhat with folded Rhat for `eta`, `Delta1`, `Delta2`, `Q1`, and `Q2`.
- The original running classic-Rhat figure design and the selected traceplot scenarios (13, 11, 6; replicate 50) are retained.

The 54 datasets are not stored in Git. On a fresh clone, `run_all.sh` calls
`data/generate_diagnostic_data.R` and reproduces the same response values from
the fixed deterministic seeds before validating them.

## Single server (32 cores / 64 GB)

```bash
cd simulation/convergence_diagnostics
nohup env N_WORKERS=30 bash run_all.sh > log/local_master.log 2>&1 &
tail -f log/local_master.log
```

The threaded numerical-library variables are forced to one thread per worker. Valid completed chains are skipped, so the same command resumes safely. To rerun only missing/invalid tasks:

```bash
nohup env N_WORKERS=30 bash rerun_missing.sh > log/rerun_missing.log 2>&1 &
```

Monitor completion:

```bash
watch -n 10 "find output/chains -name 'chain_*.rds' | wc -l"
Rscript code/audit_chains.R 3000 2000
```

Final files are under `output/diagnostics`, `output/traceplots`, and `output/running_rhat`.
