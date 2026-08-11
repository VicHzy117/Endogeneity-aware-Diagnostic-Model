# Exact-Q EACDM versus conventional CDM

This standalone package reruns only the proposed EACDM part of Section 4.3 using the revised exact-Q algorithm. The original conventional-CDM results are bundled unchanged as the fixed paper baseline and are not rerun.

## Original design retained

- 100 datasets; `n = 1000`; `J1 = J2 = 24`; true `K1 = K2 = 3`.
- Every item has `M_j = 3`, with categories encoded `0, 1, 2`.
- No observed covariates.
- The first two cross-block attribute pairs have probabilities about 0.10/0.90; the third has about 0.40/0.60.
- Data seed 237 and the original replicate-specific seed formula are retained.
- EACDM candidates: `(K1,K2)` in `{2,3,4} x {2,3,4}` (900 fits).
- Conventional candidates: `K` in `{2,3,4,5,6}` (the original 500 completed fits are included under `baseline/conventional`).
- 3,000 iterations and 2,000 burn-in iterations for every fit.

## What is rerun

Only the 900 EACDM candidate fits are rerun. EACDM uses the new partially collapsed exact-Q sampler, and its complete likelihood/BIC now includes `pi2`. The 500 conventional-CDM fits retain the original implementation, BIC, Q estimates, and seeds so that the conventional side of the published comparison is unchanged.

For the revised EACDM fits, posterior-averaged complete-data BIC counts:

- one item intercept per item;
- an inclusion indicator and active magnitude for every active Q entry;
- `2^K2-1` free `pi2` probabilities;
- EACDM structural coefficients.

The original conventional outputs are used only as the previously reported benchmark, rather than being silently refitted under a different algorithm.

## Run on a 32-core / 64-GB server

```bash
cd compare_to_conventional_cdm
nohup env N_WORKERS=30 bash run_all.sh > log/local_master.log 2>&1 &
tail -f log/local_master.log
```

Monitor and audit:

```bash
watch -n 10 "find result/eacdm -name '*.rds' | wc -l"
Rscript code/audit_results.R 3000 2000
```

There are 900 new result files. Valid completed fits are skipped. The audit also checks that all 500 bundled conventional baseline files are present. To rerun only missing or invalid EACDM fits:

```bash
nohup env N_WORKERS=30 bash rerun_missing.sh > log/rerun_missing.log 2>&1 &
```

Selection counts, a paper-ready LaTeX table, BIC files, and headline results are written to `result/summary`. Modal-Q inclusion-frequency figures are written to `result/plots`.
