# EACDM Exact-Q Simulation Reproducibility Package

This repository contains the simulation code for the revised
endogeneity-aware cognitive diagnosis model (EACDM). It reproduces the
parameter-recovery, convergence-diagnostic, latent-dimension-selection, and
conventional-CDM comparison studies reported in the manuscript and supplement.

## Revised algorithm

The current implementation differs from the original public code in four
important ways:

- The Q-matrix enters the ordinal response likelihood through
  `Delta = Q * beta` (elementwise multiplication).
- An inactive entry, `q[j,k] = 0`, fixes `Delta[j,k]` to zero exactly; no
  inactive auxiliary loading is sampled.
- Q is updated by a collapsed exact-Q step, followed by a positive
  half-normal loading draw for each active entry.
- The exogenous-profile probabilities `pi2` are sampled and their contribution
  is included in the complete-data likelihood and BIC. The BIC penalty includes
  `2^K2 - 1` free `pi2` probabilities.

Accordingly, measurement recovery is summarized by `RMSE(Delta)` rather than
RMSE of an unrestricted loading matrix.

## Repository layout

```text
simulation/
  parameter_recovery/         # 18 scenarios x 100 replicates
  convergence_diagnostics/    # four-chain running/split-Rhat diagnostics
  bic_model_selection/        # candidate (K1,K2) selection
compare_to_conventional_cdm/  # exact-Q EACDM versus the fixed original baseline
figures/                      # lightweight paper/supplement figures
```

Each experiment is self-contained and has its own tested copy of the sampler.
This avoids hidden dependencies between server jobs and lets any experiment be
uploaded and run independently.

## Software

The scripts require R, a C++ compiler, and:

```r
install.packages(c("Rcpp", "RcppArmadillo"))
```

The run scripts automatically use Slurm when `sbatch` is available and use a
parallel single-server runner otherwise. If necessary, edit the R module line
in the relevant `slurm/*.sbatch` files.

## Reproducing the studies

Generated datasets, MCMC chains, replicate-level fits, and logs are not stored
in Git. On a fresh clone, each `run_all.sh` generates any missing data from the
fixed seeds, validates the data and sampler, and then starts or resumes the
analysis.

### 1. Parameter recovery

The design crosses `n = 500, 1000, 2000`, total `J = 48, 72`, and
`K1 = K2 = 2, 3, 4`, with 100 replicates per scenario.

```bash
cd simulation/parameter_recovery
N_WORKERS=30 bash run_all.sh
```

Key summaries are written to `result/summary/`, including the LaTeX table and
replicate-level ARI, `RMSE(Delta)`, and `RMSE(eta)` metrics.

### 2. Convergence diagnostics

Four chains are fitted to replicates 1, 50, and 100 in every parameter-recovery
scenario (216 chains total).

```bash
cd simulation/convergence_diagnostics
N_WORKERS=30 bash run_all.sh
```

The diagnostics and running-Rhat figures are written to `output/diagnostics/`
and `output/running_rhat/`.

### 3. Latent-dimension selection

For each simulated dataset, all 25 candidates in
`K1,K2 in {1,2,3,4,5}` are fitted. The full design contains 600 datasets and
15,000 candidate fits.

```bash
cd simulation/bic_model_selection
N_WORKERS=30 bash run_all.sh
```

The lightweight `result/j72_only/` folder contains the manuscript analysis for
`n = 1000` and total `J = 72`. Run
`Rscript code/aggregate_j72_only.R` after a complete rerun to regenerate it.

### 4. Comparison with a conventional CDM

This experiment reruns only the 900 revised EACDM candidate fits. The 500
conventional-CDM fit files in `baseline/conventional/` are the unchanged
original benchmark and are intentionally included in the repository.

```bash
cd compare_to_conventional_cdm
N_WORKERS=30 bash run_all.sh
```

Selection summaries, LaTeX output, and modal-Q inclusion-frequency figures are
written to `result/summary/` and `result/plots/`.

## Resuming and resource control

Valid completed result files are skipped, so rerunning the same command resumes
an interrupted job. Use `N_WORKERS` to control single-server parallelism and
`MAX_CONCURRENT` to control Slurm array concurrency. The experiment READMEs
contain progress, audit, stop, and missing-task commands.

## Included lightweight results

The repository retains CSV/TeX summaries and final figures but excludes large
regenerable objects. The included results correspond to the revised exact-Q
algorithm:

- parameter recovery: all 18 scenarios and 1,800 replicates;
- convergence diagnostics: 216 chains across 54 datasets;
- dimension selection at `n = 1000, J = 72`: `(K,K)` selected in all 100
  replicates for each true `K = 2, 3, 4`;
- model comparison: EACDM selected `(3,3)` in all 100 replicates, while the
  fixed conventional-CDM baseline selected `K = 4, 5, 6` in 65, 33, and 2
  replicates, respectively.

See each experiment's `ALGORITHM_NOTES.md` and `README.md` for the exact
formula-to-code mapping and server workflow.
