# Server Simulation Model Comparison

This folder is a self-contained server package for comparing the old CDM model
and the new EACDM model by BIC on simulated data.

## Simulation Design

- Number of datasets: 100
- Subjects per dataset: 1000
- True latent dimensions: `K1 = K2 = 3`
- Items: `J1 = J2 = 24`
- Response levels: 3 ordinal categories, saved as `0, 1, 2`
- True Q matrices: 24-item repeat of the K = 3 setting
- Structure equation from Section 4.3.1:

```r
logit Pr(alpha_ik^(1) = 1) = gamma_k^T alpha_i^(2)
```

Here `alpha_i^(2)` includes the intercept column. For `K1 = K2 = 3`, this
simulation uses a one-to-one dependency structure:

- `alpha1_1` depends on `alpha2_1`
- `alpha1_2` depends on `alpha2_2`
- `alpha1_3` depends on `alpha2_3`

```r
gamma <- matrix(c(
  -2.20, -2.20, -0.40,
   4.40,  0.00,  0.00,
   0.00,  4.40,  0.00,
   0.00,  0.00,  0.80
), nrow = 4, ncol = 3, byrow = TRUE)
```

For `alpha1_1` and `alpha1_2`, the probability is approximately 0.10 when the
corresponding `alpha2` attribute is 0 and 0.90 when it is 1. For `alpha1_3`,
the corresponding probabilities are approximately 0.40 and 0.60.

The data generator sets `set.seed(seed + dataset_id - 1)` for each replicate.
The default data seed is `237`, so rerunning the generator with the same
arguments reproduces the same 100 datasets. The model runners also set a
deterministic seed for each array task:

- conventional CDM: `seed + dataset_id * 100 + K`, with default runner seed `1000`
- EACDM: `seed + dataset_id * 100 + K1 * 10 + K2`, with default runner seed `2000`

No covariates are used in this simulation setting.

## Folder Layout

```text
eacdm_simulation_github/compare_to_conventional_cdm/
  ../src/
    eacdm_model.R
    eacdm_mcmc.cpp
    regular_cdm_model.R
    regular_cdm_mcmc.cpp
  data/
    generate_simulation_data.R
    simulation_data.rds
    simulation_data.RData
    true_Q_y.csv
    true_Q_v.csv
    true_gamma.csv
    simudata_original.R
  code/
    run_conventional_cdm_bic.R
    run_eacdm_bic.R
    aggregate_and_plot.R
  slurm/
    generate_data.sbatch
    run_conventional_cdm_bic_array.sbatch
    run_eacdm_bic_array.sbatch
    aggregate_and_plot.sbatch
  log/
  result/
```

The runners load the shared model implementations from `../src/`, so model
changes are made in one place.

## Run On Server

From the project folder:

```bash
sbatch slurm/generate_data.sbatch
sbatch slurm/run_conventional_cdm_bic_array.sbatch
sbatch slurm/run_eacdm_bic_array.sbatch
```

After both arrays finish:

```bash
sbatch slurm/aggregate_and_plot.sbatch
```

By default:

- conventional CDM tests `K = 2, 3, 4, 5, 6` for each dataset, so the array has `100 * 5 = 500` tasks.
- EACDM tests `K1 = 2, 3, 4` and `K2 = 2, 3, 4`, so the array has `100 * 9 = 900` tasks.
- MCMC settings are `iteration = 3000`, `burnin = 2000`.

To override MCMC length at submission time:

```bash
ITERATION=5000 BURNIN=4000 sbatch slurm/run_conventional_cdm_bic_array.sbatch
ITERATION=5000 BURNIN=4000 sbatch slurm/run_eacdm_bic_array.sbatch
```

## Outputs

Each model fit writes one `.rds` file:

```text
result/conventional_cdm/dataset_001_K_3.rds
result/eacdm/dataset_001_K1_3_K2_3.rds
```

Aggregation writes:

```text
result/summary/conventional_cdm_all_bic.csv
result/summary/eacdm_all_bic.csv
result/summary/conventional_cdm_best_k.csv
result/summary/eacdm_best_k.csv
result/summary/conventional_cdm_best_k_counts.csv
result/summary/eacdm_best_k_counts.csv
result/summary/model_comparison_summary.rds
```

Best-model Q plots are written per dataset:

```text
result/plots/dataset_001_conventional_cdm_best_Q.png
result/plots/dataset_001_eacdm_best_Q.png
```

The plots compare true Q against the posterior estimated Q. Estimated columns
are aligned to the true Q when the selected K matches the true K; otherwise the
estimated matrix is plotted with its selected number of columns.
