# EACDM Simulation Convergence Diagnostics

This server package evaluates MCMC convergence for the Section 4 simulation study using multiple independent chains, rank-normalized split-Rhat, and trace plots.

## Diagnostic Design

The study covers all 18 simulation scenarios:

- `n = 500, 1000, 2000`
- `J1 = J2 = 24, 36`
- `K1 = K2 = 2, 3, 4`

For each scenario, replicates `1`, `50`, and `100` are selected in advance. Four independent chains are run for each selected dataset:

```text
18 scenarios * 3 replicates * 4 chains = 216 array tasks
```

Each chain uses the same MCMC length reported in the paper:

```text
iteration = 3000
burn-in = 2000
post-burn-in draws = 1000
```

The selected datasets reproduce the original parameter-recovery simulation exactly because the generator uses the same data seed and replicate-specific seed formula.

## Diagnostics

Before computing diagnostics, each chain is aligned to the true simulation labels using the posterior mean Q-matrices. The resulting fixed attribute permutations are applied consistently to:

- `Q1` and `B1`
- `Q2` and `B2`
- rows and columns of the structural coefficient matrix `eta`

The diagnostic script computes rank-normalized split-Rhat, including the folded-Rhat check, for every element of:

- `eta`
- `B1`
- `B2`
- `Q1`
- `Q2`

It reports parameter-level results and summaries by replicate, scenario, and parameter block.

## Trace Plots

Trace plots are generated for replicate 50 in three scenarios selected before examining the diagnostics:

- Difficult: scenario 13, `n = 500`, total `J = 48`, `K = 4`
- Intermediate: scenario 11, `n = 1000`, total `J = 72`, `K = 3`
- Easier: scenario 6, `n = 2000`, total `J = 72`, `K = 2`

For each scenario, the package produces:

- continuous-parameter trace plots for selected `eta`, `B1`, and `B2` elements;
- binary Q-element trace plots;
- running posterior means for the selected Q-elements.

The trace-plot choices are based on the true active/inactive structure and are not selected after examining convergence.

## Folder Layout

```text
eacdm_simulation_github/simulation/convergence_diagnostics/
  ../../src/
    eacdm_model.R
    eacdm_mcmc.cpp
  code/
    run_chain.R
    diagnose_chains.R
  data/
    generate_diagnostic_data.R
    generated/                  # created on the server
  slurm/
    generate_data.sbatch
    run_chains_array.sbatch
    diagnose_chains.sbatch
  output/                       # created on the server
  run_all.sh
```

The chain runner loads the shared EACDM implementation from `../../src/`.

## Run

Upload the complete folder, enter it on the server, and run:

```bash
cd simulation/convergence_diagnostics
bash run_all.sh
```

The wrapper submits data generation, then the 216-task chain array, and finally the dependent diagnostic job. The diagnostic job uses an `afterany` dependency so that it still produces `missing_chains.csv` if an individual chain fails. Complete four-chain datasets are diagnosed, while incomplete datasets are listed with their array task IDs for rerunning.

To inspect progress:

```bash
squeue -u "$USER"
```

If `output/diagnostics/missing_chains.csv` lists failed tasks, rerun the indicated task IDs and then resubmit the diagnostic job:

```bash
sbatch --array=TASK_ID slurm/run_chains_array.sbatch
sbatch slurm/diagnose_chains.sbatch
```

## Reproducibility

Default seeds are:

- Data base seed: `20260517`
- Chain base seed: `20260610`
- Chain seed:

```text
20260610 + scenario_id * 100000 + replicate_id * 100 + chain_id
```

All Slurm jobs send `END` and `FAIL` notifications to `zhuang@fredhutch.org`.

## Main Outputs

```text
output/diagnostics/parameter_rhat.csv
output/diagnostics/replicate_block_summary.csv
output/diagnostics/scenario_block_summary.csv
output/diagnostics/overall_block_summary.csv
output/diagnostics/missing_chains.csv
output/diagnostics/convergence_diagnostics.rds
output/traceplots/*.pdf
output/traceplots/traceplot_manifest.csv
```

The most useful supplementary table is `scenario_block_summary.csv`. The trace-plot PDFs can be included directly in the Supplementary Material.
