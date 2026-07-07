# Shared Model Implementations

This folder contains the model code used by all simulation workflows.

- `eacdm_model.R` and `eacdm_mcmc.cpp`: proposed EACDM sampler and helper functions.
- `regular_cdm_model.R` and `regular_cdm_mcmc.cpp`: conventional single-block CDM sampler used in the comparison study.

Experiment folders call these shared implementations directly. This avoids
duplicating sampler code across parameter recovery, BIC selection, convergence
diagnostics, and the conventional-CDM comparison.
