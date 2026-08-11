# Exact-Q and BIC specification

- The effective measurement coefficient is `Delta = Q * beta`.
- If `q_jk = 0`, the corresponding effective coefficient is exactly zero in
  the response likelihood.
- Active magnitudes use the positive half-normal slab from the revised EACDM
  sampler.
- The complete-data likelihood includes the exogenous latent-class prevalence
  contribution `log(pi2)`.
- The BIC penalty includes `2^K2 - 1` free exogenous class-prevalence
  probabilities, in addition to item intercepts, active Q/loading terms, and
  structural coefficients.
- Every saved candidate result records and audits these invariants.
