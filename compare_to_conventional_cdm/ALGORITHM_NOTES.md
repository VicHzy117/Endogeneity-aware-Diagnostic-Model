# Comparison decisions

1. The data-generating mechanism and candidate grids are copied from the original GitHub simulation package.
2. The generated responses remain valid because the DGP already used `Delta = Q * beta`; no data-model mismatch is introduced by exact-Q inference.
3. The 900 revised EACDM fits enforce `Delta_jk = 0` whenever `q_jk = 0` and use a positive half-normal active slab.
4. The revised EACDM sampler uses the tail-stable truncated-normal routine.
5. Revised EACDM BIC contains `p(alpha2 | pi2)` and counts the `2^K2-1` free prevalence parameters.
6. The 500 conventional-CDM outputs are copied unchanged from the completed original comparison and are not rerun or redefined.
7. The old EACDM burn-in off-by-one is not carried forward.
