# Exact-Q parameter-recovery results

## Integrity checks

- Expected fits: 1,800.
- Valid fits: 1,800.
- Missing fits: 0.
- Invalid fits: 0.
- Exact-Q invariant failures: 0.
- Every scenario contains 100 replicates.

## Main findings

The revised sampler shows coherent parameter-recovery behavior across all 18
simulation scenarios. For every fixed combination of item length and latent
dimension, the median ARI increases monotonically with sample size, whereas
the median RMSE values for both the effective measurement loadings
`Delta = Q * B_A` and the structural coefficients `eta` decrease monotonically
with sample size.

At `n = 2000`, median ARI values range from 0.947 to 0.967. Thus, the exact-Q
implementation achieves near-perfect Q-matrix recovery in the largest sample,
although the former statement that every median ARI equals 1.000 is not
supported by the rerun.

Measurement recovery is strong. For `K = 3`, median RMSE(Delta) decreases from
0.090 to 0.042 when `J = 48` and `n` increases from 500 to 2000; for `J = 72`,
it decreases from 0.081 to 0.039. The longer test produces a lower
RMSE(Delta) in every matched `(n, K)` setting.

Structural coefficients remain more difficult to estimate as `K` increases,
but their recovery improves consistently with additional information. For
example, when `K = 4` and `J = 72`, median RMSE(eta) decreases from 0.263 at
`n = 500` to 0.140 at `n = 2000`.

The effect of increasing `J` on ARI is positive in most, but not all, matched
settings. Two small reversals occur for `(K = 2, n = 2000)` and
`(K = 4, n = 1000)`. They are not accompanied by worse continuous-parameter
recovery: RMSE(Delta) and RMSE(eta) still improve with the longer test.

## Comparison with the former table

The exact-Q rerun produces median ARI values that are lower than the former
table by approximately 0.03 to 0.07 across the 18 scenarios. The manuscript's
old Q-recovery values and its claim of median ARI equal to 1.000 at `n = 2000`
should therefore be replaced.

Median RMSE(eta) is essentially unchanged: across scenarios, the difference
from the former table ranges from approximately -0.006 to 0.000. This supports
the stability of the substantive structural-parameter recovery conclusion.

The new RMSE(Delta) values should not be interpreted as a direct like-for-like
improvement over the old RMSE(B) values. Under the revised likelihood, Delta
is the effective `J x K` loading matrix and inactive auxiliary coefficients are
not model parameters; consequently RMSE(Delta) is the appropriate estimand.

## Replicate-level diagnostics

Across all 1,800 fits, ARI ranges from 0.541 to 1.000, with an overall median
of 0.917; no replicate has ARI below 0.5. The lowest values are concentrated in
the smallest-information settings, especially `n = 500, J = 48`. RMSE(Delta)
ranges from 0.029 to 0.127, and RMSE(eta) ranges from 0.036 to 0.437. The
largest structural errors occur primarily when `n = 500` and `K = 4`, which is
consistent with the larger number of latent classes and structural
coefficients in that setting. These tails support reporting medians and IQRs
rather than means alone.

## Suggested manuscript text

Table 1 reports parameter-recovery results under the exact-Q likelihood. The
proposed method recovered the block-specific Q-matrices increasingly well as
the sample size grew. Across the six settings with n = 2000, the median ARI
ranged from 0.947 to 0.967, indicating near-perfect recovery, and the IQRs were
generally smaller than those obtained at n = 500. Recovery of the effective
measurement loadings also improved with increasing information. For K = 3,
the median RMSE of Delta decreased from 0.090 to 0.042 as n increased from 500
to 2000 when J = 48, and from 0.081 to 0.039 when J = 72. Structural
coefficients were more difficult to estimate at larger latent dimensions, but
their RMSE decreased consistently with sample size. For example, for K = 4
and J = 72, the median RMSE of eta decreased from 0.263 at n = 500 to 0.140 at
n = 2000. Overall, these results support accurate recovery of the measurement
and directed structural components under the revised exact-Q formulation.
