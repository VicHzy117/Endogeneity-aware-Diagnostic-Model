# Revised algorithm: formula-to-code map

## Exact measurement likelihood

For block `d`, item `j`, and attribute `k`, the code stores the effective
loading

```text
delta[d,j,k] = q[d,j,k] * beta[d,j,k].
```

`new_model_mcmc.cpp::f_Q_Beta_omega()` enforces `delta = 0` whenever `q = 0`.
The class likelihood therefore uses only active Q entries.

## Collapsed Q update

For the coordinate being updated, define

```text
r_i = ystar_ij - beta_j0 - sum_{h != k} delta_jh * alpha_ih
Sxx = sum_i alpha_ik^2
Sxr = sum_i alpha_ik * r_i
A   = Sxx + 1 / sigma_beta^2.
```

With a positive half-normal slab, the log marginal-likelihood ratio used by
the code is

```text
log(m1 / m0) = log(2)
              - 0.5 * log(sigma_beta^2 * A)
              + 0.5 * Sxr^2 / A
              + log Phi(Sxr / sqrt(A)).
```

The posterior inclusion log odds are

```text
logit Pr(q_jk = 1 | rest) = logit(omega_d) + log(m1 / m0).
```

After a one is drawn, the active magnitude is sampled from

```text
N_+(Sxr / A, 1 / A).
```

After a zero is drawn, its effective loading is set to zero immediately. The
intercept is updated after all loading coordinates for that item, giving the
coordinate-wise partially collapsed sweep described in the revised paper.

## Latent profiles and pi2

The implementation uses two exact Gibbs substeps for the complete profile:

1. update `alpha1` conditional on current `alpha2` using its measurement block
   and the logistic structural probability;
2. update `alpha2` conditional on current `alpha1` using its measurement block,
   the structural probability, and the current `pi2`.

These two conditional updates have the same joint posterior target as drawing
the complete profile in one `2^(K1+K2)` multinomial step, while avoiding that
larger enumeration. Then

```text
pi2 | alpha2 ~ Dirichlet(1 + exogenous-profile counts).
```

## Complete-data BIC contribution

For each retained draw and subject, `complete_loglik()` stores

```text
log p(Y_i | alpha_i, Delta)
+ log p(alpha1_i | alpha2_i, Z_i, eta)
+ log pi2[alpha2_i].
```

The final term is the corrected `pi2` contribution. The implementation keeps
log likelihoods directly, avoiding probability-product underflow. The BIC
calculation averages each subject's complete-data log likelihood over retained
draws before summing across subjects.

## Recovery targets

- Q recovery: posterior mean Q, thresholded at 0.5, blockwise label aligned,
  then ARI computed on the block-diagonal row profiles.
- Measurement recovery: RMSE of the effective `J x K` loading matrix `Delta`,
  excluding item intercepts, after the same blockwise label alignment. Item
  intercept recovery is stored separately as `RMSE_beta0`.
- Structural recovery: RMSE of `eta` after applying the corresponding row and
  column permutations.
