source(file.path("code", "new_model_main.R"))

set.seed(81206)
tail_draws <- replicate(100L, rtruncnorm1(0, 1, 8, Inf))
stopifnot(all(is.finite(tail_draws)), all(tail_draws >= 8))

n <- 24L
Y <- matrix(sample(0:2, n * 5L, replace = TRUE), nrow = n)
V <- matrix(sample(0:2, n * 5L, replace = TRUE), nrow = n)
Z <- matrix(rbinom(n, 1L, 0.5), ncol = 1L)
fit <- ECDM_main(Y, V, Z, K_a = 2L, K_g = 2L, iteration = 8L,
                 verbose_every = 0L, keep_loglik = FALSE, keep_pi2 = FALSE)
stopifnot(is.null(fit$log_lik_complete), is.null(fit$pi2_list))
for (i in seq_along(fit$Q1_list)) {
  stopifnot(all(fit$B_list[[i]][, -1L][fit$Q1_list[[i]] == 0L] == 0))
  stopifnot(all(fit$L_list[[i]][, -1L][fit$Q2_list[[i]] == 0L] == 0))
}
cat("Exact-Q convergence sampler tests passed.\n")
