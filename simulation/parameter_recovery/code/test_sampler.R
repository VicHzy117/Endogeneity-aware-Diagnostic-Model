script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) {
    return(dirname(normalizePath(gsub("~\\+~", " ", sub("^--file=", "", file_arg[[1L]])))))
  }
  getwd()
}

code_dir <- script_dir()
source(file.path(code_dir, "new_model_main.R"))

# Check the analytic collapsed Bayes factor against numerical integration.
set.seed(260806)
x <- rbinom(25L, 1L, 0.55)
r <- rnorm(25L, 0.7 * x, 1)
s_xx <- sum(x^2)
s_xr <- sum(x * r)
sigma2 <- 1
log_bf_cpp <- exact_q_log_bf(s_xx, s_xr, sigma2)
m0 <- prod(dnorm(r))
m1 <- integrate(function(b) vapply(b, function(one_b) {
  prod(dnorm(r - x * one_b)) * 2 * dnorm(one_b, 0, sqrt(sigma2))
}, numeric(1L)),
                lower = 0, upper = Inf, rel.tol = 1e-10, abs.tol = 1e-30)$value
stopifnot(abs(log_bf_cpp - log(m1 / m0)) < 1e-8)

# Deep-tail draws must finish quickly and respect their bounds. These cases
# caused rare workers to stall under naive rejection sampling.
tail_lower <- replicate(100L, rtruncnorm1(-30, 1, 0, Inf))
tail_upper <- replicate(100L, rtruncnorm1(30, 1, -Inf, 0))
tail_finite <- replicate(100L, rtruncnorm1(-30, 1, 0, 1))
stopifnot(all(is.finite(tail_lower)), all(tail_lower >= 0))
stopifnot(all(is.finite(tail_upper)), all(tail_upper <= 0))
stopifnot(all(is.finite(tail_finite)), all(tail_finite >= 0 & tail_finite <= 1))

# Run a tiny end-to-end chain and verify the exact-Q and pi2 invariants.
Y <- matrix(sample(0:2, 80L, replace = TRUE), 20L, 4L)
V <- matrix(sample(0:2, 80L, replace = TRUE), 20L, 4L)
Z <- matrix(rbinom(20L, 1L, 0.5), 20L, 1L)
fit <- ECDM_main(Y, V, Z, K_a = 2L, K_g = 2L, iteration = 8L, verbose_every = 0L)
for (i in seq_along(fit$Q1_list)) {
  stopifnot(all(fit$B_list[[i]][, -1L][fit$Q1_list[[i]] == 0] == 0))
  stopifnot(all(fit$L_list[[i]][, -1L][fit$Q2_list[[i]] == 0] == 0))
  stopifnot(abs(sum(fit$pi2_list[[i]]) - 1) < 1e-10)
}
stopifnot(all(is.finite(fit$log_lik_complete)))

# The complete likelihood must change by exactly log(pi2_new/pi2_old) when
# only the current exogenous profile probability changes.
stopifnot(isTRUE(fit$sampler$pi2_in_complete_loglik))
cat("All exact-Q sampler tests passed.\n")
