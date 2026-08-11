script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) return(dirname(normalizePath(sub("^--file=", "", file_arg[[1L]]))))
  getwd()
}

source(file.path(script_dir(), "new_model_main.R"))
set.seed(260806)

check_fit <- function(K1, K2) {
  n <- 18L
  J <- 4L
  pattern <- rep(0:2, length.out = n * J)
  Y <- matrix(sample(pattern), nrow = n, ncol = J)
  V <- matrix(sample(pattern), nrow = n, ncol = J)
  Z <- matrix(rbinom(n, 1L, 0.5), ncol = 1L)
  fit <- ECDM_main(
    Y, V, Z, K_a = K1, K_g = K2, iteration = 6L,
    verbose_every = 0L
  )
  for (i in seq_along(fit$Q1_list)) {
    stopifnot(all(fit$B_list[[i]][, -1L, drop = FALSE][fit$Q1_list[[i]] == 0] == 0))
    stopifnot(all(fit$L_list[[i]][, -1L, drop = FALSE][fit$Q2_list[[i]] == 0] == 0))
    stopifnot(abs(sum(fit$pi2_list[[i]]) - 1) < 1e-10)
  }
  stopifnot(length(fit$final$pi2) == 2^K2)
  stopifnot(isTRUE(fit$sampler$pi2_in_complete_loglik))
  bic <- PBIC_from_result(fit, Y, V, Z, K1, K2, burn_in = 3L)
  stopifnot(is.finite(bic))
  invisible(TRUE)
}

# Explicitly test the two candidate-grid boundaries.
check_fit(1L, 1L)
check_fit(5L, 5L)
cat("Exact-Q sampler and corrected-BIC tests passed for K=1 and K=5.\n")
