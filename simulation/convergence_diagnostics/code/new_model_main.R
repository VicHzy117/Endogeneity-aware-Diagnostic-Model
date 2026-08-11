library(Rcpp)
library(RcppArmadillo)

# Locate this script and compile the C++ sampler next to it. This keeps the
# project portable and avoids hard-coded local machine paths.
this_file <- tryCatch(
  normalizePath(sys.frame(1)$ofile),
  error = function(e) NA_character_
)
if (is.na(this_file)) {
  cmd_file <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  script_arg <- if (length(cmd_file)) {
    gsub("~\\+~", " ", sub("^--file=", "", cmd_file[[1]]))
  } else {
    NA_character_
  }
  this_file <- if (!is.na(script_arg)) normalizePath(script_arg) else NA_character_
}
script_dir <- if (is.na(this_file)) getwd() else dirname(this_file)
sourceCpp(file.path(script_dir, "new_model_mcmc.cpp"))

NumResponse <- function(Y) {
  apply(Y, 2, max) + 1L
}

make_binary_design <- function(K) {
  states <- 0:(2^K - 1L)
  bits <- vapply(states, function(x) as.integer(intToBits(x))[seq_len(K)], integer(K))
  cbind(1, t(bits))
}

make_thresholds <- function(M) {
  J <- length(M)
  tau <- matrix(0, nrow = J, ncol = max(M) + 1L)
  for (j in seq_len(J)) {
    tau[j, seq_len(M[j] + 1L)] <- c(-Inf, seq_len(M[j] - 1L) - 1L, Inf)
  }
  tau
}

draw_initial_classes <- function(n, design) {
  n_class <- nrow(design)
  idx <- sample.int(n_class, n, replace = TRUE) - 1L
  counts <- tabulate(idx + 1L, nbins = n_class)
  list(index = idx, counts = counts, design_rows = design[idx + 1L, , drop = FALSE])
}

ECDM_main <- function(Y, V, covarites, K_a, K_g, iteration,
                      verbose_every = 1000L,
                      keep_categories = FALSE,
                      keep_loglik = TRUE,
                      keep_pi2 = TRUE,
                      sigma_beta2 = 1,
                      sigma_intercept2 = 1) {
  n <- nrow(Y)
  J_y <- ncol(Y)
  J_v <- ncol(V)

  # Build the latent class design matrices and ordinal thresholds once.
  M_y <- NumResponse(Y)
  M_v <- NumResponse(V)
  A <- make_binary_design(K_a)
  G <- make_binary_design(K_g)
  thres_y <- make_thresholds(M_y)
  thres_v <- make_thresholds(M_v)

  # Exact-Q initialization: B_i and L_i are effective loading matrices Delta,
  # so every inactive coefficient starts and remains exactly zero.
  B_i <- matrix(0, J_y, K_a + 1L)
  Q_MH_1 <- matrix(rbinom(J_y * K_a, 1L, 0.4), J_y, K_a)
  B_i[, -1L] <- Q_MH_1 * matrix(abs(rnorm(J_y * K_a, 0, sqrt(sigma_beta2))), J_y, K_a)
  omega1 <- 0.4

  L_i <- matrix(0, J_v, K_g + 1L)
  Q_MH_2 <- matrix(rbinom(J_v * K_g, 1L, 0.4), J_v, K_g)
  L_i[, -1L] <- Q_MH_2 * matrix(abs(rnorm(J_v * K_g, 0, sqrt(sigma_beta2))), J_v, K_g)
  omega2 <- 0.4

  init_a <- draw_initial_classes(n, A)
  i_c_a <- init_a$index
  n_cate_a <- init_a$counts
  A_cate <- init_a$design_rows

  init_g <- draw_initial_classes(n, G)
  i_c_g <- init_g$index
  n_cate_g <- init_g$counts
  G_cate <- init_g$design_rows
  pi_g <- rep(1 / nrow(G), nrow(G))
  G_catecov <- cbind(G_cate, covarites)

  # Regression from gamma/covariates to alpha, sampled with Polya-Gamma weights.
  Sita <- matrix(rnorm(ncol(G_catecov) * K_a, 0.01, 1), ncol(G_catecov), K_a)
  Means_prior <- matrix(0, nrow(Sita), K_a)
  Cov_prior <- array(0, c(nrow(Sita), nrow(Sita), K_a))
  for (k in seq_len(K_a)) Cov_prior[, , k] <- diag(nrow(Sita))
  W <- matrix(0.5, K_a, n)

  # Preallocate storage. This is much faster than repeatedly growing lists.
  Q1_list <- vector("list", iteration + 1L)
  Q2_list <- vector("list", iteration + 1L)
  B_list <- vector("list", iteration + 1L)
  L_list <- vector("list", iteration + 1L)
  Sita_list <- vector("list", iteration + 1L)
  if (keep_pi2) pi2_list <- vector("list", iteration + 1L)
  Q1_list[[1]] <- Q_MH_1
  Q2_list[[1]] <- Q_MH_2
  B_list[[1]] <- B_i
  L_list[[1]] <- L_i
  Sita_list[[1]] <- Sita
  if (keep_pi2) pi2_list[[1]] <- pi_g

  if (keep_categories) {
    i_a_list <- vector("list", iteration + 1L)
    i_g_list <- vector("list", iteration + 1L)
    i_a_list[[1]] <- i_c_a
    i_g_list[[1]] <- i_c_g
  }

  if (keep_loglik) {
    log_lik_complete <- matrix(NA_real_, n, iteration)
    active_q_count <- integer(iteration)
  }

  for (iter in seq_len(iteration)) {
    if (verbose_every > 0L && iter %% verbose_every == 0L) message("iteration ", iter)

    # 1. Sample alpha class and latent normal Y*.
    y_step <- f_alp_Ystar_pi(
      n_cate_a, i_c_a, A_cate, A, B_i, thres_y, Y, M_y, Sita, G_cate, G_catecov
    )
    n_cate_a <- y_step$n_cate
    i_c_a <- y_step$i_cate
    A_cate <- y_step$A_cate
    Y_star_new <- y_step$Y_star_gibbs

    # 2. Sample gamma class and latent normal V*.
    v_step <- f_gam_Vstar_pi(
      n_cate_g, i_c_g, G_cate, G, L_i, thres_v, V, M_v, Sita,
      A_cate, covarites, pi_g
    )
    n_cate_g <- v_step$n_cate
    i_c_g <- v_step$i_cate
    G_cate <- v_step$G_cate
    V_star_new <- v_step$V_star_gibbs
    pi_g <- v_step$pi_gibbs
    G_catecov <- cbind(G_cate, covarites)

    # 3. Update Sita using Polya-Gamma augmentation.
    sita_step <- f_Sitacoef_w(A_cate, W, G_cate, Sita, Means_prior, Cov_prior, G_catecov)
    W <- sita_step$W
    Sita <- sita_step$Sita

    # 4. Update Q/Beta for the Y block.
    q1_step <- f_Q_Beta_omega(
      A_cate, B_i, Y_star_new, sigma_beta2, sigma_intercept2, omega1, Q_MH_1
    )
    Q_MH_1 <- q1_step$Q_MH
    B_i <- q1_step$B_i
    omega1 <- q1_step$omega

    # 5. Update Q/Beta for the V block.
    q2_step <- f_Q_Beta_omega(
      G_cate, L_i, V_star_new, sigma_beta2, sigma_intercept2, omega2, Q_MH_2
    )
    Q_MH_2 <- q2_step$Q_MH
    L_i <- q2_step$B_i
    omega2 <- q2_step$omega

    Q1_list[[iter + 1L]] <- Q_MH_1
    Q2_list[[iter + 1L]] <- Q_MH_2
    B_list[[iter + 1L]] <- B_i
    L_list[[iter + 1L]] <- L_i
    Sita_list[[iter + 1L]] <- Sita
    if (keep_pi2) pi2_list[[iter + 1L]] <- pi_g
    if (keep_categories) {
      i_a_list[[iter + 1L]] <- i_c_a
      i_g_list[[iter + 1L]] <- i_c_g
    }

    if (keep_loglik) {
      ll <- complete_loglik(
        Y, V, i_c_a, i_c_g, A, G, B_i, L_i, M_y, M_v,
        thres_y, thres_v, Sita, G_catecov, A_cate, pi_g
      )
      log_lik_complete[, iter] <- ll$log_lik_complete
      active_q_count[iter] <- sum(Q_MH_1) + sum(Q_MH_2)
    }
  }

  out <- list(
    Q1_list = Q1_list,
    Q2_list = Q2_list,
    B_list = B_list,
    L_list = L_list,
    Sita_list = Sita_list,
    final = list(pi2 = pi_g, A = A, G = G),
    sampler = list(
      likelihood = "exact Q restriction",
      slab = "positive half-normal",
      sigma_beta2 = sigma_beta2,
      sigma_intercept2 = sigma_intercept2,
      pi2_in_complete_loglik = TRUE
    )
  )
  if (keep_pi2) out$pi2_list <- pi2_list
  if (keep_loglik) {
    out$log_lik_complete <- log_lik_complete
    out$active_q_count <- active_q_count
  }
  if (keep_categories) {
    out$i_a_list <- i_a_list
    out$i_g_list <- i_g_list
  }
  out
}

PostMean <- function(Mat_afburn, burn_in) {
  if (burn_in < 0L || burn_in >= length(Mat_afburn) - 1L) {
    stop("burn_in must be between 0 and iteration - 1")
  }
  # Element 1 is the initialization; element burn_in + 2 is MCMC iteration
  # burn_in + 1, the first retained draw.
  idx <- seq.int(burn_in + 2L, length(Mat_afburn))
  Reduce(`+`, Mat_afburn[idx]) / length(idx)
}

PBIC_from_result <- function(mcmc_result, Y, V, covarites, K_a, K_g, burn_in) {
  keep <- seq.int(burn_in + 1L, ncol(mcmc_result$log_lik_complete))
  log_lik <- mcmc_result$log_lik_complete[, keep, drop = FALSE]
  n <- nrow(Y)
  J_y <- ncol(Y)
  J_v <- ncol(V)
  # One intercept per item; an inclusion indicator and an active magnitude per
  # retained Q entry; structural eta; and 2^K_g-1 free pi2 probabilities.
  p_draw <- J_y + J_v + 2 * mcmc_result$active_q_count[keep] +
    K_a * (K_g + 1L + ncol(covarites)) + (2^K_g - 1L)
  mean(p_draw) * log(n) - 2 * sum(rowMeans(log_lik))
}

PlotQMatrix <- function(Q,
                        threshold = 0.5,
                        title = "Estimated Q-matrix",
                        xlab = "Latent Attributes",
                        ylab = "Question Numbers",
                        save_path = NULL,
                        width = 8,
                        height = 4.5,
                        pointsize = 14) {
  Q <- as.matrix(Q)
  if (!is.numeric(Q)) stop("Q must be a numeric matrix.")

  # Posterior mean Q matrices are continuous, so threshold them before plotting.
  Q_binary <- ifelse(Q >= threshold, 1L, 0L)
  storage.mode(Q_binary) <- "integer"

  if (!is.null(save_path)) {
    png(save_path, width = width, height = height, units = "in", res = 300, pointsize = pointsize)
    on.exit(dev.off(), add = TRUE)
  }

  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)

  n_item <- nrow(Q_binary)
  n_attr <- ncol(Q_binary)

  par(mar = c(4.2, 4.5, 2.2, 1.0), family = "serif")
  plot(
    NA,
    xlim = c(0.5, n_attr + 0.5),
    ylim = c(n_item + 0.5, 0.5),
    xaxt = "n",
    yaxt = "n",
    xlab = xlab,
    ylab = ylab,
    main = title,
    bty = "n"
  )

  # Draw the full J by K matrix area first, then fill q_jk = 1 cells in black.
  rect(0.5, 0.5, n_attr + 0.5, n_item + 0.5, col = "white", border = "black")
  abline(v = seq(0.5, n_attr + 0.5, by = 1), col = "grey85", lwd = 0.4)
  abline(h = seq(0.5, n_item + 0.5, by = 1), col = "grey85", lwd = 0.4)

  one_pos <- which(Q_binary == 1L, arr.ind = TRUE)
  if (nrow(one_pos) > 0L) {
    rect(
      xleft = one_pos[, 2] - 0.5,
      ybottom = one_pos[, 1] - 0.5,
      xright = one_pos[, 2] + 0.5,
      ytop = one_pos[, 1] + 0.5,
      col = "black",
      border = "black"
    )
  }

  axis(1, at = seq_len(n_attr), labels = seq_len(n_attr), tick = FALSE)
  axis(2, at = seq_len(n_item), labels = seq_len(n_item), las = 1, tick = FALSE)
  box()

  invisible(Q_binary)
}
