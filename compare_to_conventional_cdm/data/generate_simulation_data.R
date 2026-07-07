parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list()
  for (arg in args) {
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
    if (length(kv) == 2L) out[[kv[1L]]] <- kv[2L]
  }
  out
}

arg_value <- function(args, name, default) {
  if (!is.null(args[[name]])) args[[name]] else default
}

script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) {
    return(dirname(normalizePath(gsub("~\\+~", " ", sub("^--file=", "", file_arg[[1L]])))))
  }
  getwd()
}

make_binary_design <- function(K) {
  states <- 0:(2^K - 1L)
  bits <- t(vapply(states, function(x) as.integer(intToBits(x))[seq_len(K)], integer(K)))
  cbind(intercept = 1L, bits)
}

make_q_matrix <- function(K, J) {
  if (K != 3L) stop("The simulation design in this project is fixed at K = 3.")
  Q0 <- diag(K)
  Q1 <- matrix(c(
    1, 1, 0,
    1, 0, 1,
    0, 1, 1
  ), nrow = 3L, byrow = TRUE)
  Q <- rbind(Q0, Q0, Q0, Q1)
  Q <- rbind(Q, Q)
  Q[seq_len(J), , drop = FALSE]
}

make_thresholds <- function(M) {
  J <- length(M)
  tau <- matrix(0, nrow = J, ncol = max(M) + 1L)
  for (j in seq_len(J)) {
    tau[j, seq_len(M[j] + 1L)] <- c(-Inf, seq(0, M[j] - 2L, by = 1), Inf)
  }
  tau
}

sample_ordinal <- function(eta, tau, M) {
  n <- nrow(eta)
  J <- ncol(eta)
  out <- matrix(0L, nrow = n, ncol = J)
  for (i in seq_len(n)) {
    for (j in seq_len(J)) {
      prob <- numeric(M[j])
      for (m in seq_len(M[j])) {
        prob[m] <- pnorm(tau[j, m + 1L] - eta[i, j]) -
          pnorm(tau[j, m] - eta[i, j])
      }
      out[i, j] <- sample.int(M[j], 1L, prob = prob) - 1L
    }
  }
  out
}

row_match_class <- function(class_rows, design) {
  apply(class_rows, 1L, function(row) {
    which(apply(design, 1L, function(candidate) all(candidate == row)))[1L] - 1L
  })
}

generate_one_dataset <- function(dataset_id,
                                 n,
                                 K_a,
                                 K_g,
                                 J_y,
                                 J_v,
                                 gamma,
                                 seed) {
  set.seed(seed + dataset_id - 1L)

  M_y <- rep(3L, J_y)
  M_v <- rep(3L, J_v)
  A <- make_binary_design(K_a)
  G <- make_binary_design(K_g)
  Q_y <- make_q_matrix(K_a, J_y)
  Q_v <- make_q_matrix(K_g, J_v)
  B <- cbind(intercept = rep(-0.2, J_y), Q_y)
  L <- cbind(intercept = rep(-0.2, J_v), Q_v)
  tau_y <- make_thresholds(M_y)
  tau_v <- make_thresholds(M_v)

  covariates <- matrix(numeric(0), nrow = n, ncol = 0L)

  gamma_class <- sample.int(nrow(G), n, replace = TRUE) - 1L
  G_cate <- G[gamma_class + 1L, , drop = FALSE]
  G_catecov <- G_cate

  log_odds <- G_catecov %*% gamma
  A_binary <- matrix(rbinom(n * K_a, 1L, plogis(log_odds)), nrow = n, ncol = K_a)
  A_cate <- cbind(intercept = 1L, A_binary)
  alpha_class <- row_match_class(A_cate, A)

  Y <- sample_ordinal(A_cate %*% t(B), tau_y, M_y)
  V <- sample_ordinal(G_cate %*% t(L), tau_v, M_v)
  colnames(Y) <- paste0("Y", seq_len(J_y))
  colnames(V) <- paste0("V", seq_len(J_v))

  list(
    id = dataset_id,
    Y = Y,
    V = V,
    covariates = covariates,
    alpha_class = alpha_class,
    gamma_class = gamma_class
  )
}

generate_simulation_data <- function(setnum = 100L,
                                     n = 1000L,
                                     K_a = 3L,
                                     K_g = 3L,
                                     J_y = 24L,
                                     J_v = 24L,
                                     seed = 237L) {
  # Section 4.3.1 structure equation setting, using a one-to-one
  # dependency variant: each alpha^(1) attribute depends on one alpha^(2)
  # attribute.
  # logit Pr(alpha_ik^(1) = 1) = gamma_k^T alpha_i^(2),
  # where alpha_i^(2) includes the intercept column.
  gamma <- matrix(c(
    -2.20, -2.20, -0.40,
     4.40,  0.00,  0.00,
     0.00,  4.40,  0.00,
     0.00,  0.00,  0.80
  ), nrow = K_g + 1L, ncol = K_a, byrow = TRUE)
  rownames(gamma) <- c("intercept", paste0("alpha2_", seq_len(K_g)))
  colnames(gamma) <- paste0("alpha1_", seq_len(K_a))

  datasets <- lapply(seq_len(setnum), generate_one_dataset,
                     n = n, K_a = K_a, K_g = K_g, J_y = J_y, J_v = J_v,
                     gamma = gamma, seed = seed)

  list(
    datasets = datasets,
    truth = list(
      K_a = K_a,
      K_g = K_g,
      J_y = J_y,
      J_v = J_v,
      Q_y = make_q_matrix(K_a, J_y),
      Q_v = make_q_matrix(K_g, J_v),
      gamma = gamma
    ),
    metadata = list(
      setnum = setnum,
      n = n,
      response_levels_y = rep(3L, J_y),
      response_levels_v = rep(3L, J_v),
      seed = seed,
      response_encoding = "0-based ordinal categories: 0, 1, 2",
      covariates = "none"
    )
  )
}

args <- parse_args()
out_dir <- arg_value(args, "out_dir", script_dir())
setnum <- as.integer(arg_value(args, "setnum", 100L))
n <- as.integer(arg_value(args, "n", 1000L))
seed <- as.integer(arg_value(args, "seed", 237L))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
sim <- generate_simulation_data(setnum = setnum, n = n, seed = seed)

saveRDS(sim, file.path(out_dir, "simulation_data.rds"))
save(sim, file = file.path(out_dir, "simulation_data.RData"))
write.csv(sim$truth$Q_y, file.path(out_dir, "true_Q_y.csv"), row.names = FALSE)
write.csv(sim$truth$Q_v, file.path(out_dir, "true_Q_v.csv"), row.names = FALSE)
write.csv(sim$truth$gamma, file.path(out_dir, "true_gamma.csv"))

cat("Saved simulation data to", normalizePath(out_dir), "\n")
cat("Datasets:", length(sim$datasets), "n:", sim$metadata$n, "\n")
