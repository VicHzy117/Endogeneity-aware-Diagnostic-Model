library(Rcpp)
library(RcppArmadillo)

optional_packages <- c("ggplot2", "reshape2", "gridExtra")
invisible(lapply(optional_packages, function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Optional plotting package not installed: ", pkg)
  }
}))

script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) return(normalizePath(gsub("~\\+~", " ", sub("^--file=", "", file_arg[[1]]))))
  from_source <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NA_character_)
  if (!is.na(from_source)) return(from_source)
  file.path(getwd(), "regular_cdm_model.R")
}

script_dir <- dirname(script_path())
project_dir <- normalizePath(file.path(script_dir, ".."))
sourceCpp(file.path(script_dir, "regular_cdm_mcmc.cpp"))

make_thresholds <- function(Mj) {
  tau <- matrix(0, nrow = Mj + 1L, ncol = 1L)
  tau[1L, 1L] <- -Inf
  tau[Mj + 1L, 1L] <- Inf
  for (m in 2:Mj) tau[m, 1L] <- m - 2L
  tau
}

make_latent_design <- function(K, interaction_order = 1L) {
  if (interaction_order < 1L) stop("interaction_order must be at least 1.")
  states <- 0:(2^K - 1L)
  main_effects <- t(vapply(states, function(x) as.integer(intToBits(x))[seq_len(K)], integer(K)))
  colnames(main_effects) <- paste0("a", seq_len(K))

  effects <- main_effects
  if (interaction_order >= 2L) {
    for (order in 2L:min(interaction_order, K)) {
      combos <- utils::combn(seq_len(K), order)
      for (col in seq_len(ncol(combos))) {
        idx <- combos[, col]
        effects <- cbind(effects, apply(main_effects[, idx, drop = FALSE], 1L, prod))
        colnames(effects)[ncol(effects)] <- paste0("a", paste(idx, collapse = "_a"))
      }
    }
  }

  cbind(intercept = 1L, effects)
}

initial_classes <- function(n, design) {
  class_id <- sample.int(nrow(design), size = n, replace = TRUE) - 1L
  list(
    alpha = class_id,
    counts = matrix(tabulate(class_id + 1L, nbins = nrow(design)), nrow = 1L),
    design_rows = design[class_id + 1L, , drop = FALSE]
  )
}

prepare_real_data <- function(data_path = file.path(project_dir, "realdata_list.RData"),
                              drop_v_columns = 1:7) {
  load(data_path)
  if (!exists("realdata_list")) stop("Expected object `realdata_list` in ", data_path)

  Y <- realdata_list[[1]]
  V <- realdata_list[[2]]
  covariates <- realdata_list[[3]]
  if (length(drop_v_columns)) V <- V[, -drop_v_columns, drop = FALSE]

  list(Y = cbind(Y, V), covariates = covariates)
}

run_regular_cdm <- function(Y,
                            K = 6L,
                            Mj = 4L,
                            iteration = 3000L,
                            burnin = 2500L,
                            interaction_order = 1L,
                            seed = 1214131L) {
  if (burnin >= iteration) stop("burnin must be smaller than iteration.")
  set.seed(seed)

  n <- nrow(Y)
  J <- ncol(Y)
  design <- make_latent_design(K, interaction_order)
  n_effects <- ncol(design) - 1L
  tau <- make_thresholds(Mj)
  init <- initial_classes(n, design)

  c1 <- 1L
  c0 <- 500L
  omega <- 0.4
  B_gibbs <- matrix(0, nrow = J, ncol = ncol(design))
  Q_MH <- matrix(0, nrow = J, ncol = n_effects)
  Q_qta <- cbind(intercept = 1, Q_MH)
  V_prior <- Q_qta / c1 + (1 - Q_qta) / c0

  mcmc <- f_mcmc(
    init$counts, init$alpha, design, tau, as.matrix(Y), Mj, init$design_rows,
    B_gibbs, c1, c0, omega, V_prior, Q_MH, Q_qta, iteration
  )

  posterior_idx <- burnin:iteration
  likelihood <- pmax(mcmc$P_Y[, posterior_idx, drop = FALSE], 1e-300)
  log_likelihood <- log(likelihood)
  n_parameters <- (2L * n_effects + 1L) * J

  average_Q <- apply(mcmc$Q_MH_trace[, , posterior_idx, drop = FALSE], 1:2, mean)
  average_B <- apply(mcmc$B_gibbs_trace[, , posterior_idx, drop = FALSE], 1:2, mean)
  alpha_final <- apply(mcmc$alpha_gibbs_trace[, posterior_idx, drop = FALSE], 1, function(x) {
    as.integer(names(which.max(table(x))))
  })

  list(
    mcmc = mcmc,
    design = design,
    effect_names = colnames(design)[-1L],
    posterior = list(
      Q_mean = average_Q,
      Q_binary = ifelse(average_Q >= 0.5, 1L, 0L),
      B_mean = average_B,
      alpha_mode = alpha_final
    ),
    criteria = list(
      pbic = n_parameters * log(n) - 2 * sum(colMeans(log_likelihood)),
      pwaic = -2 * sum(log(rowMeans(likelihood))) + 2 * sum(apply(log_likelihood, 1, var)),
      n_parameters = n_parameters
    ),
    settings = list(K = K, Mj = Mj, iteration = iteration, burnin = burnin,
                    interaction_order = interaction_order, seed = seed)
  )
}

plot_q_blocks <- function(Q, item_names = NULL, split_at = 22L) {
  if (is.null(item_names)) item_names <- seq_len(nrow(Q))
  rownames(Q) <- item_names

  q1 <- reshape2::melt(Q[seq_len(min(split_at, nrow(Q))), , drop = FALSE])
  p1 <- ggplot2::ggplot(q1, ggplot2::aes(x = Var2, y = Var1, fill = value)) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_gradient(low = "white", high = "black") +
    ggplot2::labs(title = "Q block 1", x = "Latent effects", y = "Items")

  if (nrow(Q) <= split_at) return(p1)

  q2 <- reshape2::melt(Q[(split_at + 1L):nrow(Q), , drop = FALSE])
  p2 <- ggplot2::ggplot(q2, ggplot2::aes(x = Var2, y = Var1, fill = value)) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_gradient(low = "white", high = "black") +
    ggplot2::labs(title = "Q block 2", x = "Latent effects", y = "Items")

  gridExtra::grid.arrange(p1, p2, ncol = 2)
}

if (sys.nframe() == 0L) {
  data <- prepare_real_data()

  result <- run_regular_cdm(
    Y = data$Y,
    K = 6L,
    Mj = 4L,
    iteration = 3000L,
    burnin = 2500L,
    interaction_order = 1L
  )

  print(result$criteria)
  plot_q_blocks(result$posterior$Q_binary, item_names = colnames(data$Y))
}
