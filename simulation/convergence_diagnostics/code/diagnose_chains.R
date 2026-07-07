parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list()
  for (arg in args) {
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
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

all_permutations <- function(K) {
  if (K == 1L) return(matrix(1L, nrow = 1L))
  prev <- all_permutations(K - 1L)
  out <- vector("list", K * nrow(prev))
  z <- 1L
  for (pos in seq_len(K)) {
    for (r in seq_len(nrow(prev))) {
      out[[z]] <- append(prev[r, ], K, after = pos - 1L)
      z <- z + 1L
    }
  }
  do.call(rbind, out)
}

post_mean <- function(draws, burnin) {
  keep <- (burnin + 2L):length(draws)
  Reduce(`+`, draws[keep]) / length(keep)
}

best_permutation <- function(Q_mean, Q_truth) {
  permutations <- all_permutations(ncol(Q_truth))
  discrepancy <- apply(permutations, 1L, function(p) {
    sum((Q_mean[, p, drop = FALSE] - Q_truth)^2)
  })
  as.integer(permutations[which.min(discrepancy), ])
}

align_chain <- function(chain, truth, burnin) {
  fit <- chain$fit
  K <- ncol(truth$Q1)
  perm_a <- best_permutation(post_mean(fit$Q1_list, burnin), truth$Q1)
  perm_g <- best_permutation(post_mean(fit$Q2_list, burnin), truth$Q2)

  for (t in seq_along(fit$Q1_list)) {
    fit$Q1_list[[t]] <- fit$Q1_list[[t]][, perm_a, drop = FALSE]
    fit$Q2_list[[t]] <- fit$Q2_list[[t]][, perm_g, drop = FALSE]
    fit$B_list[[t]] <- cbind(
      fit$B_list[[t]][, 1L, drop = FALSE],
      fit$B_list[[t]][, 1L + perm_a, drop = FALSE]
    )
    fit$L_list[[t]] <- cbind(
      fit$L_list[[t]][, 1L, drop = FALSE],
      fit$L_list[[t]][, 1L + perm_g, drop = FALSE]
    )
    eta <- fit$Sita_list[[t]][, perm_a, drop = FALSE]
    eta[2L:(K + 1L), ] <- eta[1L + perm_g, , drop = FALSE]
    fit$Sita_list[[t]] <- eta
  }

  chain$fit <- fit
  chain$metadata$perm_a <- perm_a
  chain$metadata$perm_g <- perm_g
  chain
}

extract_draw_matrix <- function(fit, field, burnin = 0L, include_burnin = FALSE) {
  draws <- fit[[field]]
  if (include_burnin) {
    keep <- 2L:length(draws)
  } else {
    keep <- (burnin + 2L):length(draws)
  }
  out <- matrix(NA_real_, nrow = length(keep), ncol = length(c(draws[[keep[1L]]])))
  for (i in seq_along(keep)) out[i, ] <- c(draws[[keep[i]]])
  out
}

split_chains <- function(chain_mats) {
  n <- min(vapply(chain_mats, nrow, integer(1L)))
  half <- floor(n / 2L)
  if (half < 2L) stop("At least four post-burn-in draws per chain are required.")
  chain_mats <- lapply(chain_mats, function(x) x[seq_len(2L * half), , drop = FALSE])
  c(
    lapply(chain_mats, function(x) x[seq_len(half), , drop = FALSE]),
    lapply(chain_mats, function(x) x[half + seq_len(half), , drop = FALSE])
  )
}

classic_rhat <- function(chains) {
  n <- length(chains[[1L]])
  pooled <- unlist(chains, use.names = FALSE)
  if (all(pooled == pooled[1L])) return(1)
  means <- vapply(chains, mean, numeric(1L))
  variances <- vapply(chains, stats::var, numeric(1L))
  W <- mean(variances)
  B <- n * stats::var(means)
  if (W == 0) return(Inf)
  sqrt((((n - 1) / n) * W + B / n) / W)
}

rank_normalize <- function(chains) {
  sizes <- lengths(chains)
  pooled <- unlist(chains, use.names = FALSE)
  ranks <- rank(pooled, ties.method = "average")
  z <- qnorm((ranks - 3 / 8) / (length(ranks) + 1 / 4))
  split(z, rep(seq_along(sizes), sizes))
}

rank_normalized_split_rhat <- function(chains) {
  pooled <- unlist(chains, use.names = FALSE)
  if (all(pooled == pooled[1L])) return(1)
  rank_rhat <- classic_rhat(rank_normalize(chains))
  center <- stats::median(pooled)
  folded <- lapply(chains, function(x) abs(x - center))
  folded_rhat <- classic_rhat(rank_normalize(folded))
  max(rank_rhat, folded_rhat)
}

compute_block_rhat <- function(aligned_chains, field, burnin, labels) {
  mats <- lapply(aligned_chains, function(x) extract_draw_matrix(x$fit, field, burnin))
  split_mats <- split_chains(mats)
  p <- ncol(split_mats[[1L]])
  rhat <- numeric(p)
  for (j in seq_len(p)) {
    rhat[j] <- rank_normalized_split_rhat(lapply(split_mats, function(x) x[, j]))
  }
  data.frame(parameter = labels, rhat = rhat)
}

parameter_labels <- function(field, truth) {
  K <- ncol(truth$Q1)
  J1 <- nrow(truth$Q1)
  J2 <- nrow(truth$Q2)
  if (field == "Q1_list") {
    return(unlist(lapply(seq_len(K), function(k) sprintf("Q1[item=%d,attr=%d]", seq_len(J1), k))))
  }
  if (field == "Q2_list") {
    return(unlist(lapply(seq_len(K), function(k) sprintf("Q2[item=%d,attr=%d]", seq_len(J2), k))))
  }
  if (field == "B_list") {
    return(unlist(lapply(0:K, function(k) sprintf("B1[item=%d,col=%d]", seq_len(J1), k))))
  }
  if (field == "L_list") {
    return(unlist(lapply(0:K, function(k) sprintf("B2[item=%d,col=%d]", seq_len(J2), k))))
  }
  if (field == "Sita_list") {
    nr <- nrow(truth$eta)
    return(unlist(lapply(seq_len(K), function(k) sprintf("eta[row=%d,col=%d]", seq_len(nr), k))))
  }
  stop("Unknown field: ", field)
}

summarize_block <- function(x) {
  finite <- is.finite(x)
  data.frame(
    n_parameter = length(x),
    median_rhat = stats::median(x),
    q95_rhat = unname(stats::quantile(x, 0.95, names = FALSE)),
    max_rhat = max(x),
    n_nonfinite = sum(!finite),
    n_over_1.01 = sum(x > 1.01),
    n_over_1.05 = sum(x > 1.05),
    proportion_below_1.01 = mean(x < 1.01),
    proportion_below_1.05 = mean(x < 1.05)
  )
}

find_q_index <- function(Q, value) {
  idx <- which(Q == value, arr.ind = TRUE)
  if (!nrow(idx)) return(c(1L, 1L))
  idx[1L, ]
}

selected_trace_parameters <- function(truth) {
  K <- ncol(truth$Q1)
  q1_active <- find_q_index(truth$Q1, 1L)
  q1_inactive <- find_q_index(truth$Q1, 0L)
  q2_active <- find_q_index(truth$Q2, 1L)
  q2_inactive <- find_q_index(truth$Q2, 0L)
  cross_row <- if (K >= 2L) 3L else 2L

  list(
    continuous = list(
      list(field = "Sita_list", row = 1L, col = 1L, label = "eta: intercept"),
      list(field = "Sita_list", row = 2L, col = 1L, label = "eta: corresponding latent effect"),
      list(field = "Sita_list", row = cross_row, col = 1L, label = "eta: cross-latent effect"),
      list(field = "Sita_list", row = K + 2L, col = 1L, label = "eta: covariate effect"),
      list(field = "B_list", row = q1_active[1L], col = q1_active[2L] + 1L, label = "B1: active loading"),
      list(field = "B_list", row = q1_inactive[1L], col = q1_inactive[2L] + 1L, label = "B1: inactive loading"),
      list(field = "L_list", row = q2_active[1L], col = q2_active[2L] + 1L, label = "B2: active loading"),
      list(field = "L_list", row = q2_inactive[1L], col = q2_inactive[2L] + 1L, label = "B2: inactive loading")
    ),
    q = list(
      list(field = "Q1_list", row = q1_active[1L], col = q1_active[2L], label = "Q1: true active"),
      list(field = "Q1_list", row = q1_inactive[1L], col = q1_inactive[2L], label = "Q1: true inactive"),
      list(field = "Q2_list", row = q2_active[1L], col = q2_active[2L], label = "Q2: true active"),
      list(field = "Q2_list", row = q2_inactive[1L], col = q2_inactive[2L], label = "Q2: true inactive")
    )
  )
}

extract_scalar_trace <- function(chain, spec) {
  vapply(chain$fit[[spec$field]], function(x) x[spec$row, spec$col], numeric(1L))[-1L]
}

plot_trace_panels <- function(chains, specs, burnin, path, title, running_mean = FALSE) {
  colors <- c("#0072B2", "#D55E00", "#009E73", "#CC79A7")
  grDevices::pdf(path, width = 10, height = if (length(specs) > 4L) 8 else 6)
  old <- par(no.readonly = TRUE)
  on.exit({
    par(old)
    grDevices::dev.off()
  }, add = TRUE)
  par(mfrow = c(ceiling(length(specs) / 2), 2), mar = c(3.2, 3.4, 2.2, 1), oma = c(0, 0, 2, 0))

  for (spec in specs) {
    traces <- lapply(chains, extract_scalar_trace, spec = spec)
    if (running_mean) {
      traces <- lapply(traces, function(x) cumsum(x) / seq_along(x))
    }
    ylim <- range(unlist(traces), finite = TRUE)
    plot(seq_along(traces[[1L]]), traces[[1L]], type = "l", col = colors[1L],
         xlab = "Iteration", ylab = if (running_mean) "Running mean" else "Draw",
         main = spec$label, ylim = ylim)
    for (i in 2:length(traces)) lines(traces[[i]], col = colors[i])
    abline(v = burnin, lty = 2, col = "gray40")
    if (identical(spec, specs[[1L]])) {
      legend("topright", legend = paste("Chain", seq_along(chains)), col = colors,
             lty = 1, bty = "n", cex = 0.75)
    }
  }
  mtext(title, outer = TRUE, cex = 1.1)
}

args <- parse_args()
project_dir <- normalizePath(file.path(script_dir(), ".."), mustWork = TRUE)
setwd(project_dir)

data_dir <- arg_value(args, "data_dir", file.path("data", "generated"))
output_dir <- arg_value(args, "output_dir", "output")
burnin <- as.integer(arg_value(args, "burnin", 2000L))
chains_per_dataset <- as.integer(arg_value(args, "chains", 4L))
trace_scenarios <- as.integer(strsplit(arg_value(args, "trace_scenarios", "13,11,6"), ",")[[1L]])
trace_replicate <- as.integer(arg_value(args, "trace_replicate", 50L))

manifest <- read.csv(file.path(data_dir, "manifest.csv"))
diagnostic_dir <- file.path(output_dir, "diagnostics")
trace_dir <- file.path(output_dir, "traceplots")
dir.create(diagnostic_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(trace_dir, recursive = TRUE, showWarnings = FALSE)

blocks <- c(
  eta = "Sita_list",
  B1 = "B_list",
  B2 = "L_list",
  Q1 = "Q1_list",
  Q2 = "Q2_list"
)
parameter_rows <- list()
summary_rows <- list()
missing_rows <- list()
trace_rows <- list()

for (i in seq_len(nrow(manifest))) {
  job <- manifest[i, ]
  chain_dir <- file.path(
    output_dir,
    "chains",
    sprintf("scenario_%02d_n%d_J%d_K%d", job$scenario_id, job$n, job$J_block, job$K),
    sprintf("replicate_%03d", job$replicate_id)
  )
  paths <- file.path(chain_dir, sprintf("chain_%d.rds", seq_len(chains_per_dataset)))
  if (any(!file.exists(paths))) {
    missing_chain <- which(!file.exists(paths))
    missing_rows[[length(missing_rows) + 1L]] <- data.frame(
      task_id = (job$diagnostic_dataset_id - 1L) * chains_per_dataset + missing_chain,
      scenario_id = job$scenario_id,
      replicate_id = job$replicate_id,
      n = job$n,
      J_block = job$J_block,
      K = job$K,
      missing_chain = missing_chain
    )
    next
  }

  cat("Diagnosing scenario", job$scenario_id, "replicate", job$replicate_id, "\n")
  raw_chains <- lapply(paths, readRDS)
  truth <- raw_chains[[1L]]$truth
  aligned <- lapply(raw_chains, align_chain, truth = truth, burnin = burnin)

  for (block_name in names(blocks)) {
    field <- blocks[[block_name]]
    result <- compute_block_rhat(
      aligned_chains = aligned,
      field = field,
      burnin = burnin,
      labels = parameter_labels(field, truth)
    )
    result$scenario_id <- job$scenario_id
    result$replicate_id <- job$replicate_id
    result$n <- job$n
    result$J <- 2L * job$J_block
    result$J_block <- job$J_block
    result$K <- job$K
    result$parameter_block <- block_name
    parameter_rows[[length(parameter_rows) + 1L]] <- result

    block_summary <- summarize_block(result$rhat)
    block_summary$scenario_id <- job$scenario_id
    block_summary$replicate_id <- job$replicate_id
    block_summary$n <- job$n
    block_summary$J <- 2L * job$J_block
    block_summary$J_block <- job$J_block
    block_summary$K <- job$K
    block_summary$parameter_block <- block_name
    summary_rows[[length(summary_rows) + 1L]] <- block_summary
  }

  if (job$scenario_id %in% trace_scenarios && job$replicate_id == trace_replicate) {
    specs <- selected_trace_parameters(truth)
    stem <- sprintf(
      "scenario_%02d_n%d_J%d_K%d_rep%03d",
      job$scenario_id, job$n, 2L * job$J_block, job$K, job$replicate_id
    )
    continuous_path <- file.path(trace_dir, paste0(stem, "_continuous_trace.pdf"))
    q_path <- file.path(trace_dir, paste0(stem, "_q_trace.pdf"))
    q_mean_path <- file.path(trace_dir, paste0(stem, "_q_running_mean.pdf"))
    plot_trace_panels(aligned, specs$continuous, burnin, continuous_path,
                      paste("Continuous-parameter traces:", stem))
    plot_trace_panels(aligned, specs$q, burnin, q_path, paste("Q-matrix traces:", stem))
    plot_trace_panels(aligned, specs$q, burnin, q_mean_path,
                      paste("Q-matrix running means:", stem), running_mean = TRUE)
    trace_rows[[length(trace_rows) + 1L]] <- data.frame(
      scenario_id = job$scenario_id,
      replicate_id = job$replicate_id,
      n = job$n,
      J = 2L * job$J_block,
      K = job$K,
      continuous_trace = continuous_path,
      q_trace = q_path,
      q_running_mean = q_mean_path
    )
  }
}

if (!length(parameter_rows)) stop("No complete four-chain datasets were found.")
parameter_results <- do.call(rbind, parameter_rows)
parameter_results <- parameter_results[, c(
  "scenario_id", "replicate_id", "n", "J", "J_block", "K",
  "parameter_block", "parameter", "rhat"
)]
write.csv(parameter_results, file.path(diagnostic_dir, "parameter_rhat.csv"), row.names = FALSE)

replicate_summary <- do.call(rbind, summary_rows)
replicate_summary <- replicate_summary[, c(
  "scenario_id", "replicate_id", "n", "J", "J_block", "K", "parameter_block",
  "n_parameter", "median_rhat", "q95_rhat", "max_rhat", "n_nonfinite",
  "n_over_1.01", "n_over_1.05", "proportion_below_1.01", "proportion_below_1.05"
)]
write.csv(replicate_summary, file.path(diagnostic_dir, "replicate_block_summary.csv"), row.names = FALSE)

scenario_keys <- interaction(
  parameter_results$scenario_id, parameter_results$parameter_block, drop = TRUE
)
scenario_summary <- do.call(rbind, lapply(split(parameter_results, scenario_keys), function(x) {
  ans <- summarize_block(x$rhat)
  ans$scenario_id <- x$scenario_id[1L]
  ans$n <- x$n[1L]
  ans$J <- x$J[1L]
  ans$J_block <- x$J_block[1L]
  ans$K <- x$K[1L]
  ans$parameter_block <- x$parameter_block[1L]
  ans
}))
scenario_summary <- scenario_summary[order(scenario_summary$scenario_id, scenario_summary$parameter_block), ]
write.csv(scenario_summary, file.path(diagnostic_dir, "scenario_block_summary.csv"), row.names = FALSE)

overall_summary <- do.call(rbind, lapply(split(parameter_results, parameter_results$parameter_block), function(x) {
  ans <- summarize_block(x$rhat)
  ans$parameter_block <- x$parameter_block[1L]
  ans
}))
write.csv(overall_summary, file.path(diagnostic_dir, "overall_block_summary.csv"), row.names = FALSE)

missing <- if (length(missing_rows)) do.call(rbind, missing_rows) else
  data.frame(
    task_id = integer(), scenario_id = integer(), replicate_id = integer(),
    n = integer(), J_block = integer(), K = integer(), missing_chain = integer()
  )
write.csv(missing, file.path(diagnostic_dir, "missing_chains.csv"), row.names = FALSE)

trace_manifest <- if (length(trace_rows)) do.call(rbind, trace_rows) else data.frame()
write.csv(trace_manifest, file.path(trace_dir, "traceplot_manifest.csv"), row.names = FALSE)

saveRDS(
  list(
    parameter_rhat = parameter_results,
    replicate_summary = replicate_summary,
    scenario_summary = scenario_summary,
    overall_summary = overall_summary,
    missing_chains = missing,
    traceplots = trace_manifest
  ),
  file.path(diagnostic_dir, "convergence_diagnostics.rds")
)

cat("Saved convergence diagnostics to", normalizePath(diagnostic_dir), "\n")
cat("Missing chain rows:", nrow(missing), "\n")
