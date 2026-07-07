script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[[1L]]))))
  }
  getwd()
}

all_permutations <- function(K) {
  if (K == 1L) return(matrix(1L, nrow = 1L))
  previous <- all_permutations(K - 1L)
  out <- vector("list", K * nrow(previous))
  index <- 1L
  for (position in seq_len(K)) {
    for (row in seq_len(nrow(previous))) {
      out[[index]] <- append(previous[row, ], K, after = position - 1L)
      index <- index + 1L
    }
  }
  do.call(rbind, out)
}

posterior_mean <- function(draws, burnin) {
  keep <- (burnin + 2L):length(draws)
  Reduce(`+`, draws[keep]) / length(keep)
}

best_permutation <- function(Q_mean, Q_truth) {
  permutations <- all_permutations(ncol(Q_truth))
  discrepancy <- apply(permutations, 1L, function(permutation) {
    sum((Q_mean[, permutation, drop = FALSE] - Q_truth)^2)
  })
  as.integer(permutations[which.min(discrepancy), ])
}

align_chain <- function(chain, burnin) {
  fit <- chain$fit
  truth <- chain$truth
  K <- ncol(truth$Q1)
  perm_1 <- best_permutation(posterior_mean(fit$Q1_list, burnin), truth$Q1)
  perm_2 <- best_permutation(posterior_mean(fit$Q2_list, burnin), truth$Q2)

  for (iteration in seq_along(fit$Q1_list)) {
    fit$Q1_list[[iteration]] <- fit$Q1_list[[iteration]][, perm_1, drop = FALSE]
    fit$Q2_list[[iteration]] <- fit$Q2_list[[iteration]][, perm_2, drop = FALSE]
    fit$B_list[[iteration]] <- cbind(
      fit$B_list[[iteration]][, 1L, drop = FALSE],
      fit$B_list[[iteration]][, 1L + perm_1, drop = FALSE]
    )
    fit$L_list[[iteration]] <- cbind(
      fit$L_list[[iteration]][, 1L, drop = FALSE],
      fit$L_list[[iteration]][, 1L + perm_2, drop = FALSE]
    )
    eta <- fit$Sita_list[[iteration]][, perm_1, drop = FALSE]
    eta[2L:(K + 1L), ] <- eta[1L + perm_2, , drop = FALSE]
    fit$Sita_list[[iteration]] <- eta
  }
  chain$fit <- fit
  chain
}

extract_matrix <- function(draws) {
  draws <- draws[-1L]
  out <- matrix(NA_real_, nrow = length(draws), ncol = length(c(draws[[1L]])))
  for (iteration in seq_along(draws)) out[iteration, ] <- c(draws[[iteration]])
  out
}

extract_all_parameters <- function(chain) {
  fields <- c("B_list", "L_list", "Sita_list", "Q1_list", "Q2_list")
  blocks <- c("B1", "B2", "eta", "Q1", "Q2")
  matrices <- lapply(fields, function(field) extract_matrix(chain$fit[[field]]))
  list(
    draws = do.call(cbind, matrices),
    block = rep(blocks, vapply(matrices, ncol, integer(1L)))
  )
}

classic_rhat_vector <- function(chain_matrices, iteration) {
  selected <- lapply(chain_matrices, function(x) x[seq_len(iteration), , drop = FALSE])
  chain_means <- do.call(rbind, lapply(selected, colMeans))
  chain_variances <- do.call(rbind, lapply(selected, function(x) {
    colSums((x - rep(colMeans(x), each = nrow(x)))^2) / (nrow(x) - 1)
  }))
  within <- colMeans(chain_variances)
  between <- iteration * apply(chain_means, 2L, stats::var)
  rhat <- sqrt((((iteration - 1) / iteration) * within + between / iteration) / within)
  constant <- within == 0 & apply(chain_means, 2L, function(x) max(x) == min(x))
  rhat[constant] <- 1
  rhat[!is.finite(rhat)] <- NA_real_
  rhat
}

scenario_paths <- function(chain_dir, scenario_id) {
  scenario_pattern <- sprintf("^scenario_%02d_", scenario_id)
  scenario_folder <- list.files(chain_dir, pattern = scenario_pattern, full.names = TRUE)
  if (length(scenario_folder) != 1L) stop("Could not identify scenario ", scenario_id)
  replicate_folders <- list.dirs(scenario_folder, recursive = FALSE, full.names = TRUE)
  replicate_folders[order(replicate_folders)]
}

compute_scenario <- function(chain_dir, scenario_id, checkpoints, burnin) {
  replicate_folders <- scenario_paths(chain_dir, scenario_id)
  replicate_results <- vector("list", length(replicate_folders))
  metadata <- NULL

  for (replicate_index in seq_along(replicate_folders)) {
    chain_files <- list.files(
      replicate_folders[[replicate_index]],
      pattern = "^chain_[1-4]\\.rds$",
      full.names = TRUE
    )
    chain_files <- chain_files[order(chain_files)]
    if (length(chain_files) != 4L) stop("Four chains are required in ", replicate_folders[[replicate_index]])

    chains <- lapply(chain_files, readRDS)
    chains <- lapply(chains, align_chain, burnin = burnin)
    extracted <- lapply(chains, extract_all_parameters)
    block <- extracted[[1L]]$block
    chain_matrices <- lapply(extracted, `[[`, "draws")
    values <- do.call(rbind, lapply(checkpoints, function(iteration) {
      classic_rhat_vector(chain_matrices, iteration)
    }))
    replicate_results[[replicate_index]] <- list(
      replicate = chains[[1L]]$metadata$replicate_id,
      values = values,
      block = block
    )
    metadata <- chains[[1L]]$metadata
    rm(chains, extracted, chain_matrices)
    invisible(gc())
  }

  list(
    scenario_id = scenario_id,
    n = metadata$n,
    J = metadata$J_total,
    K = metadata$K1,
    checkpoints = checkpoints,
    replicates = replicate_results
  )
}

plot_panel <- function(result, y_limit = c(0.95, 3.7)) {
  block_colors <- c(
    B1 = "#0072B2",
    B2 = "#D55E00",
    eta = "#009E73",
    Q1 = "#CC79A7",
    Q2 = "#E69F00"
  )
  graphics::plot(
    NA,
    xlim = range(result$checkpoints),
    ylim = y_limit,
    xlab = "Iteration",
    ylab = expression(hat(R)),
    main = bquote(
      n == .(result$n) * ", " ~ J == .(result$J) * ", " ~
        K[1] == .(result$K) * ", " ~ K[2] == .(result$K)
    ),
    las = 1
  )
  graphics::grid(col = "grey90")
  graphics::abline(h = 1.05, lty = 2L, col = "grey35", lwd = 1.2)

  replicate_array <- simplify2array(lapply(result$replicates, `[[`, "values"))
  averaged_values <- apply(replicate_array, c(1L, 2L), mean, na.rm = TRUE)
  parameter_blocks <- result$replicates[[1L]]$block
  for (parameter_index in seq_len(ncol(averaged_values))) {
    values <- pmin(averaged_values[, parameter_index], y_limit[2L])
    graphics::lines(
      result$checkpoints,
      values,
      col = block_colors[[parameter_blocks[[parameter_index]]]],
      lwd = 0.55
    )
  }
  graphics::box()
}

save_page <- function(results, K, output_stem) {
  draw_page <- function() {
    old_par <- graphics::par(
      mfrow = c(2L, 3L),
      mar = c(3.5, 3.8, 3.0, 1.0),
      oma = c(4.0, 1.0, 2.8, 0.5),
      mgp = c(2.1, 0.7, 0),
      tcl = -0.25,
      cex = 0.78
    )
    on.exit(graphics::par(old_par), add = TRUE)
    for (result in results) plot_panel(result)
    graphics::mtext(
      bquote(
        "Running Gelman--Rubin diagnostics for " ~
          K[1] == .(K) * " and " ~ K[2] == .(K)
      ),
      outer = TRUE,
      side = 3L,
      line = 1.0,
      font = 2L,
      cex = 1.05
    )
    graphics::par(xpd = NA)
    graphics::legend(
      x = grconvertX(0.5, from = "ndc", to = "user"),
      y = grconvertY(0.015, from = "ndc", to = "user"),
      legend = c(expression(B[1]), expression(B[2]), expression(eta), expression(Q[1]), expression(Q[2])),
      col = c("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"),
      lty = 1L,
      lwd = 2.2,
      horiz = TRUE,
      xjust = 0.5,
      yjust = 0,
      bty = "n",
      seg.len = 2.5,
      x.intersp = 0.8,
      text.width = strwidth(expression(B[2])),
      cex = 0.9
    )
  }

  grDevices::png(paste0(output_stem, ".png"), width = 2400, height = 1650, res = 220)
  draw_page()
  grDevices::dev.off()
}

project_dir <- normalizePath(file.path(script_dir(), ".."), mustWork = TRUE)
chain_dir <- file.path(project_dir, "output", "chains")
output_dir <- file.path(project_dir, "output", "running_rhat")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

cache_path <- file.path(output_dir, "running_rhat_all_scenarios.rds")
if (file.exists(cache_path)) {
  all_results <- readRDS(cache_path)
} else {
  checkpoints <- seq.int(100L, 3000L, by = 100L)
  burnin <- 2000L
  all_results <- vector("list", 18L)
  for (scenario_id in seq_len(18L)) {
    message("Computing running Rhat for scenario ", scenario_id, " of 18")
    all_results[[scenario_id]] <- compute_scenario(
      chain_dir = chain_dir,
      scenario_id = scenario_id,
      checkpoints = checkpoints,
      burnin = burnin
    )
  }
  saveRDS(all_results, cache_path)
}

for (K in 2:4) {
  results <- Filter(function(x) x$K == K, all_results)
  results <- results[order(vapply(results, `[[`, integer(1L), "J"),
                           vapply(results, `[[`, integer(1L), "n"))]
  save_page(
    results = results,
    K = K,
    output_stem = file.path(output_dir, sprintf("simulation_running_rhat_K%d", K))
  )
}

message("Saved running-Rhat figures to ", output_dir)
