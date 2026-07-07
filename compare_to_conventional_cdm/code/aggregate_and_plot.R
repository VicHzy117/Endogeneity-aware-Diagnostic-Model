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

q_binary <- function(Q, threshold = 0.5) {
  1L * (as.matrix(Q) >= threshold)
}

align_q_columns <- function(Q_est, Q_true, threshold = 0.5) {
  Q_est_bin <- q_binary(Q_est, threshold)
  Q_true <- as.matrix(Q_true)
  if (ncol(Q_est_bin) != ncol(Q_true)) {
    return(list(Q = Q_est_bin, permutation = seq_len(ncol(Q_est_bin)), score = NA_integer_))
  }

  make_perms <- function(x) {
    if (length(x) == 1L) return(matrix(x, nrow = 1L))
    do.call(rbind, lapply(seq_along(x), function(i) cbind(x[i], make_perms(x[-i]))))
  }
  perms <- make_perms(seq_len(ncol(Q_true)))
  scores <- apply(perms, 1L, function(idx) sum(Q_est_bin[, idx, drop = FALSE] == Q_true))
  best <- perms[which.max(scores), ]
  list(Q = Q_est_bin[, best, drop = FALSE], permutation = best, score = max(scores))
}

draw_q_panel <- function(Q, title, xlab = "Latent Attributes", ylab = "Question Numbers") {
  Q <- q_binary(Q)
  n_item <- nrow(Q)
  n_attr <- ncol(Q)
  par(mar = c(4.2, 4.5, 2.2, 1.0), family = "serif")
  plot(NA, xlim = c(0.5, n_attr + 0.5), ylim = c(n_item + 0.5, 0.5),
       xaxt = "n", yaxt = "n", xlab = xlab, ylab = ylab, main = title, bty = "n")
  rect(0.5, 0.5, n_attr + 0.5, n_item + 0.5, col = "white", border = "black")
  abline(v = seq(0.5, n_attr + 0.5, by = 1), col = "grey85", lwd = 0.4)
  abline(h = seq(0.5, n_item + 0.5, by = 1), col = "grey85", lwd = 0.4)
  one_pos <- which(Q == 1L, arr.ind = TRUE)
  if (nrow(one_pos) > 0L) {
    rect(one_pos[, 2] - 0.5, one_pos[, 1] - 0.5,
         one_pos[, 2] + 0.5, one_pos[, 1] + 0.5,
         col = "black", border = "black")
  }
  axis(1, at = seq_len(n_attr), labels = seq_len(n_attr), tick = FALSE)
  axis(2, at = seq_len(n_item), labels = seq_len(n_item), las = 1, tick = FALSE)
  box()
}

save_q_compare <- function(path, panels, width = 12, height = 5) {
  png(path, width = width, height = height, units = "in", res = 300, pointsize = 13)
  on.exit(dev.off(), add = TRUE)
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)
  par(mfrow = c(1, length(panels)))
  for (panel in panels) draw_q_panel(panel$Q, panel$title)
}

args <- parse_args()
code_dir <- script_dir()
project_dir <- normalizePath(file.path(code_dir, ".."), mustWork = TRUE)
setwd(project_dir)

data_path <- arg_value(args, "data", file.path("data", "simulation_data.rds"))
result_dir <- arg_value(args, "result_dir", "result")
plot_best <- tolower(arg_value(args, "plot_best", "TRUE")) %in% c("1", "true", "t", "yes", "y")

sim <- readRDS(data_path)
dir.create(file.path(result_dir, "summary"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(result_dir, "plots"), recursive = TRUE, showWarnings = FALSE)

old_files <- list.files(file.path(result_dir, "conventional_cdm"), pattern = "\\.rds$", full.names = TRUE)
new_files <- list.files(file.path(result_dir, "eacdm"), pattern = "\\.rds$", full.names = TRUE)

old_rows <- do.call(rbind, lapply(old_files, function(path) {
  x <- readRDS(path)
  data.frame(dataset_id = x$dataset_id, K = x$K, bic = x$bic,
             elapsed_min = x$elapsed_min, path = path)
}))
new_rows <- do.call(rbind, lapply(new_files, function(path) {
  x <- readRDS(path)
  data.frame(dataset_id = x$dataset_id, K_a = x$K_a, K_g = x$K_g, bic = x$bic,
             elapsed_min = x$elapsed_min, path = path)
}))

write.csv(old_rows, file.path(result_dir, "summary", "conventional_cdm_all_bic.csv"), row.names = FALSE)
write.csv(new_rows, file.path(result_dir, "summary", "eacdm_all_bic.csv"), row.names = FALSE)

old_best <- do.call(rbind, lapply(split(old_rows, old_rows$dataset_id), function(df) df[which.min(df$bic), ]))
new_best <- do.call(rbind, lapply(split(new_rows, new_rows$dataset_id), function(df) df[which.min(df$bic), ]))
write.csv(old_best, file.path(result_dir, "summary", "conventional_cdm_best_k.csv"), row.names = FALSE)
write.csv(new_best, file.path(result_dir, "summary", "eacdm_best_k.csv"), row.names = FALSE)

selection_summary <- list(
  conventional_cdm_K = as.data.frame(table(old_best$K)),
  eacdm_K = as.data.frame(table(paste0("K1_", new_best$K_a, "_K2_", new_best$K_g)))
)
write.csv(selection_summary$conventional_cdm_K, file.path(result_dir, "summary", "conventional_cdm_best_k_counts.csv"), row.names = FALSE)
write.csv(selection_summary$eacdm_K, file.path(result_dir, "summary", "eacdm_best_k_counts.csv"), row.names = FALSE)

if (plot_best) {
  for (idx in seq_len(nrow(old_best))) {
    best <- old_best[idx, ]
    x <- readRDS(best$path)
    old_y <- x$Q_mean[seq_len(sim$truth$J_y), , drop = FALSE]
    old_v <- x$Q_mean[(sim$truth$J_y + 1L):(sim$truth$J_y + sim$truth$J_v), , drop = FALSE]
    old_y_aligned <- align_q_columns(old_y, sim$truth$Q_y)$Q
    old_v_aligned <- align_q_columns(old_v, sim$truth$Q_v)$Q
    save_q_compare(
      file.path(result_dir, "plots", sprintf("dataset_%03d_conventional_cdm_best_Q.png", best$dataset_id)),
      list(
        list(Q = sim$truth$Q_y, title = "True Q-Y"),
        list(Q = old_y_aligned, title = sprintf("Conventional CDM Q-Y K=%d", best$K)),
        list(Q = sim$truth$Q_v, title = "True Q-V"),
        list(Q = old_v_aligned, title = sprintf("Conventional CDM Q-V K=%d", best$K))
      ),
      width = 16,
      height = 5
    )
  }

  for (idx in seq_len(nrow(new_best))) {
    best <- new_best[idx, ]
    x <- readRDS(best$path)
    q1_aligned <- align_q_columns(x$Q1_mean, sim$truth$Q_y)$Q
    q2_aligned <- align_q_columns(x$Q2_mean, sim$truth$Q_v)$Q
    save_q_compare(
      file.path(result_dir, "plots", sprintf("dataset_%03d_eacdm_best_Q.png", best$dataset_id)),
      list(
        list(Q = sim$truth$Q_y, title = "True Q-Y"),
        list(Q = q1_aligned, title = sprintf("EACDM Q-Y K1=%d", best$K_a)),
        list(Q = sim$truth$Q_v, title = "True Q-V"),
        list(Q = q2_aligned, title = sprintf("EACDM Q-V K2=%d", best$K_g))
      ),
      width = 16,
      height = 5
    )
  }
}

saveRDS(list(conventional_cdm_all = old_rows, eacdm_all = new_rows,
             conventional_cdm_best = old_best, eacdm_best = new_best,
             selection_summary = selection_summary),
        file.path(result_dir, "summary", "model_comparison_summary.rds"))
cat("Wrote summaries to", file.path(result_dir, "summary"), "\n")
