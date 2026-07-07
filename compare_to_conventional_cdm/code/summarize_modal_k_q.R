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

q_binary <- function(Q, threshold = 0.5) {
  1L * (as.matrix(Q) >= threshold)
}

make_perms <- function(x) {
  if (length(x) == 1L) return(matrix(x, nrow = 1L))
  do.call(rbind, lapply(seq_along(x), function(i) cbind(x[i], make_perms(x[-i]))))
}

align_binary_q_columns <- function(Q_est, Q_true, threshold = 0.5) {
  Q_est_bin <- q_binary(Q_est, threshold)
  Q_true <- q_binary(Q_true, threshold)
  if (ncol(Q_est_bin) != ncol(Q_true)) {
    return(list(Q = Q_est_bin, permutation = seq_len(ncol(Q_est_bin)), score = NA_integer_))
  }

  perms <- make_perms(seq_len(ncol(Q_true)))
  scores <- apply(perms, 1L, function(idx) sum(Q_est_bin[, idx, drop = FALSE] == Q_true))
  best <- perms[which.max(scores), ]
  list(Q = Q_est_bin[, best, drop = FALSE], permutation = best, score = max(scores))
}

draw_q_panel <- function(Q, title, threshold = 0.5,
                         xlab = "Latent Attributes", ylab = "Question Numbers") {
  Q <- q_binary(Q, threshold)
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

save_q_plot <- function(path, panels, width = 14, height = 6) {
  png(path, width = width, height = height, units = "in", res = 300, pointsize = 13)
  on.exit(dev.off(), add = TRUE)
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)
  par(mfrow = c(1, length(panels)))
  for (panel in panels) draw_q_panel(panel$Q, panel$title)
}

draw_blockwise_q_panel <- function(Q, title, row_break = 24L, col_break = NULL,
                                   threshold = 0.5) {
  Q <- q_binary(Q, threshold)
  n_item <- nrow(Q)
  n_attr <- ncol(Q)
  par(mar = c(4.2, 5.0, 2.2, 1.0), family = "serif")
  plot(NA, xlim = c(0.5, n_attr + 0.5), ylim = c(n_item + 0.5, 0.5),
       xaxt = "n", yaxt = "n", xlab = "Latent Attributes",
       ylab = "Item Numbers", main = title, cex.main = 0.92, bty = "n")
  rect(0.5, 0.5, n_attr + 0.5, n_item + 0.5, col = "white", border = "black")
  abline(v = seq(0.5, n_attr + 0.5, by = 1), col = "grey85", lwd = 0.4)
  abline(h = seq(0.5, n_item + 0.5, by = 1), col = "grey85", lwd = 0.4)
  one_pos <- which(Q == 1L, arr.ind = TRUE)
  if (nrow(one_pos) > 0L) {
    rect(one_pos[, 2] - 0.5, one_pos[, 1] - 0.5,
         one_pos[, 2] + 0.5, one_pos[, 1] + 0.5,
         col = "black", border = "black")
  }
  abline(h = row_break + 0.5, lwd = 2)
  if (!is.null(col_break)) abline(v = col_break + 0.5, lwd = 2)
  axis(1, at = seq_len(n_attr), labels = seq_len(n_attr), tick = FALSE)
  axis(2, at = seq_len(n_item), labels = seq_len(n_item), las = 1,
       tick = FALSE, cex.axis = 0.65)
  box()
}

save_blockwise_q_plot <- function(path, true_q, estimated_q, estimated_title,
                                  true_col_break = 3L,
                                  estimated_col_break = NULL) {
  png(path, width = 11.5, height = 8, units = "in", res = 300, pointsize = 13)
  on.exit(dev.off(), add = TRUE)
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par), add = TRUE)
  par(mfrow = c(1, 2))
  draw_blockwise_q_panel(true_q, "True Blockwise Q-matrix",
                         col_break = true_col_break)
  draw_blockwise_q_panel(estimated_q, estimated_title,
                         col_break = estimated_col_break)
}

mean_matrix <- function(mats) {
  Reduce(`+`, mats) / length(mats)
}

args <- parse_args()
data_path <- arg_value(args, "data", file.path("data", "simulation_data.rds"))
result_dir <- arg_value(args, "result_dir", "result")
out_dir <- file.path(result_dir, "modal_k_summary")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

sim <- readRDS(data_path)

old_files <- list.files(file.path(result_dir, "conventional_cdm"), pattern = "[.]rds$", full.names = TRUE)
new_files <- list.files(file.path(result_dir, "eacdm"), pattern = "[.]rds$", full.names = TRUE)

old_all <- do.call(rbind, lapply(old_files, function(path) {
  x <- readRDS(path)
  data.frame(model = "conventional_cdm", dataset_id = x$dataset_id, K = x$K, bic = x$bic, path = path)
}))
new_all <- do.call(rbind, lapply(new_files, function(path) {
  x <- readRDS(path)
  data.frame(model = "eacdm", dataset_id = x$dataset_id, K_a = x$K_a, K_g = x$K_g,
             combo = sprintf("K1_%d_K2_%d", x$K_a, x$K_g), bic = x$bic, path = path)
}))

old_best <- do.call(rbind, lapply(split(old_all, old_all$dataset_id), function(df) {
  df[which.min(df$bic), , drop = FALSE]
}))
new_best <- do.call(rbind, lapply(split(new_all, new_all$dataset_id), function(df) {
  df[which.min(df$bic), , drop = FALSE]
}))

old_counts <- as.data.frame(table(old_best$K))
names(old_counts) <- c("K", "n_replicates")
old_counts$proportion <- old_counts$n_replicates / nrow(old_best)

new_counts <- as.data.frame(table(new_best$combo))
names(new_counts) <- c("K_combo", "n_replicates")
new_counts$proportion <- new_counts$n_replicates / nrow(new_best)

old_modal_K <- as.integer(as.character(old_counts$K[which.max(old_counts$n_replicates)]))
new_modal_combo <- as.character(new_counts$K_combo[which.max(new_counts$n_replicates)])

old_modal_best <- old_best[old_best$K == old_modal_K, , drop = FALSE]
new_modal_best <- new_best[new_best$combo == new_modal_combo, , drop = FALSE]

old_y_mats <- list()
old_v_mats <- list()
if (old_modal_K == sim$truth$K_a + sim$truth$K_g) {
  true_old_q <- rbind(
    cbind(sim$truth$Q_y, matrix(0L, nrow = sim$truth$J_y, ncol = sim$truth$K_g)),
    cbind(matrix(0L, nrow = sim$truth$J_v, ncol = sim$truth$K_a), sim$truth$Q_v)
  )
  old_full_mats <- list()
  for (path in old_modal_best$path) {
    x <- readRDS(path)
    old_full_mats[[length(old_full_mats) + 1L]] <- align_binary_q_columns(x$Q_mean, true_old_q)$Q
  }
  old_full_mean <- mean_matrix(old_full_mats)
  old_q_y_mean <- old_full_mean[seq_len(sim$truth$J_y), , drop = FALSE]
  old_q_v_mean <- old_full_mean[(sim$truth$J_y + 1L):(sim$truth$J_y + sim$truth$J_v), , drop = FALSE]
} else {
  for (path in old_modal_best$path) {
    x <- readRDS(path)
    q <- x$Q_mean
    q_y <- q[seq_len(sim$truth$J_y), , drop = FALSE]
    q_v <- q[(sim$truth$J_y + 1L):(sim$truth$J_y + sim$truth$J_v), , drop = FALSE]
    old_y_mats[[length(old_y_mats) + 1L]] <- align_binary_q_columns(q_y, sim$truth$Q_y)$Q
    old_v_mats[[length(old_v_mats) + 1L]] <- align_binary_q_columns(q_v, sim$truth$Q_v)$Q
  }
  old_q_y_mean <- mean_matrix(old_y_mats)
  old_q_v_mean <- mean_matrix(old_v_mats)
}

new_q1_mats <- list()
new_q2_mats <- list()
for (path in new_modal_best$path) {
  x <- readRDS(path)
  new_q1_mats[[length(new_q1_mats) + 1L]] <- align_binary_q_columns(x$Q1_mean, sim$truth$Q_y)$Q
  new_q2_mats[[length(new_q2_mats) + 1L]] <- align_binary_q_columns(x$Q2_mean, sim$truth$Q_v)$Q
}
new_q1_mean <- mean_matrix(new_q1_mats)
new_q2_mean <- mean_matrix(new_q2_mats)

true_blockwise_q <- rbind(
  cbind(sim$truth$Q_y,
        matrix(0L, nrow = sim$truth$J_y, ncol = sim$truth$K_g)),
  cbind(matrix(0L, nrow = sim$truth$J_v, ncol = sim$truth$K_a),
        sim$truth$Q_v)
)
new_blockwise_q <- rbind(
  cbind(new_q1_mean,
        matrix(0, nrow = sim$truth$J_y, ncol = ncol(new_q2_mean))),
  cbind(matrix(0, nrow = sim$truth$J_v, ncol = ncol(new_q1_mean)),
        new_q2_mean)
)
old_blockwise_q <- rbind(old_q_y_mean, old_q_v_mean)

write.csv(old_all, file.path(out_dir, "conventional_cdm_all_bic.csv"), row.names = FALSE)
write.csv(new_all, file.path(out_dir, "eacdm_all_bic.csv"), row.names = FALSE)
write.csv(old_best, file.path(out_dir, "conventional_cdm_best_k_by_replicate.csv"), row.names = FALSE)
write.csv(new_best, file.path(out_dir, "eacdm_best_k_by_replicate.csv"), row.names = FALSE)
write.csv(old_counts, file.path(out_dir, "conventional_cdm_best_k_counts.csv"), row.names = FALSE)
write.csv(new_counts, file.path(out_dir, "eacdm_best_k_counts.csv"), row.names = FALSE)
write.csv(old_q_y_mean, file.path(out_dir, "conventional_cdm_modal_K_Q_y_mean.csv"), row.names = FALSE)
write.csv(old_q_v_mean, file.path(out_dir, "conventional_cdm_modal_K_Q_v_mean.csv"), row.names = FALSE)
write.csv(new_q1_mean, file.path(out_dir, "eacdm_modal_K_Q_y_mean.csv"), row.names = FALSE)
write.csv(new_q2_mean, file.path(out_dir, "eacdm_modal_K_Q_v_mean.csv"), row.names = FALSE)

save_q_plot(
  file.path(out_dir, sprintf("conventional_cdm_modal_K_%d_Q.png", old_modal_K)),
  list(
    list(Q = sim$truth$Q_y, title = "True Q-Y"),
    list(Q = old_q_y_mean, title = sprintf("Conventional CDM Mean Q-Y K=%d", old_modal_K)),
    list(Q = sim$truth$Q_v, title = "True Q-V"),
    list(Q = old_q_v_mean, title = sprintf("Conventional CDM Mean Q-V K=%d", old_modal_K))
  ),
  width = 16,
  height = 6
)

save_q_plot(
  file.path(out_dir, sprintf("eacdm_modal_%s_Q.png", new_modal_combo)),
  list(
    list(Q = sim$truth$Q_y, title = "True Q-Y"),
    list(Q = new_q1_mean, title = sprintf("EACDM Mean Q-Y %s", new_modal_combo)),
    list(Q = sim$truth$Q_v, title = "True Q-V"),
    list(Q = new_q2_mean, title = sprintf("EACDM Mean Q-V %s", new_modal_combo))
  ),
  width = 16,
  height = 6
)

save_blockwise_q_plot(
  file.path(out_dir, "EACDM_blockwise_Q.png"),
  true_q = true_blockwise_q,
  estimated_q = new_blockwise_q,
  estimated_title = "Estimated Q-matrix from the EACDM",
  true_col_break = sim$truth$K_a,
  estimated_col_break = ncol(new_q1_mean)
)

save_blockwise_q_plot(
  file.path(out_dir, "conventional_CDM_blockwise_Q.png"),
  true_q = true_blockwise_q,
  estimated_q = old_blockwise_q,
  estimated_title = "Estimated Q-matrix from the conventional CDM",
  true_col_break = sim$truth$K_a,
  estimated_col_break = NULL
)

summary <- list(
  conventional_cdm_counts = old_counts,
  eacdm_counts = new_counts,
  conventional_cdm_modal_K = old_modal_K,
  eacdm_modal_combo = new_modal_combo,
  conventional_cdm_modal_n = nrow(old_modal_best),
  eacdm_modal_n = nrow(new_modal_best),
  conventional_cdm_q_y_mean = old_q_y_mean,
  conventional_cdm_q_v_mean = old_q_v_mean,
  eacdm_q1_mean = new_q1_mean,
  eacdm_q2_mean = new_q2_mean
)
saveRDS(summary, file.path(out_dir, "modal_k_q_summary.rds"))

cat("Conventional CDM selected K counts:\n")
print(old_counts)
cat("\nEACDM selected K counts:\n")
print(new_counts)
cat("\nConventional CDM modal K:", old_modal_K, "n =", nrow(old_modal_best), "\n")
cat("EACDM modal combo:", new_modal_combo, "n =", nrow(new_modal_best), "\n")
cat("Wrote modal-K Q summaries to", out_dir, "\n")
