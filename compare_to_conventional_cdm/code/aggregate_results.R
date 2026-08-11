make_perms <- function(x) {
  if (length(x) == 1L) return(matrix(x, nrow = 1L))
  do.call(rbind, lapply(seq_along(x), function(i) cbind(x[i], make_perms(x[-i]))))
}
align_to_reference <- function(Q, reference) {
  Q <- as.matrix(Q); reference <- as.matrix(reference)
  if (ncol(Q) != ncol(reference)) stop("Alignment requires equal column counts.")
  perms <- make_perms(seq_len(ncol(Q)))
  score <- apply(perms, 1L, function(p) sum((Q[, p, drop = FALSE] - reference)^2))
  Q[, perms[which.min(score), ], drop = FALSE]
}
mean_matrices <- function(x) Reduce(`+`, x) / length(x)
read_expected <- function() {
  e <- do.call(rbind, lapply(1:100, function(id) do.call(rbind, lapply(2:4, function(k1)
    do.call(rbind, lapply(2:4, function(k2) {
      path <- file.path("result", "eacdm", sprintf("dataset_%03d_K1_%d_K2_%d.rds", id, k1, k2))
      x <- readRDS(path)
      data.frame(dataset_id = id, K1 = k1, K2 = k2, BIC_mod = x$BIC_mod,
                 elapsed_min = x$elapsed_min, path = path)
    }))))))
  conventional <- do.call(rbind, lapply(1:100, function(id) do.call(rbind, lapply(2:6, function(k) {
    path <- file.path("baseline", "conventional", sprintf("dataset_%03d_K_%d.rds", id, k))
    x <- readRDS(path)
    data.frame(dataset_id = id, K = k, BIC_mod = x$bic,
               elapsed_min = x$elapsed_min, path = path)
  }))))
  list(eacdm = e, conventional = conventional)
}
best_by_dataset <- function(df) {
  out <- do.call(rbind, lapply(split(df, df$dataset_id), function(x) x[which.min(x$BIC_mod), , drop = FALSE]))
  rownames(out) <- NULL
  out
}
draw_probability_matrix <- function(Q, title) {
  Q <- as.matrix(Q); nr <- nrow(Q); nc <- ncol(Q)
  plot(NA, xlim = c(0.5, nc + 0.5), ylim = c(nr + 0.5, 0.5), xaxt = "n", yaxt = "n",
       xlab = "Latent attribute", ylab = "Item", main = title, bty = "n")
  for (j in seq_len(nr)) for (k in seq_len(nc)) {
    shade <- gray(1 - Q[j, k])
    rect(k - 0.5, j - 0.5, k + 0.5, j + 0.5, col = shade, border = "grey85")
  }
  axis(1, at = seq_len(nc), labels = seq_len(nc), tick = FALSE)
  axis(2, at = seq_len(nr), labels = seq_len(nr), las = 1, cex.axis = 0.55, tick = FALSE)
  box()
}
save_q_figure <- function(path_stem, matrices, titles) {
  draw <- function() {
    old <- par(mfrow = c(1, length(matrices)), mar = c(4, 4, 3, 1), family = "serif")
    on.exit(par(old), add = TRUE)
    for (i in seq_along(matrices)) draw_probability_matrix(matrices[[i]], titles[[i]])
  }
  png(paste0(path_stem, ".png"), width = 1600, height = 700, res = 160)
  draw(); dev.off()
  pdf(paste0(path_stem, ".pdf"), width = 12, height = 5.5)
  draw(); dev.off()
}

dir.create(file.path("result", "summary"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path("result", "plots"), recursive = TRUE, showWarnings = FALSE)
sim <- readRDS(file.path("data", "simulation_data.rds"))
all <- read_expected()
e_best <- best_by_dataset(all$eacdm)
c_best <- best_by_dataset(all$conventional)

e_candidates <- expand.grid(K1 = 2:4, K2 = 2:4)
e_candidates$count <- mapply(function(k1, k2) sum(e_best$K1 == k1 & e_best$K2 == k2),
                             e_candidates$K1, e_candidates$K2)
e_candidates$percent <- e_candidates$count
c_candidates <- data.frame(K = 2:6)
c_candidates$count <- vapply(c_candidates$K, function(k) sum(c_best$K == k), integer(1L))
c_candidates$percent <- c_candidates$count

selection_table <- rbind(
  data.frame(model = "EACDM", candidate = sprintf("(%d,%d)", e_candidates$K1, e_candidates$K2),
             count = e_candidates$count, percent = e_candidates$percent),
  data.frame(model = "Conventional CDM", candidate = sprintf("K=%d", c_candidates$K),
             count = c_candidates$count, percent = c_candidates$percent)
)
write.csv(all$eacdm, file.path("result", "summary", "eacdm_all_bic.csv"), row.names = FALSE)
write.csv(all$conventional, file.path("result", "summary", "conventional_all_bic.csv"), row.names = FALSE)
write.csv(e_best, file.path("result", "summary", "eacdm_best_model_by_replicate.csv"), row.names = FALSE)
write.csv(c_best, file.path("result", "summary", "conventional_best_model_by_replicate.csv"), row.names = FALSE)
write.csv(e_candidates, file.path("result", "summary", "eacdm_selection_counts.csv"), row.names = FALSE)
write.csv(c_candidates, file.path("result", "summary", "conventional_selection_counts.csv"), row.names = FALSE)
write.csv(selection_table, file.path("result", "summary", "paper_model_selection_table.csv"), row.names = FALSE)

latex <- c(
  "\\begin{tabular}{llrr}", "\\hline",
  "Model & Selected dimension & Count & Percent \\\\", "\\hline",
  sprintf("%s & %s & %d & %.1f\\%% \\\\", selection_table$model,
          selection_table$candidate, selection_table$count, selection_table$percent),
  "\\hline", "\\end{tabular}"
)
writeLines(latex, file.path("result", "summary", "paper_model_selection_table.tex"))

modal_e <- e_candidates[which.max(e_candidates$count), ]
modal_c <- c_candidates[which.max(c_candidates$count), ]
e_modal <- e_best[e_best$K1 == modal_e$K1 & e_best$K2 == modal_e$K2, ]
c_modal <- c_best[c_best$K == modal_c$K, ]

e_objects <- lapply(e_modal$path, readRDS)
ref1 <- if (modal_e$K1 == 3L) sim$truth$Q_y else e_objects[[1L]]$Q1_binary
ref2 <- if (modal_e$K2 == 3L) sim$truth$Q_v else e_objects[[1L]]$Q2_binary
e_q1 <- mean_matrices(lapply(e_objects, function(x) align_to_reference(x$Q1_mean, ref1)))
e_q2 <- mean_matrices(lapply(e_objects, function(x) align_to_reference(x$Q2_mean, ref2)))
write.csv(e_q1, file.path("result", "summary", "eacdm_modal_Q1_inclusion_frequency.csv"), row.names = FALSE)
write.csv(e_q2, file.path("result", "summary", "eacdm_modal_Q2_inclusion_frequency.csv"), row.names = FALSE)
save_q_figure(file.path("result", "plots", "eacdm_modal_q_frequency"),
              list(e_q1, e_q2),
              list(sprintf("EACDM Q1: modal (%d,%d), n=%d", modal_e$K1, modal_e$K2, nrow(e_modal)),
                   sprintf("EACDM Q2: modal (%d,%d), n=%d", modal_e$K1, modal_e$K2, nrow(e_modal))))

c_objects <- lapply(c_modal$path, readRDS)
c_ref <- c_objects[[1L]]$Q_binary
c_q <- mean_matrices(lapply(c_objects, function(x) align_to_reference(x$Q_mean, c_ref)))
c_q1 <- c_q[seq_len(sim$truth$J_y), , drop = FALSE]
c_q2 <- c_q[sim$truth$J_y + seq_len(sim$truth$J_v), , drop = FALSE]
write.csv(c_q, file.path("result", "summary", "conventional_modal_Q_inclusion_frequency.csv"), row.names = FALSE)
save_q_figure(file.path("result", "plots", "conventional_modal_q_frequency"),
              list(c_q1, c_q2),
              list(sprintf("Conventional items 1-24: modal K=%d, n=%d", modal_c$K, nrow(c_modal)),
                   sprintf("Conventional items 25-48: modal K=%d, n=%d", modal_c$K, nrow(c_modal))))

true_e <- e_best[e_best$K1 == 3L & e_best$K2 == 3L, ]
e_cell_accuracy <- if (nrow(true_e)) mean(vapply(true_e$path, function(path) {
  x <- readRDS(path)
  q1 <- align_to_reference(x$Q1_binary, sim$truth$Q_y)
  q2 <- align_to_reference(x$Q2_binary, sim$truth$Q_v)
  mean(c(q1 == sim$truth$Q_y, q2 == sim$truth$Q_v))
}, numeric(1L))) else NA_real_

headline <- data.frame(
  quantity = c("EACDM selected (3,3)", "Conventional selected K=6",
               "EACDM Q cell accuracy conditional on selected (3,3)",
               "Total valid EACDM fits", "Total valid conventional fits"),
  value = c(nrow(true_e), sum(c_best$K == 6L), e_cell_accuracy, nrow(all$eacdm), nrow(all$conventional))
)
write.csv(headline, file.path("result", "summary", "headline_results.csv"), row.names = FALSE)
saveRDS(list(eacdm_all = all$eacdm, conventional_all = all$conventional,
             eacdm_best = e_best, conventional_best = c_best,
             selection_table = selection_table, headline = headline),
        file.path("result", "summary", "model_comparison_summary.rds"))
cat("EACDM selection counts:\n"); print(e_candidates)
cat("Conventional selection counts:\n"); print(c_candidates)
cat("Saved comparison summaries and modal-Q figures.\n")
