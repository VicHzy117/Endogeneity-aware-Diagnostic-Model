manifest <- read.csv(file.path("data", "generated", "manifest.csv"), stringsAsFactors = FALSE)
manifest <- manifest[order(manifest$scenario_id), , drop = FALSE]
fit_values <- 1:5
rows <- vector("list", nrow(manifest) * 100L * 25L)
index <- 1L

for (scenario_index in seq_len(nrow(manifest))) {
  scenario <- manifest[scenario_index, ]
  for (replicate_id in 1:100) {
    for (fit_K2 in fit_values) for (fit_K1 in fit_values) {
      path <- file.path(
        "result", "bic_fits",
        sprintf("scenario_%02d_n%d_J%d_trueK%d", scenario$scenario_id,
                scenario$n, scenario$J_block, scenario$K),
        sprintf("replicate_%03d", replicate_id),
        sprintf("fit_K1_%d_K2_%d.rds", fit_K1, fit_K2)
      )
      if (!file.exists(path)) stop("Missing expected result: ", path)
      x <- readRDS(path)
      rows[[index]] <- data.frame(
        scenario_id = x$scenario_id, replicate_id = x$replicate_id,
        n = x$n, J = x$J, J_block = x$J_block,
        true_K1 = x$true_K1, true_K2 = x$true_K2,
        fit_K1 = x$fit_K1, fit_K2 = x$fit_K2,
        BIC_mod = x$BIC_mod, mean_parameter_count = x$mean_parameter_count,
        elapsed_min = x$elapsed_min, seed = x$seed, path = path
      )
      index <- index + 1L
    }
  }
}

all_fits <- do.call(rbind, rows)
summary_dir <- file.path("result", "summary")
plot_dir <- file.path("result", "plots")
dir.create(summary_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(all_fits, file.path(summary_dir, "bic_all_fits.csv"), row.names = FALSE)

replicate_key <- interaction(all_fits$scenario_id, all_fits$replicate_id, drop = TRUE)
best <- do.call(rbind, lapply(split(all_fits, replicate_key), function(df) {
  true_row <- df[df$fit_K1 == df$true_K1 & df$fit_K2 == df$true_K2, , drop = FALSE]
  wrong <- df[!(df$fit_K1 == df$true_K1 & df$fit_K2 == df$true_K2), , drop = FALSE]
  selected <- df[which.min(df$BIC_mod), , drop = FALSE]
  selected$true_model_BIC <- true_row$BIC_mod[[1L]]
  selected$best_incorrect_BIC <- min(wrong$BIC_mod)
  selected$BIC_margin_best_wrong_minus_true <-
    selected$best_incorrect_BIC - selected$true_model_BIC
  selected
}))
rownames(best) <- NULL
best$correct <- best$fit_K1 == best$true_K1 & best$fit_K2 == best$true_K2
best$correct_K1 <- best$fit_K1 == best$true_K1
best$correct_K2 <- best$fit_K2 == best$true_K2
write.csv(best, file.path(summary_dir, "bic_best_by_replicate.csv"), row.names = FALSE)

scenario_key <- interaction(best$scenario_id, drop = TRUE)
selection_summary <- do.call(rbind, lapply(split(best, scenario_key), function(df) {
  data.frame(
    scenario_id = df$scenario_id[[1L]], n = df$n[[1L]], J = df$J[[1L]],
    J_block = df$J_block[[1L]], true_K1 = df$true_K1[[1L]],
    true_K2 = df$true_K2[[1L]], n_replicates = nrow(df),
    correct_count = sum(df$correct), correct_rate = mean(df$correct),
    K1_correct_rate = mean(df$correct_K1), K2_correct_rate = mean(df$correct_K2),
    median_BIC_margin = median(df$BIC_margin_best_wrong_minus_true),
    min_BIC_margin = min(df$BIC_margin_best_wrong_minus_true)
  )
}))
selection_summary <- selection_summary[order(selection_summary$true_K1,
                                             selection_summary$J), ]
write.csv(selection_summary, file.path(summary_dir, "bic_selection_summary.csv"),
          row.names = FALSE)

selected_counts <- do.call(rbind, lapply(split(best, scenario_key), function(df) {
  grid <- expand.grid(fit_K1 = 1:5, fit_K2 = 1:5)
  grid$count <- mapply(function(k1, k2) {
    sum(df$fit_K1 == k1 & df$fit_K2 == k2)
  }, grid$fit_K1, grid$fit_K2)
  grid$proportion <- grid$count / nrow(df)
  grid$scenario_id <- df$scenario_id[[1L]]
  grid$J <- df$J[[1L]]
  grid$true_K1 <- df$true_K1[[1L]]
  grid$true_K2 <- df$true_K2[[1L]]
  grid
}))
write.csv(selected_counts, file.path(summary_dir, "bic_selected_model_counts.csv"),
          row.names = FALSE)

latex <- c(
  "\\begin{table}[ht]", "\\centering",
  "\\caption{BIC-based latent-dimension recovery for the exact-$Q$ EACDM.}",
  "\\begin{tabular}{rrrrrr}", "\\toprule",
  "n & J & True $K_1$ & True $K_2$ & Correct & Rate \\\\",
  "\\midrule",
  sprintf(
    "%d & %d & %d & %d & %d/%d & %.2f %s",
    selection_summary$n, selection_summary$J,
    selection_summary$true_K1, selection_summary$true_K2,
    selection_summary$correct_count, selection_summary$n_replicates,
    selection_summary$correct_rate, "\\\\"
  ),
  "\\bottomrule", "\\end{tabular}",
  "\\label{tab:eacdm_big_model_selection_exact_q}", "\\end{table}"
)
writeLines(latex, file.path(summary_dir, "bic_model_selection_table.tex"))

draw_heatmaps <- function() {
  old_par <- par(mfrow = c(2L, 3L), mar = c(3.8, 4.0, 3.0, 1.0),
                 oma = c(0, 0, 2.0, 0), family = "serif")
  on.exit(par(old_par), add = TRUE)
  palette <- colorRampPalette(c("white", "#56B4E9", "#0072B2"))(101L)
  for (i in seq_len(nrow(selection_summary))) {
    scenario <- selection_summary[i, ]
    counts <- selected_counts[selected_counts$scenario_id == scenario$scenario_id, ]
    matrix_value <- matrix(0, nrow = 5L, ncol = 5L)
    matrix_value[cbind(counts$fit_K1, counts$fit_K2)] <- counts$proportion
    image(1:5, 1:5, matrix_value, col = palette, zlim = c(0, 1),
          xlab = expression("Selected " * K[1]),
          ylab = expression("Selected " * K[2]), axes = FALSE,
          main = sprintf(
            "J=%d, true (K1,K2)=(%d,%d)", scenario$J,
            scenario$true_K1, scenario$true_K2
          ))
    axis(1, at = 1:5)
    axis(2, at = 1:5, las = 1)
    box()
    for (k1 in 1:5) for (k2 in 1:5) {
      value <- matrix_value[k1, k2]
      if (value > 0) text(k1, k2, sprintf("%.0f", 100 * value),
                          col = if (value > 0.55) "white" else "black", cex = 0.8)
    }
    points(scenario$true_K1, scenario$true_K2, pch = 0, cex = 2.0, lwd = 2)
  }
  mtext("Selected dimension percentages (square marks the truth)",
        side = 3, outer = TRUE, font = 2, line = 0.4)
}

png(file.path(plot_dir, "big_model_selection_heatmaps.png"),
    width = 2400, height = 1550, res = 220)
draw_heatmaps()
dev.off()
pdf(file.path(plot_dir, "big_model_selection_heatmaps.pdf"), width = 11.5, height = 7.5)
draw_heatmaps()
dev.off()

saveRDS(
  list(all_fits = all_fits, best = best, summary = selection_summary,
       counts = selected_counts),
  file.path(summary_dir, "bic_selection_summary.rds")
)
cat("Aggregated 15,000 candidate fits across 600 replicates.\n")
print(selection_summary)
