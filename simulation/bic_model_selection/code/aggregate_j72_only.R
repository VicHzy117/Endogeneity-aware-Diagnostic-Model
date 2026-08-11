all_fits_path <- file.path("result", "summary", "bic_all_fits.csv")
if (!file.exists(all_fits_path)) stop("Missing ", all_fits_path)

out_dir <- file.path("result", "j72_only")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

all_fits <- read.csv(all_fits_path, stringsAsFactors = FALSE)
fits <- all_fits[all_fits$n == 1000L & all_fits$J == 72L, , drop = FALSE]
if (nrow(fits) != 7500L) {
  stop("Expected 7,500 J=72 candidate fits, found ", nrow(fits), ".")
}

replicate_key <- interaction(fits$scenario_id, fits$replicate_id, drop = TRUE)
best <- do.call(rbind, lapply(split(fits, replicate_key), function(df) {
  true_row <- df[df$fit_K1 == df$true_K1 & df$fit_K2 == df$true_K2, , drop = FALSE]
  wrong <- df[!(df$fit_K1 == df$true_K1 & df$fit_K2 == df$true_K2), , drop = FALSE]
  selected <- df[which.min(df$BIC_mod), , drop = FALSE]
  best_wrong <- wrong[which.min(wrong$BIC_mod), , drop = FALSE]
  selected$true_model_BIC <- true_row$BIC_mod[[1L]]
  selected$best_incorrect_BIC <- best_wrong$BIC_mod[[1L]]
  selected$BIC_margin_best_wrong_minus_true <-
    best_wrong$BIC_mod[[1L]] - true_row$BIC_mod[[1L]]
  selected$best_wrong_K1 <- best_wrong$fit_K1[[1L]]
  selected$best_wrong_K2 <- best_wrong$fit_K2[[1L]]
  selected
}))
rownames(best) <- NULL
best$correct <- best$fit_K1 == best$true_K1 & best$fit_K2 == best$true_K2

scenario_key <- interaction(best$scenario_id, drop = TRUE)
summary <- do.call(rbind, lapply(split(best, scenario_key), function(df) {
  data.frame(
    scenario_id = df$scenario_id[[1L]], n = 1000L, J = 72L,
    true_K1 = df$true_K1[[1L]], true_K2 = df$true_K2[[1L]],
    n_replicates = nrow(df), correct_count = sum(df$correct),
    correct_rate = mean(df$correct),
    min_BIC_margin = min(df$BIC_margin_best_wrong_minus_true),
    Q1_BIC_margin = unname(quantile(df$BIC_margin_best_wrong_minus_true, 0.25)),
    median_BIC_margin = median(df$BIC_margin_best_wrong_minus_true),
    Q3_BIC_margin = unname(quantile(df$BIC_margin_best_wrong_minus_true, 0.75)),
    max_BIC_margin = max(df$BIC_margin_best_wrong_minus_true)
  )
}))
summary <- summary[order(summary$true_K1), , drop = FALSE]

selected_counts <- do.call(rbind, lapply(split(best, scenario_key), function(df) {
  grid <- expand.grid(fit_K1 = 1:5, fit_K2 = 1:5)
  grid$count <- mapply(function(k1, k2) {
    sum(df$fit_K1 == k1 & df$fit_K2 == k2)
  }, grid$fit_K1, grid$fit_K2)
  grid$proportion <- grid$count / nrow(df)
  grid$true_K1 <- df$true_K1[[1L]]
  grid$true_K2 <- df$true_K2[[1L]]
  grid
}))

best_wrong_counts <- do.call(rbind, lapply(split(best, scenario_key), function(df) {
  tab <- as.data.frame(table(df$best_wrong_K1, df$best_wrong_K2),
                       stringsAsFactors = FALSE)
  names(tab) <- c("best_wrong_K1", "best_wrong_K2", "count")
  tab <- tab[tab$count > 0L, , drop = FALSE]
  tab$true_K1 <- df$true_K1[[1L]]
  tab$true_K2 <- df$true_K2[[1L]]
  tab$proportion <- tab$count / nrow(df)
  tab
}))

write.csv(fits, file.path(out_dir, "bic_all_fits_J72.csv"), row.names = FALSE)
write.csv(best, file.path(out_dir, "bic_best_by_replicate_J72.csv"), row.names = FALSE)
write.csv(summary, file.path(out_dir, "bic_selection_summary_J72.csv"), row.names = FALSE)
write.csv(selected_counts, file.path(out_dir, "bic_selected_model_counts_J72.csv"),
          row.names = FALSE)
write.csv(best_wrong_counts, file.path(out_dir, "bic_best_wrong_counts_J72.csv"),
          row.names = FALSE)

latex <- c(
  "\\begin{table}[ht]", "\\centering",
  "\\caption{BIC-based latent-dimension recovery for the exact-$Q$ EACDM with $n=1000$ and $J=72$.}",
  "\\begin{tabular}{rrrrrrr}", "\\toprule",
  "True $K_1$ & True $K_2$ & Correct & Rate & Minimum margin & Median margin & Maximum margin \\\\",
  "\\midrule",
  sprintf(
    "%d & %d & %d/%d & %.2f & %.1f & %.1f & %.1f %s",
    summary$true_K1, summary$true_K2, summary$correct_count,
    summary$n_replicates, summary$correct_rate, summary$min_BIC_margin,
    summary$median_BIC_margin, summary$max_BIC_margin, "\\\\"
  ),
  "\\bottomrule", "\\end{tabular}",
  "\\label{tab:eacdm_big_model_selection_j72}", "\\end{table}"
)
writeLines(latex, file.path(out_dir, "bic_model_selection_J72.tex"))

draw_heatmaps <- function() {
  old <- par(mfrow = c(1L, 3L), mar = c(4.0, 4.0, 3.0, 1.0),
             oma = c(0, 0, 2.0, 0), family = "serif")
  on.exit(par(old), add = TRUE)
  palette <- colorRampPalette(c("white", "#56B4E9", "#0072B2"))(101L)
  for (i in seq_len(nrow(summary))) {
    one <- summary[i, ]
    counts <- selected_counts[selected_counts$true_K1 == one$true_K1, ]
    values <- matrix(0, nrow = 5L, ncol = 5L)
    values[cbind(counts$fit_K1, counts$fit_K2)] <- counts$proportion
    image(1:5, 1:5, values, col = palette, zlim = c(0, 1), axes = FALSE,
          xlab = expression("Selected " * K[1]),
          ylab = expression("Selected " * K[2]),
          main = sprintf("True (K1,K2)=(%d,%d)", one$true_K1, one$true_K2))
    axis(1, at = 1:5)
    axis(2, at = 1:5, las = 1)
    box()
    points(one$true_K1, one$true_K2, pch = 0, cex = 2.2, lwd = 2)
    for (k1 in 1:5) for (k2 in 1:5) {
      value <- values[k1, k2]
      if (value > 0) text(k1, k2, sprintf("%.0f", 100 * value),
                          col = if (value > 0.55) "white" else "black",
                          font = 2, cex = 1.0)
    }
  }
  mtext("Selected dimension percentages: n=1000, J=72",
        side = 3, outer = TRUE, line = 0.4, font = 2)
}

png(file.path(out_dir, "big_model_selection_heatmaps_J72.png"),
    width = 2100, height = 750, res = 220)
draw_heatmaps()
dev.off()
pdf(file.path(out_dir, "big_model_selection_heatmaps_J72.pdf"),
    width = 11.5, height = 4.2, family = "Times")
draw_heatmaps()
dev.off()

draw_margins <- function() {
  labels <- sprintf("True K=%d", sort(unique(best$true_K1)))
  boxplot(
    BIC_margin_best_wrong_minus_true ~ true_K1, data = best,
    names = labels, col = "grey88", border = "grey20",
    ylab = "BIC margin: best incorrect minus true",
    xlab = "True latent dimension", las = 1
  )
  abline(h = 0, lty = 2, col = "#D55E00", lwd = 1.5)
  grid(nx = NA, ny = NULL, col = "grey92")
  box()
}

png(file.path(out_dir, "bic_margin_J72.png"),
    width = 1600, height = 1000, res = 220)
draw_margins()
dev.off()
pdf(file.path(out_dir, "bic_margin_J72.pdf"),
    width = 7.5, height = 4.8, family = "Times")
draw_margins()
dev.off()

saveRDS(
  list(all_fits = fits, best = best, summary = summary,
       selected_counts = selected_counts, best_wrong_counts = best_wrong_counts),
  file.path(out_dir, "bic_selection_J72.rds")
)

cat("J=72 exact-Q big model selection: 7,500 fits and 300 replicates.\n")
print(summary)
