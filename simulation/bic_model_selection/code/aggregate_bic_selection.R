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

args <- parse_args()
project_dir <- normalizePath(file.path(script_dir(), ".."), mustWork = TRUE)
setwd(project_dir)

result_dir <- arg_value(args, "result_dir", "result")
summary_dir <- file.path(result_dir, "summary")
dir.create(summary_dir, recursive = TRUE, showWarnings = FALSE)

files <- list.files(file.path(result_dir, "bic_fits"), pattern = "\\.rds$",
                    full.names = TRUE, recursive = TRUE)
if (!length(files)) stop("No BIC files found under ", file.path(result_dir, "bic_fits"))

rows <- do.call(rbind, lapply(files, function(path) {
  x <- readRDS(path)
  data.frame(
    scenario_id = x$scenario_id,
    replicate_id = x$replicate_id,
    n = x$n,
    J = x$J,
    J_block = x$J_block,
    true_K1 = x$true_K1,
    true_K2 = x$true_K2,
    fit_K1 = x$fit_K1,
    fit_K2 = x$fit_K2,
    bic = x$bic,
    elapsed_min = x$elapsed_min,
    seed = x$seed,
    path = path
  )
}))
write.csv(rows, file.path(summary_dir, "bic_all_fits.csv"), row.names = FALSE)

rep_key <- interaction(rows$scenario_id, rows$replicate_id, drop = TRUE)
best <- do.call(rbind, lapply(split(rows, rep_key), function(df) {
  df[which.min(df$bic), ]
}))
best$correct <- best$fit_K1 == best$true_K1 & best$fit_K2 == best$true_K2
write.csv(best, file.path(summary_dir, "bic_best_by_replicate.csv"), row.names = FALSE)

scenario_key <- interaction(best$scenario_id, drop = TRUE)
summary <- do.call(rbind, lapply(split(best, scenario_key), function(df) {
  data.frame(
    scenario_id = df$scenario_id[1L],
    n = df$n[1L],
    J = df$J[1L],
    J_block = df$J_block[1L],
    true_K1 = df$true_K1[1L],
    true_K2 = df$true_K2[1L],
    n_replicates = nrow(df),
    correct_count = sum(df$correct),
    correct_rate = mean(df$correct)
  )
}))
summary <- summary[order(summary$true_K1, summary$J), ]
write.csv(summary, file.path(summary_dir, "bic_selection_summary.csv"), row.names = FALSE)

counts <- do.call(rbind, lapply(split(best, scenario_key), function(df) {
  out <- as.data.frame(table(paste0("K1_", df$fit_K1, "_K2_", df$fit_K2)))
  names(out) <- c("selected_model", "count")
  out <- out[out$count > 0L, ]
  out$scenario_id <- df$scenario_id[1L]
  out$n <- df$n[1L]
  out$J <- df$J[1L]
  out$J_block <- df$J_block[1L]
  out$true_K1 <- df$true_K1[1L]
  out$true_K2 <- df$true_K2[1L]
  out$proportion <- out$count / nrow(df)
  out
}))
counts <- counts[order(counts$true_K1, counts$J, counts$selected_model), ]
write.csv(counts, file.path(summary_dir, "bic_selected_model_counts.csv"), row.names = FALSE)

saveRDS(list(all_fits = rows, best = best, summary = summary, counts = counts),
        file.path(summary_dir, "bic_selection_summary.rds"))

cat("Number of BIC fit files:", nrow(rows), "\n")
cat("Number of selected-replicate rows:", nrow(best), "\n")
cat("Wrote BIC summaries to", normalizePath(summary_dir), "\n")

