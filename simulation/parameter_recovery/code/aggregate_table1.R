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

iqr <- function(x) {
  unname(stats::IQR(x, na.rm = TRUE))
}

args <- parse_args()
project_dir <- normalizePath(file.path(script_dir(), ".."), mustWork = TRUE)
setwd(project_dir)

result_dir <- arg_value(args, "result_dir", "result")
allow_incomplete <- tolower(arg_value(args, "allow_incomplete", "false")) == "true"
summary_dir <- file.path(result_dir, "summary")
dir.create(summary_dir, recursive = TRUE, showWarnings = FALSE)

files <- list.files(file.path(result_dir, "fits"), pattern = "\\.rds$",
                    full.names = TRUE, recursive = TRUE)
if (!length(files)) stop("No fit files found under ", file.path(result_dir, "fits"))

rows <- do.call(rbind, lapply(files, function(path) {
  x <- readRDS(path)
  cbind(x$metrics, path = path)
}))
expected_n <- 18L * 100L
if (!allow_incomplete && nrow(rows) != expected_n) {
  stop("Expected ", expected_n, " fit files but found ", nrow(rows),
       ". Run code/audit_results.R and rerun missing tasks, or pass ",
       "--allow_incomplete=true for an explicitly provisional summary.")
}
if (anyDuplicated(rows[c("scenario_id", "replicate_id")])) {
  stop("Duplicate scenario/replicate results detected")
}
if (any(!rows$exact_Q_invariant)) stop("At least one fit violated the exact-Q invariant")
write.csv(rows, file.path(summary_dir, "all_replicate_metrics.csv"), row.names = FALSE)

split_key <- interaction(rows$n, rows$J, rows$K1, rows$K2, drop = TRUE)
table1 <- do.call(rbind, lapply(split(rows, split_key), function(df) {
  data.frame(
    n = df$n[1L],
    J = df$J[1L],
    K1 = df$K1[1L],
    K2 = df$K2[1L],
    n_replicates = nrow(df),
    ARI_Q_median = median(df$ARI_Q, na.rm = TRUE),
    ARI_Q_IQR = iqr(df$ARI_Q),
    RMSE_Delta_median = median(df$RMSE_Delta, na.rm = TRUE),
    RMSE_Delta_IQR = iqr(df$RMSE_Delta),
    RMSE_eta_median = median(df$RMSE_eta, na.rm = TRUE),
    RMSE_eta_IQR = iqr(df$RMSE_eta),
    BIC_mod_median = median(df$BIC_mod, na.rm = TRUE),
    exact_Q_failures = sum(!df$exact_Q_invariant),
    elapsed_min_median = median(df$elapsed_min, na.rm = TRUE)
  )
}))
table1 <- table1[order(table1$K1, table1$n, table1$J), ]
write.csv(table1, file.path(summary_dir, "table1_extended.csv"), row.names = FALSE)

format_metric <- function(med, spread, digits = 3L) {
  sprintf(paste0("%.", digits, "f (%.", digits, "f)"), med, spread)
}
table1_md <- data.frame(
  n = table1$n,
  J = table1$J,
  K1 = table1$K1,
  K2 = table1$K2,
  `ARI(Q) median (IQR)` = format_metric(table1$ARI_Q_median, table1$ARI_Q_IQR),
  `RMSE(Delta) median (IQR)` = format_metric(
    table1$RMSE_Delta_median, table1$RMSE_Delta_IQR
  ),
  `RMSE(eta) median (IQR)` = format_metric(table1$RMSE_eta_median, table1$RMSE_eta_IQR),
  check.names = FALSE
)

md_path <- file.path(summary_dir, "table1_extended.md")
con <- file(md_path, open = "wt")
on.exit(close(con), add = TRUE)
writeLines("| n | J | K1 | K2 | ARI(Q) median (IQR) | RMSE(Delta) median (IQR) | RMSE(eta) median (IQR) |", con)
writeLines("|---:|---:|---:|---:|---:|---:|---:|", con)
for (i in seq_len(nrow(table1_md))) {
  writeLines(sprintf("| %d | %d | %d | %d | %s | %s | %s |",
                     table1_md$n[i], table1_md$J[i], table1_md$K1[i], table1_md$K2[i],
                     table1_md[[5L]][i], table1_md[[6L]][i], table1_md[[7L]][i]), con)
}

saveRDS(list(all_metrics = rows, table1 = table1),
        file.path(summary_dir, "table1_extended_summary.rds"))
cat("Wrote summaries to", normalizePath(summary_dir), "\n")
