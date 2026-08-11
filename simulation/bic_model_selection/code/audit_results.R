parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list()
  for (arg in args) {
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    if (length(kv) == 2L) out[[kv[[1L]]]] <- kv[[2L]]
  }
  out
}
arg_value <- function(args, name, default) if (!is.null(args[[name]])) args[[name]] else default

args <- parse_args()
data_dir <- arg_value(args, "data_dir", file.path("data", "generated"))
result_dir <- arg_value(args, "result_dir", "result")
iteration <- as.integer(arg_value(args, "iteration", 3000L))
burnin <- as.integer(arg_value(args, "burnin", 2000L))
fit_k1_values <- as.integer(strsplit(arg_value(args, "fit_k1_values", "1,2,3,4,5"), ",")[[1L]])
fit_k2_values <- as.integer(strsplit(arg_value(args, "fit_k2_values", "1,2,3,4,5"), ",")[[1L]])

manifest <- read.csv(file.path(data_dir, "manifest.csv"), stringsAsFactors = FALSE)
manifest <- manifest[order(manifest$scenario_id), , drop = FALSE]
setnum <- unique(manifest$setnum)
task_grid <- do.call(rbind, lapply(seq_len(nrow(manifest)), function(i) {
  data.frame(task_id = (i - 1L) * setnum + seq_len(setnum),
             manifest_row = i, replicate_id = seq_len(setnum))
}))
candidate_grid <- expand.grid(fit_K1 = fit_k1_values, fit_K2 = fit_k2_values)

audit_rows <- vector("list", nrow(task_grid) * nrow(candidate_grid))
index <- 1L
for (task_index in seq_len(nrow(task_grid))) {
  task <- task_grid[task_index, ]
  job <- manifest[task$manifest_row, ]
  for (candidate_index in seq_len(nrow(candidate_grid))) {
    candidate <- candidate_grid[candidate_index, ]
    path <- file.path(
      result_dir, "bic_fits",
      sprintf("scenario_%02d_n%d_J%d_trueK%d", job$scenario_id, job$n,
              job$J_block, job$K),
      sprintf("replicate_%03d", task$replicate_id),
      sprintf("fit_K1_%d_K2_%d.rds", candidate$fit_K1, candidate$fit_K2)
    )
    status <- "missing"
    reason <- "file does not exist"
    if (file.exists(path)) {
      checked <- tryCatch({
        x <- readRDS(path)
        conditions <- c(
          scenario = identical(as.integer(x$scenario_id), as.integer(job$scenario_id)),
          replicate = identical(as.integer(x$replicate_id), as.integer(task$replicate_id)),
          dimensions = identical(as.integer(x$fit_K1), as.integer(candidate$fit_K1)) &&
            identical(as.integer(x$fit_K2), as.integer(candidate$fit_K2)),
          mcmc = identical(as.integer(x$iteration), iteration) &&
            identical(as.integer(x$burnin), burnin),
          finite_bic = length(x$BIC_mod) == 1L && is.finite(x$BIC_mod),
          exact_q = isTRUE(x$exact_Q_invariant),
          pi2_likelihood = isTRUE(x$complete_likelihood_includes_class_prevalence),
          pi2_bic = isTRUE(x$bic_includes_pi2)
        )
        if (all(conditions)) list(status = "valid", reason = "") else
          list(status = "invalid", reason = paste(names(conditions)[!conditions], collapse = ";"))
      }, error = function(e) list(status = "invalid", reason = conditionMessage(e)))
      status <- checked$status
      reason <- checked$reason
    }
    audit_rows[[index]] <- data.frame(
      task_id = task$task_id, scenario_id = job$scenario_id,
      replicate_id = task$replicate_id, true_K = job$K, J_block = job$J_block,
      fit_K1 = candidate$fit_K1, fit_K2 = candidate$fit_K2,
      status = status, reason = reason, path = path,
      stringsAsFactors = FALSE
    )
    index <- index + 1L
  }
}

audit <- do.call(rbind, audit_rows)
summary_dir <- file.path(result_dir, "summary")
dir.create(summary_dir, recursive = TRUE, showWarnings = FALSE)
write.csv(audit, file.path(summary_dir, "result_audit.csv"), row.names = FALSE)
missing <- audit[audit$status != "valid", , drop = FALSE]
write.csv(missing, file.path(summary_dir, "missing_or_invalid_candidates.csv"), row.names = FALSE)
missing_task_ids <- sort(unique(missing$task_id))
writeLines(as.character(missing_task_ids), file.path(summary_dir, "missing_task_ids.txt"))

cat("Valid candidate fits:", sum(audit$status == "valid"), "/", nrow(audit), "\n")
cat("Complete replicate tasks:", 600L - length(missing_task_ids), "/ 600\n")
if (nrow(missing)) {
  cat("Missing/invalid candidates:", nrow(missing), "across", length(missing_task_ids), "tasks.\n")
  quit(save = "no", status = 1L)
}
cat("All exact-Q big-model-selection results passed the audit.\n")
