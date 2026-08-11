args <- commandArgs(trailingOnly = TRUE)
iteration <- if (length(args) >= 1L) as.integer(args[[1L]]) else 3000L
burnin <- if (length(args) >= 2L) as.integer(args[[2L]]) else 2000L
grid <- expand.grid(dataset_id = 1:100, K1 = 2:4, K2 = 2:4)
grid$task_id <- seq_len(nrow(grid))

check_new <- function(i) {
  row <- grid[i, ]
  path <- file.path("result", "eacdm",
                    sprintf("dataset_%03d_K1_%d_K2_%d.rds", row$dataset_id, row$K1, row$K2))
  reason <- "ok"
  valid <- if (!file.exists(path)) { reason <- "missing"; FALSE } else tryCatch({
    x <- readRDS(path)
    ok <- identical(as.integer(x$iteration), iteration) &&
      identical(as.integer(x$burnin), burnin) && is.finite(x$BIC_mod) &&
      isTRUE(x$exact_Q_invariant) && identical(x$likelihood, "exact Q restriction") &&
      isTRUE(x$complete_likelihood_includes_class_prevalence)
    if (!ok) reason <<- "incompatible_or_invalid"
    ok
  }, error = function(err) { reason <<- paste0("read_error: ", conditionMessage(err)); FALSE })
  data.frame(task_id = row$task_id, dataset_id = row$dataset_id,
             K1 = row$K1, K2 = row$K2, valid = valid, reason = reason, path = path)
}

status <- do.call(rbind, lapply(seq_len(nrow(grid)), check_new))
baseline_files <- list.files(file.path("baseline", "conventional"),
                             pattern = "^dataset_[0-9]{3}_K_[2-6][.]rds$", full.names = TRUE)
baseline_ok <- length(baseline_files) == 500L && all(vapply(baseline_files, function(path) {
  tryCatch({ x <- readRDS(path); is.finite(x$bic) && x$iteration == 3000L && x$burnin == 2000L },
           error = function(e) FALSE)
}, logical(1L)))

dir.create(file.path("result", "summary"), recursive = TRUE, showWarnings = FALSE)
write.csv(status, file.path("result", "summary", "eacdm_result_audit.csv"), row.names = FALSE)
writeLines(as.character(status$task_id[!status$valid]),
           file.path("result", "summary", "missing_task_ids.txt"))
cat(sum(status$valid), "of 900 new EACDM results valid;",
    sum(!status$valid), "need running.\n")
cat(length(baseline_files), "of 500 original conventional baseline results present.\n")
if (!baseline_ok) stop("Original conventional baseline is incomplete or invalid.")
if (any(!status$valid)) quit(save = "no", status = 2L)
