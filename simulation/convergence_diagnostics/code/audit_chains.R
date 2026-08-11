args <- commandArgs(trailingOnly = TRUE)
iteration <- if (length(args) >= 1L) as.integer(args[[1L]]) else 3000L
burnin <- if (length(args) >= 2L) as.integer(args[[2L]]) else 2000L
manifest <- read.csv(file.path("data", "generated", "manifest.csv"))
grid <- merge(manifest, data.frame(chain_id = 1:4))
grid <- grid[order(grid$scenario_id, grid$replicate_id, grid$chain_id), ]

check_one <- function(i) {
  job <- grid[i, ]
  path <- file.path(
    "output", "chains",
    sprintf("scenario_%02d_n%d_J%d_K%d", job$scenario_id, job$n, job$J_block, job$K),
    sprintf("replicate_%03d", job$replicate_id), sprintf("chain_%d.rds", job$chain_id)
  )
  reason <- "ok"
  valid <- if (!file.exists(path)) {
    reason <- "missing"; FALSE
  } else tryCatch({
    x <- readRDS(path)
    checks <- c(
      identical(as.integer(x$metadata$iteration), iteration),
      identical(as.integer(x$metadata$burnin), burnin),
      identical(as.integer(x$metadata$chain_id), as.integer(job$chain_id)),
      isTRUE(x$metadata$exact_Q_invariant),
      length(x$fit$Q1_list) == iteration + 1L,
      length(x$fit$Q2_list) == iteration + 1L,
      length(x$fit$B_list) == iteration + 1L,
      length(x$fit$L_list) == iteration + 1L,
      length(x$fit$Sita_list) == iteration + 1L
    )
    if (!all(checks)) reason <<- "incompatible_or_incomplete"
    all(checks)
  }, error = function(e) { reason <<- paste0("read_error: ", conditionMessage(e)); FALSE })
  data.frame(task_id = i, scenario_id = job$scenario_id,
             replicate_id = job$replicate_id, chain_id = job$chain_id,
             valid = valid, reason = reason, path = path)
}

status <- do.call(rbind, lapply(seq_len(nrow(grid)), check_one))
dir.create(file.path("output", "diagnostics"), recursive = TRUE, showWarnings = FALSE)
write.csv(status, file.path("output", "diagnostics", "chain_audit.csv"), row.names = FALSE)
writeLines(as.character(status$task_id[!status$valid]),
           file.path("output", "diagnostics", "missing_task_ids.txt"))
cat(sum(status$valid), "of", nrow(status), "chains valid;",
    sum(!status$valid), "need running.\n")
if (any(!status$valid)) quit(save = "no", status = 2L)
