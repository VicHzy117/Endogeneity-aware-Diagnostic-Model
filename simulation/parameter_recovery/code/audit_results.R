parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list()
  for (arg in args) {
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    if (length(kv) == 2L) out[[kv[1L]]] <- kv[2L]
  }
  out
}

collapse_ranges <- function(x) {
  if (!length(x)) return("")
  breaks <- c(TRUE, diff(x) != 1L)
  group <- cumsum(breaks)
  paste(vapply(split(x, group), function(z) {
    if (length(z) == 1L) as.character(z) else paste0(z[1L], "-", z[length(z)])
  }, character(1L)), collapse = ",")
}

args <- parse_args()
data_dir <- if (is.null(args$data_dir)) file.path("data", "generated") else args$data_dir
result_dir <- if (is.null(args$result_dir)) "result" else args$result_dir
manifest <- read.csv(file.path(data_dir, "manifest.csv"))
grid <- merge(manifest, data.frame(replicate_id = seq_len(manifest$setnum[1L])))
grid <- grid[order(grid$scenario_id, grid$replicate_id), ]

expected_paths <- file.path(
  result_dir, "fits",
  sprintf("scenario_%02d_n%d_J%d_K%d", grid$scenario_id, grid$n, grid$J_block, grid$K),
  sprintf("replicate_%03d.rds", grid$replicate_id)
)
missing <- which(!file.exists(expected_paths))
bad <- integer(0)
present <- setdiff(seq_len(nrow(grid)), missing)
for (id in present) {
  ok <- tryCatch({
    x <- readRDS(expected_paths[id])
    isTRUE(x$metrics$exact_Q_invariant) && all(is.finite(unlist(x$metrics[c(
      "ARI_Q", "RMSE_Delta", "RMSE_eta", "BIC_mod"
    )])))
  }, error = function(e) FALSE)
  if (!ok) bad <- c(bad, id)
}
rerun <- sort(unique(c(missing, bad)))
spec <- collapse_ranges(rerun)
dir.create(file.path(result_dir, "summary"), recursive = TRUE, showWarnings = FALSE)
writeLines(spec, file.path(result_dir, "summary", "rerun_array_spec.txt"))
cat("Expected:", nrow(grid), " Completed valid:", nrow(grid) - length(rerun),
    " Missing:", length(missing), " Invalid:", length(bad), "\n")
if (length(rerun)) {
  cat("Rerun array specification written to result/summary/rerun_array_spec.txt\n")
  quit(save = "no", status = 2L)
}
cat("All 1800 fits are present and valid.\n")
