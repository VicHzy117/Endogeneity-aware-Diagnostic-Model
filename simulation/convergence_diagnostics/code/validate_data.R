args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args)) args[[1L]] else file.path("data", "generated")
manifest_path <- file.path(data_dir, "manifest.csv")
if (!file.exists(manifest_path)) stop("Missing ", manifest_path)
manifest <- read.csv(manifest_path)
stopifnot(nrow(manifest) == 54L)
stopifnot(identical(sort(unique(manifest$replicate_id)), c(1L, 50L, 100L)))
stopifnot(length(unique(manifest$scenario_id)) == 18L)

for (i in seq_len(nrow(manifest))) {
  row <- manifest[i, ]
  path <- file.path(data_dir, row$file)
  if (!file.exists(path)) stop("Missing dataset: ", path)
  dat <- readRDS(path)
  stopifnot(nrow(dat$Y) == row$n, nrow(dat$V) == row$n)
  stopifnot(ncol(dat$Y) == row$J_block, ncol(dat$V) == row$J_block)
  stopifnot(all(dat$Y %in% 0:2), all(dat$V %in% 0:2))
  stopifnot(ncol(dat$truth$Q1) == row$K, ncol(dat$truth$Q2) == row$K)
  expected_seed <- row$base_seed + row$n * 100000L + row$J_block * 1000L +
    row$K * 100L + row$replicate_id
  stopifnot(identical(as.integer(dat$dataset_seed), as.integer(expected_seed)))
}
cat("Validated 18 scenarios x replicates {1,50,100}: 54 datasets.\n")
cat("Each item has M_j = 3 and response encoding 0,1,2.\n")
