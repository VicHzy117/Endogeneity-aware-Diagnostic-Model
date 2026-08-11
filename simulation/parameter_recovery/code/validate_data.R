parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list()
  for (arg in args) {
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    if (length(kv) == 2L) out[[kv[1L]]] <- kv[2L]
  }
  out
}

args <- parse_args()
data_dir <- if (is.null(args$data_dir)) file.path("data", "generated") else args$data_dir
manifest_path <- file.path(data_dir, "manifest.csv")
if (!file.exists(manifest_path)) stop("Missing ", manifest_path)

manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
expected <- expand.grid(
  n = c(500L, 1000L, 2000L),
  J_block = c(24L, 36L),
  K = c(2L, 3L, 4L)
)
if (nrow(manifest) != 18L || !all(manifest$setnum == 100L)) {
  stop("Expected 18 scenarios and 100 replicates per scenario")
}
key <- function(x) paste(x$n, x$J_block, x$K, sep = ":")
if (!setequal(key(manifest), key(expected))) stop("Scenario grid does not match the paper")

eta_truth <- list(
  `2` = matrix(c(
     0.6, -0.7,
     1.0, -1.6,
    -0.8,  0.7,
     0.8,  0.8
  ), nrow = 4L, byrow = TRUE),
  `3` = matrix(c(
     0.6, -0.7, -0.6,
     1.0, -1.6,  1.4,
    -0.8,  0.7, -0.8,
    -1.5,  1.5,  0.6,
     0.8,  0.8,  0.8
  ), nrow = 5L, byrow = TRUE),
  `4` = matrix(c(
     0.6, -0.7, -0.6,  0.4,
     1.0, -1.6,  1.4, -0.6,
    -0.8,  0.7, -0.8,  1.1,
    -1.5,  1.5,  0.6, -0.9,
     0.7, -1.0,  0.8,  0.9,
     0.8,  0.8,  0.8,  0.8
  ), nrow = 6L, byrow = TRUE)
)

missing <- manifest$file[!file.exists(file.path(data_dir, manifest$file))]
if (length(missing)) stop("Missing scenario files: ", paste(missing, collapse = ", "))

for (i in seq_len(nrow(manifest))) {
  sim <- readRDS(file.path(data_dir, manifest$file[i]))
  if (length(sim$datasets) != 100L) stop(manifest$file[i], " does not contain 100 datasets")
  if (!identical(as.integer(sim$truth$K1), as.integer(manifest$K[i])) ||
      !identical(as.integer(sim$truth$K2), as.integer(manifest$K[i]))) {
    stop("Truth dimension mismatch in ", manifest$file[i])
  }
  if (!all(sim$truth$B[, -1L, drop = FALSE] == sim$truth$Q1) ||
      !all(sim$truth$L[, -1L, drop = FALSE] == sim$truth$Q2)) {
    stop("The old data are not generated under the exact-Q measurement model: ",
         manifest$file[i])
  }
  K <- as.character(manifest$K[i])
  if (!isTRUE(all.equal(unname(sim$truth$eta), eta_truth[[K]], tolerance = 0))) {
    stop("Structural truth does not match the supplementary material: ", manifest$file[i])
  }
  first <- sim$datasets[[1L]]
  if (!identical(dim(first$Y), c(manifest$n[i], manifest$J_block[i])) ||
      !identical(dim(first$V), c(manifest$n[i], manifest$J_block[i])) ||
      !all(first$Y %in% 0:2) || !all(first$V %in% 0:2)) {
    stop("Response dimensions or ordinal encoding mismatch in ", manifest$file[i])
  }
}

cat("Validated 18 scenarios x 100 replicates (1800 datasets).\n")
cat("The responses were generated with Delta = Q * beta and can be reused.\n")
