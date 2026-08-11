args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args)) args[[1L]] else file.path("data", "generated")
manifest_path <- file.path(data_dir, "manifest.csv")
if (!file.exists(manifest_path)) stop("Missing ", manifest_path)

manifest <- read.csv(manifest_path, stringsAsFactors = FALSE)
expected <- expand.grid(
  n = 1000L, J_block = c(24L, 36L), K = c(2L, 3L, 4L),
  KEEP.OUT.ATTRS = FALSE
)
key <- function(x) paste(x$n, x$J_block, x$K, sep = ":")
if (nrow(manifest) != 6L || !setequal(key(manifest), key(expected))) {
  stop("Expected six n=1000 scenarios with J_block in {24,36} and K in {2,3,4}.")
}
if (!all(manifest$setnum == 100L)) stop("Every scenario must contain 100 replicates.")

missing <- manifest$file[!file.exists(file.path(data_dir, manifest$file))]
if (length(missing)) stop("Missing data files: ", paste(missing, collapse = ", "))

for (i in seq_len(nrow(manifest))) {
  simulation <- readRDS(file.path(data_dir, manifest$file[[i]]))
  if (length(simulation$datasets) != 100L) {
    stop(manifest$file[[i]], " does not contain 100 datasets.")
  }
  truth <- simulation$truth
  if (!identical(as.integer(truth$K1), as.integer(manifest$K[[i]])) ||
      !identical(as.integer(truth$K2), as.integer(manifest$K[[i]]))) {
    stop("Truth dimension mismatch in ", manifest$file[[i]])
  }
  if (!all(truth$B[, -1L, drop = FALSE] == truth$Q1) ||
      !all(truth$L[, -1L, drop = FALSE] == truth$Q2)) {
    stop("Data were not generated with Delta = Q * beta: ", manifest$file[[i]])
  }
  first <- simulation$datasets[[1L]]
  if (!identical(dim(first$Y), c(1000L, manifest$J_block[[i]])) ||
      !identical(dim(first$V), c(1000L, manifest$J_block[[i]])) ||
      !all(first$Y %in% 0:2) || !all(first$V %in% 0:2)) {
    stop("Response dimension or ordinal encoding mismatch in ", manifest$file[[i]])
  }
}

cat("Validated 6 scenarios x 100 replicates (600 reusable datasets).\n")
cat("Every item has M_j = 3 with response encoding 0,1,2.\n")
cat("The responses were generated with Delta = Q * beta; no regeneration is needed.\n")
