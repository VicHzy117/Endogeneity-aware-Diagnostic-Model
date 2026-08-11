path <- if (length(commandArgs(trailingOnly = TRUE))) commandArgs(trailingOnly = TRUE)[[1L]] else
  file.path("data", "simulation_data.rds")
if (!file.exists(path)) stop("Missing ", path)
sim <- readRDS(path)
stopifnot(length(sim$datasets) == 100L)
stopifnot(sim$metadata$n == 1000L, sim$truth$K_a == 3L, sim$truth$K_g == 3L)
stopifnot(sim$truth$J_y == 24L, sim$truth$J_v == 24L)
stopifnot(nrow(sim$truth$gamma) == 4L, ncol(sim$truth$gamma) == 3L)
for (i in seq_along(sim$datasets)) {
  x <- sim$datasets[[i]]
  stopifnot(dim(x$Y)[1L] == 1000L, dim(x$Y)[2L] == 24L)
  stopifnot(dim(x$V)[1L] == 1000L, dim(x$V)[2L] == 24L)
  stopifnot(all(x$Y %in% 0:2), all(x$V %in% 0:2), ncol(x$covariates) == 0L)
}
cat("Validated 100 comparison datasets: n=1000, J1=J2=24, K1=K2=3, M_j=3.\n")
cat("No observed covariates; data seed 237; original DGP retained.\n")
