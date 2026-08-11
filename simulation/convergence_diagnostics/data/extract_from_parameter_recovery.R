# Build the compact 54-dataset convergence bundle from the already validated
# parameter-recovery files. This helper is used locally when preparing the
# convergence bundle when the full parameter-recovery data are already present.
source_dir <- if (length(commandArgs(trailingOnly = TRUE))) {
  commandArgs(trailingOnly = TRUE)[[1L]]
} else {
  file.path("..", "parameter_recovery", "data", "generated")
}
out_dir <- file.path("data", "generated")
manifest_all <- read.csv(file.path(source_dir, "manifest.csv"))
replicate_ids <- c(1L, 50L, 100L)
rows <- list()
z <- 1L
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

for (i in seq_len(nrow(manifest_all))) {
  scenario <- manifest_all[i, ]
  sim <- readRDS(file.path(source_dir, scenario$file))
  for (replicate_id in replicate_ids) {
    dat <- sim$datasets[[replicate_id]]
    dat$dataset_seed <- scenario$seed + scenario$n * 100000L +
      scenario$J_block * 1000L + scenario$K * 100L + replicate_id
    dat$truth <- sim$truth
    file <- sprintf("scenario_%02d_n%d_J%d_K%d_rep%03d.rds",
                    scenario$scenario_id, scenario$n, scenario$J_block,
                    scenario$K, replicate_id)
    saveRDS(dat, file.path(out_dir, file), compress = "xz")
    rows[[z]] <- data.frame(
      n = scenario$n, J_block = scenario$J_block, K = scenario$K,
      scenario_id = scenario$scenario_id, replicate_id = replicate_id,
      diagnostic_dataset_id = z, file = file, base_seed = scenario$seed,
      dataset_seed = dat$dataset_seed
    )
    z <- z + 1L
  }
}
manifest <- do.call(rbind, rows)
manifest <- manifest[order(manifest$scenario_id, manifest$replicate_id), ]
manifest$diagnostic_dataset_id <- seq_len(nrow(manifest))
write.csv(manifest, file.path(out_dir, "manifest.csv"), row.names = FALSE)
saveRDS(manifest, file.path(out_dir, "manifest.rds"))
cat("Extracted", nrow(manifest), "datasets from", normalizePath(source_dir), "\n")
