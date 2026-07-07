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

args <- parse_args()
code_dir <- script_dir()
project_dir <- normalizePath(file.path(code_dir, ".."), mustWork = TRUE)
setwd(project_dir)

data_dir <- arg_value(args, "data_dir", file.path("..", "data", "generated"))
result_dir <- arg_value(args, "result_dir", "result")
iteration <- as.integer(arg_value(args, "iteration", 3000L))
burnin <- as.integer(arg_value(args, "burnin", 2000L))
seed <- as.integer(arg_value(args, "seed", 930000L))
task_id <- as.integer(arg_value(args, "task_id", Sys.getenv("SLURM_ARRAY_TASK_ID", "1")))
fit_k1_values <- as.integer(strsplit(arg_value(args, "fit_k1_values", "1,2,3,4,5"), ",")[[1L]])
fit_k2_values <- as.integer(strsplit(arg_value(args, "fit_k2_values", "1,2,3,4,5"), ",")[[1L]])

manifest_path <- file.path(data_dir, "manifest.csv")
if (!file.exists(manifest_path)) {
  stop("Cannot find ", manifest_path, ". Run the parent data-generation job first.")
}

manifest <- read.csv(manifest_path)
n1000 <- manifest[manifest$n == 1000L, ]
setnum <- manifest$setnum[1L]
grid <- merge(n1000, data.frame(replicate_id = seq_len(setnum)))
grid <- merge(grid, expand.grid(fit_K1 = fit_k1_values, fit_K2 = fit_k2_values))
grid <- grid[order(grid$scenario_id, grid$replicate_id, grid$fit_K1, grid$fit_K2), ]

if (task_id < 1L || task_id > nrow(grid)) stop("task_id out of range: ", task_id)
job <- grid[task_id, ]

out_dir <- file.path(
  result_dir,
  "bic_fits",
  sprintf("scenario_%02d_n%d_J%d_trueK%d", job$scenario_id, job$n, job$J_block, job$K),
  sprintf("replicate_%03d", job$replicate_id)
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_path <- file.path(out_dir, sprintf("fit_K1_%d_K2_%d.rds", job$fit_K1, job$fit_K2))

if (file.exists(out_path) && tolower(arg_value(args, "overwrite", "FALSE")) != "true") {
  cat("Result exists, skipping:", out_path, "\n")
  quit(save = "no", status = 0L)
}

source(file.path("..", "..", "src", "eacdm_model.R"))
sim <- readRDS(file.path(data_dir, job$file))
dat <- sim$datasets[[job$replicate_id]]

fit_seed <- seed + job$scenario_id * 1000000L + job$replicate_id * 1000L +
  job$fit_K1 * 10L + job$fit_K2
set.seed(fit_seed)

cat(
  "Running BIC task", task_id,
  "scenario", job$scenario_id,
  "replicate", job$replicate_id,
  "true K", job$K,
  "J_block", job$J_block,
  "fit K1", job$fit_K1,
  "fit K2", job$fit_K2,
  "\n"
)

start_time <- Sys.time()
fit <- ECDM_main(
  Y = dat$Y,
  V = dat$V,
  covarites = dat$covariates,
  K_a = job$fit_K1,
  K_g = job$fit_K2,
  iteration = iteration
)
elapsed_min <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
bic <- PBIC_from_result(fit, dat$Y, dat$V, dat$covariates, job$fit_K1, job$fit_K2, burnin)

out <- list(
  scenario_id = job$scenario_id,
  replicate_id = job$replicate_id,
  n = job$n,
  J = 2L * job$J_block,
  J_block = job$J_block,
  true_K1 = job$K,
  true_K2 = job$K,
  fit_K1 = job$fit_K1,
  fit_K2 = job$fit_K2,
  iteration = iteration,
  burnin = burnin,
  bic = bic,
  elapsed_min = elapsed_min,
  seed = fit_seed,
  data_file = job$file
)
saveRDS(out, out_path)
cat("Saved", out_path, "BIC", bic, "elapsed_min", elapsed_min, "seed", fit_seed, "\n")
