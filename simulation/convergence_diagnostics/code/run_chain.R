parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list()
  for (arg in args) {
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
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
project_dir <- normalizePath(file.path(script_dir(), ".."), mustWork = TRUE)
setwd(project_dir)

data_dir <- arg_value(args, "data_dir", file.path("data", "generated"))
output_dir <- arg_value(args, "output_dir", "output")
iteration <- as.integer(arg_value(args, "iteration", 3000L))
burnin <- as.integer(arg_value(args, "burnin", 2000L))
base_seed <- as.integer(arg_value(args, "seed", 20260610L))
task_id <- as.integer(arg_value(args, "task_id", Sys.getenv("SLURM_ARRAY_TASK_ID", "1")))
chains_per_dataset <- as.integer(arg_value(args, "chains", 4L))
overwrite <- tolower(arg_value(args, "overwrite", "false")) %in% c("1", "true", "yes")

manifest_path <- file.path(data_dir, "manifest.csv")
if (!file.exists(manifest_path)) stop("Missing ", manifest_path, ". Run data generation first.")
manifest <- read.csv(manifest_path)

grid <- merge(manifest, data.frame(chain_id = seq_len(chains_per_dataset)))
grid <- grid[order(grid$scenario_id, grid$replicate_id, grid$chain_id), ]
if (task_id < 1L || task_id > nrow(grid)) {
  stop("task_id must be between 1 and ", nrow(grid), "; received ", task_id)
}
job <- grid[task_id, ]

chain_seed <- base_seed + job$scenario_id * 100000L +
  job$replicate_id * 100L + job$chain_id
chain_dir <- file.path(
  output_dir,
  "chains",
  sprintf("scenario_%02d_n%d_J%d_K%d", job$scenario_id, job$n, job$J_block, job$K),
  sprintf("replicate_%03d", job$replicate_id)
)
dir.create(chain_dir, recursive = TRUE, showWarnings = FALSE)
out_path <- file.path(chain_dir, sprintf("chain_%d.rds", job$chain_id))

if (file.exists(out_path) && !overwrite) {
  cat("Result exists, skipping:", out_path, "\n")
  quit(save = "no", status = 0L)
}

source(file.path("..", "..", "src", "eacdm_model.R"))
dat <- readRDS(file.path(data_dir, job$file))

cat(
  "Convergence task", task_id,
  "scenario", job$scenario_id,
  "replicate", job$replicate_id,
  "chain", job$chain_id,
  "n", job$n,
  "J_block", job$J_block,
  "K", job$K,
  "seed", chain_seed,
  "\n"
)

set.seed(chain_seed)
start_time <- Sys.time()
fit <- ECDM_main(
  Y = dat$Y,
  V = dat$V,
  covarites = dat$covariates,
  K_a = job$K,
  K_g = job$K,
  iteration = iteration,
  verbose_every = 500L,
  keep_categories = FALSE,
  keep_loglik = FALSE
)
elapsed_min <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))

out <- list(
  metadata = list(
    task_id = task_id,
    scenario_id = job$scenario_id,
    replicate_id = job$replicate_id,
    chain_id = job$chain_id,
    n = job$n,
    J_block = job$J_block,
    J_total = 2L * job$J_block,
    K1 = job$K,
    K2 = job$K,
    iteration = iteration,
    burnin = burnin,
    seed = chain_seed,
    data_seed = dat$dataset_seed,
    data_file = job$file,
    elapsed_min = elapsed_min
  ),
  truth = dat$truth,
  fit = fit
)

saveRDS(out, out_path, compress = "xz")
cat("Saved", out_path, "elapsed_min", elapsed_min, "\n")
