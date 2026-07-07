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

data_path <- arg_value(args, "data", file.path("data", "simulation_data.rds"))
result_dir <- arg_value(args, "result_dir", "result")
iteration <- as.integer(arg_value(args, "iteration", 3000L))
burnin <- as.integer(arg_value(args, "burnin", 2000L))
seed <- as.integer(arg_value(args, "seed", 2000L))
task_id <- as.integer(arg_value(args, "task_id", Sys.getenv("SLURM_ARRAY_TASK_ID", "1")))
setnum <- as.integer(arg_value(args, "setnum", 100L))
k1_values <- as.integer(strsplit(arg_value(args, "k1_values", "2,3,4"), ",")[[1L]])
k2_values <- as.integer(strsplit(arg_value(args, "k2_values", "2,3,4"), ",")[[1L]])

grid <- expand.grid(dataset_id = seq_len(setnum), K_a = k1_values, K_g = k2_values)
if (task_id < 1L || task_id > nrow(grid)) stop("task_id out of range: ", task_id)
job <- grid[task_id, ]

dir.create(file.path(result_dir, "eacdm"), recursive = TRUE, showWarnings = FALSE)
source(file.path("..", "src", "eacdm_model.R"))

sim <- readRDS(data_path)
dat <- sim$datasets[[job$dataset_id]]

cat("EACDM dataset", job$dataset_id, "K_a", job$K_a, "K_g", job$K_g, "\n")
start_time <- Sys.time()
fit_seed <- seed + job$dataset_id * 100L + job$K_a * 10L + job$K_g
set.seed(fit_seed)
fit <- ECDM_main(
  Y = dat$Y,
  V = dat$V,
  covarites = dat$covariates,
  K_a = job$K_a,
  K_g = job$K_g,
  iteration = iteration
)
elapsed_min <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
bic <- PBIC_from_result(fit, dat$Y, dat$V, dat$covariates, job$K_a, job$K_g, burnin)
q1_mean <- PostMean(fit$Q1_list, burnin + 1L)
q2_mean <- PostMean(fit$Q2_list, burnin + 1L)

out <- list(
  model = "eacdm",
  dataset_id = job$dataset_id,
  K_a = job$K_a,
  K_g = job$K_g,
  iteration = iteration,
  burnin = burnin,
  elapsed_min = elapsed_min,
  bic = bic,
  Q1_mean = q1_mean,
  Q2_mean = q2_mean,
  Q1_binary = 1L * (q1_mean >= 0.5),
  Q2_binary = 1L * (q2_mean >= 0.5),
  Sita_mean = PostMean(fit$Sita_list, burnin + 1L),
  seed = fit_seed
)

out_path <- file.path(result_dir, "eacdm",
                      sprintf("dataset_%03d_K1_%d_K2_%d.rds", job$dataset_id, job$K_a, job$K_g))
saveRDS(out, out_path)
cat("Saved", out_path, "BIC", out$bic, "elapsed_min", elapsed_min, "\n")
