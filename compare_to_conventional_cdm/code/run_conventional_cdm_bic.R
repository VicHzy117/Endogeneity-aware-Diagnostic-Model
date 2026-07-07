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
seed <- as.integer(arg_value(args, "seed", 1000L))
task_id <- as.integer(arg_value(args, "task_id", Sys.getenv("SLURM_ARRAY_TASK_ID", "1")))
setnum <- as.integer(arg_value(args, "setnum", 100L))
k_values <- as.integer(strsplit(arg_value(args, "k_values", "2,3,4,5,6"), ",")[[1L]])

grid <- expand.grid(dataset_id = seq_len(setnum), K = k_values)
if (task_id < 1L || task_id > nrow(grid)) stop("task_id out of range: ", task_id)
job <- grid[task_id, ]

dir.create(file.path(result_dir, "conventional_cdm"), recursive = TRUE, showWarnings = FALSE)
source(file.path("..", "src", "regular_cdm_model.R"))

sim <- readRDS(data_path)
dat <- sim$datasets[[job$dataset_id]]
Y_old <- cbind(dat$Y, dat$V)

cat("Conventional CDM dataset", job$dataset_id, "K", job$K, "\n")
start_time <- Sys.time()
fit <- run_regular_cdm(
  Y = Y_old,
  K = job$K,
  Mj = 3L,
  iteration = iteration,
  burnin = burnin,
  interaction_order = 1L,
  seed = seed + job$dataset_id * 100L + job$K
)
elapsed_min <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))

out <- list(
  model = "conventional_cdm",
  dataset_id = job$dataset_id,
  K = job$K,
  iteration = iteration,
  burnin = burnin,
  elapsed_min = elapsed_min,
  bic = fit$criteria$pbic,
  pwaic = fit$criteria$pwaic,
  n_parameters = fit$criteria$n_parameters,
  Q_mean = fit$posterior$Q_mean,
  Q_binary = fit$posterior$Q_binary,
  settings = fit$settings
)

out_path <- file.path(result_dir, "conventional_cdm", sprintf("dataset_%03d_K_%d.rds", job$dataset_id, job$K))
saveRDS(out, out_path)
cat("Saved", out_path, "BIC", out$bic, "elapsed_min", elapsed_min, "\n")
