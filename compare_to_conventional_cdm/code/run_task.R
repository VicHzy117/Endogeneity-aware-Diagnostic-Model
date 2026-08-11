parse_args <- function() {
  out <- list()
  for (arg in commandArgs(trailingOnly = TRUE)) {
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    if (length(kv) == 2L) out[[kv[1L]]] <- kv[2L]
  }
  out
}
arg_value <- function(args, name, default) if (!is.null(args[[name]])) args[[name]] else default
script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) return(dirname(normalizePath(gsub("~\\+~", " ", sub("^--file=", "", file_arg[[1L]])))))
  getwd()
}

args <- parse_args()
project_dir <- normalizePath(file.path(script_dir(), ".."), mustWork = TRUE)
setwd(project_dir)
task_id <- as.integer(arg_value(args, "task_id", Sys.getenv("SLURM_ARRAY_TASK_ID", "1")))
iteration <- as.integer(arg_value(args, "iteration", 3000L))
burnin <- as.integer(arg_value(args, "burnin", 2000L))
overwrite <- tolower(arg_value(args, "overwrite", "false")) %in% c("1", "true", "yes")
data_path <- arg_value(args, "data", file.path("data", "simulation_data.rds"))
result_dir <- arg_value(args, "result_dir", "result")

grid <- expand.grid(dataset_id = 1:100, K1 = 2:4, K2 = 2:4)
if (task_id < 1L || task_id > nrow(grid)) stop("task_id must be between 1 and 900")
job <- grid[task_id, ]
out_dir <- file.path(result_dir, "eacdm")
out_path <- file.path(out_dir, sprintf("dataset_%03d_K1_%d_K2_%d.rds", job$dataset_id, job$K1, job$K2))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (file.exists(out_path) && !overwrite) {
  valid <- tryCatch({
    x <- readRDS(out_path)
    identical(as.integer(x$iteration), iteration) &&
      identical(as.integer(x$burnin), burnin) && is.finite(x$BIC_mod) &&
      identical(x$likelihood, "exact Q restriction") &&
      isTRUE(x$complete_likelihood_includes_class_prevalence)
  }, error = function(e) FALSE)
  if (valid) {
    cat("Valid EACDM result exists, skipping:", out_path, "\n")
    quit(save = "no", status = 0L)
  }
  quarantine <- paste0(out_path, ".invalid_", format(Sys.time(), "%Y%m%d_%H%M%S"))
  if (!file.rename(out_path, quarantine)) stop("Could not quarantine invalid result: ", out_path)
}

source(file.path("code", "new_model_main.R"))
sim <- readRDS(data_path)
dat <- sim$datasets[[job$dataset_id]]
fit_seed <- 2000L + job$dataset_id * 100L + job$K1 * 10L + job$K2
set.seed(fit_seed)
cat("EACDM dataset", job$dataset_id, "K1", job$K1, "K2", job$K2, "seed", fit_seed, "\n")
start_time <- Sys.time()
fit <- ECDM_main(dat$Y, dat$V, dat$covariates, job$K1, job$K2, iteration)
bic <- PBIC_from_result(fit, dat$Y, dat$V, dat$covariates, job$K1, job$K2, burnin)
q1_mean <- PostMean(fit$Q1_list, burnin)
q2_mean <- PostMean(fit$Q2_list, burnin)
exact_q_ok <- all(vapply(seq_along(fit$B_list), function(i) {
  all(fit$B_list[[i]][, -1L][fit$Q1_list[[i]] == 0L] == 0)
}, logical(1L))) && all(vapply(seq_along(fit$L_list), function(i) {
  all(fit$L_list[[i]][, -1L][fit$Q2_list[[i]] == 0L] == 0)
}, logical(1L)))

out <- list(
  model = "EACDM", dataset_id = job$dataset_id, K1 = job$K1, K2 = job$K2,
  iteration = iteration, burnin = burnin, seed = fit_seed,
  BIC_mod = bic,
  mean_parameter_count = mean(
    ncol(dat$Y) + ncol(dat$V) +
      2 * fit$active_q_count[(burnin + 1L):iteration] +
      job$K1 * (job$K2 + 1L + ncol(dat$covariates)) + (2^job$K2 - 1L)
  ),
  Q1_mean = q1_mean, Q2_mean = q2_mean,
  Q1_binary = 1L * (q1_mean >= 0.5), Q2_binary = 1L * (q2_mean >= 0.5),
  eta_mean = PostMean(fit$Sita_list, burnin),
  exact_Q_invariant = exact_q_ok,
  likelihood = "exact Q restriction",
  complete_likelihood_includes_class_prevalence = TRUE,
  class_prevalence_term = "pi2",
  elapsed_min = as.numeric(difftime(Sys.time(), start_time, units = "mins")),
  task_id = task_id
)
if (!isTRUE(out$exact_Q_invariant) || !is.finite(out$BIC_mod)) stop("Invalid EACDM fit.")
tmp_path <- paste0(out_path, ".tmp_", Sys.getpid())
saveRDS(out, tmp_path, compress = "xz")
if (!file.rename(tmp_path, out_path)) stop("Could not move output to ", out_path)
cat("Saved", out_path, "BIC", out$BIC_mod, "elapsed_min", out$elapsed_min, "\n")
