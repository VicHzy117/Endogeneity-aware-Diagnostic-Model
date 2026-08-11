parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list()
  for (arg in args) {
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1L]]
    if (length(kv) == 2L) out[[kv[[1L]]]] <- kv[[2L]]
  }
  out
}

arg_value <- function(args, name, default) {
  if (!is.null(args[[name]])) args[[name]] else default
}

script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg)) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[[1L]]))))
  }
  getwd()
}

resolve_under_project <- function(project_dir, path) {
  if (grepl("^/", path)) return(path)
  file.path(project_dir, path)
}

candidate_valid <- function(path, job, fit_k1, fit_k2, iteration, burnin) {
  if (!file.exists(path)) return(FALSE)
  tryCatch({
    x <- readRDS(path)
    identical(as.integer(x$scenario_id), as.integer(job$scenario_id)) &&
      identical(as.integer(x$replicate_id), as.integer(job$replicate_id)) &&
      identical(as.integer(x$fit_K1), as.integer(fit_k1)) &&
      identical(as.integer(x$fit_K2), as.integer(fit_k2)) &&
      identical(as.integer(x$iteration), as.integer(iteration)) &&
      identical(as.integer(x$burnin), as.integer(burnin)) &&
      is.finite(x$BIC_mod) &&
      isTRUE(x$exact_Q_invariant) &&
      isTRUE(x$complete_likelihood_includes_class_prevalence) &&
      isTRUE(x$bic_includes_pi2)
  }, error = function(e) FALSE)
}

args <- parse_args()
code_dir <- script_dir()
project_dir <- normalizePath(file.path(code_dir, ".."), mustWork = TRUE)
data_dir <- resolve_under_project(
  project_dir, arg_value(args, "data_dir", file.path("data", "generated"))
)
result_dir <- resolve_under_project(project_dir, arg_value(args, "result_dir", "result"))
iteration <- as.integer(arg_value(args, "iteration", 3000L))
burnin <- as.integer(arg_value(args, "burnin", 2000L))
seed <- as.integer(arg_value(args, "seed", 960000L))
task_id <- as.integer(arg_value(
  args, "task_id", Sys.getenv("SLURM_ARRAY_TASK_ID", "1")
))
fit_k1_values <- as.integer(strsplit(
  arg_value(args, "fit_k1_values", "1,2,3,4,5"), ",", fixed = TRUE
)[[1L]])
fit_k2_values <- as.integer(strsplit(
  arg_value(args, "fit_k2_values", "1,2,3,4,5"), ",", fixed = TRUE
)[[1L]])

if (iteration <= burnin || burnin < 0L) stop("Require iteration > burnin >= 0.")
if (any(fit_k1_values < 1L) || any(fit_k2_values < 1L)) {
  stop("Candidate dimensions must be positive integers.")
}

manifest <- read.csv(file.path(data_dir, "manifest.csv"), stringsAsFactors = FALSE)
manifest <- manifest[order(manifest$scenario_id), , drop = FALSE]
setnum <- unique(manifest$setnum)
if (length(setnum) != 1L) stop("Manifest must have one common replicate count.")
task_grid <- do.call(rbind, lapply(seq_len(nrow(manifest)), function(i) {
  data.frame(manifest_row = i, replicate_id = seq_len(setnum))
}))
if (task_id < 1L || task_id > nrow(task_grid)) {
  stop("task_id must be between 1 and ", nrow(task_grid), ".")
}
task <- task_grid[task_id, , drop = FALSE]
job <- manifest[task$manifest_row, , drop = FALSE]
job$replicate_id <- task$replicate_id

setwd(project_dir)
source(file.path("code", "new_model_main.R"))
simulation <- readRDS(file.path(data_dir, job$file))
data <- simulation$datasets[[job$replicate_id]]

out_dir <- file.path(
  result_dir, "bic_fits",
  sprintf(
    "scenario_%02d_n%d_J%d_trueK%d",
    job$scenario_id, job$n, job$J_block, job$K
  ),
  sprintf("replicate_%03d", job$replicate_id)
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

candidate_grid <- expand.grid(
  fit_K1 = fit_k1_values, fit_K2 = fit_k2_values,
  KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
)

cat(
  "Replicate task", task_id, "of", nrow(task_grid),
  "| scenario", job$scenario_id,
  "| replicate", job$replicate_id,
  "| true K", job$K,
  "| J_block", job$J_block,
  "| candidates", nrow(candidate_grid), "\n"
)

for (candidate_index in seq_len(nrow(candidate_grid))) {
  fit_k1 <- candidate_grid$fit_K1[[candidate_index]]
  fit_k2 <- candidate_grid$fit_K2[[candidate_index]]
  out_path <- file.path(
    out_dir, sprintf("fit_K1_%d_K2_%d.rds", fit_k1, fit_k2)
  )

  if (candidate_valid(out_path, job, fit_k1, fit_k2, iteration, burnin)) {
    cat("Valid candidate exists, skipping:", basename(out_path), "\n")
    next
  }
  if (file.exists(out_path)) {
    quarantine <- paste0(
      out_path, ".invalid_", format(Sys.time(), "%Y%m%d_%H%M%S"),
      "_", Sys.getpid()
    )
    if (!file.rename(out_path, quarantine)) {
      stop("Could not quarantine incompatible result: ", out_path)
    }
  }

  fit_seed <- seed + job$scenario_id * 1000000L +
    job$replicate_id * 10000L + fit_k1 * 100L + fit_k2
  set.seed(fit_seed)
  cat(
    sprintf(
      "Candidate %02d/%02d: fit (K1,K2)=(%d,%d), seed=%d\n",
      candidate_index, nrow(candidate_grid), fit_k1, fit_k2, fit_seed
    )
  )

  start_time <- Sys.time()
  fit <- ECDM_main(
    Y = data$Y, V = data$V, covarites = data$covariates,
    K_a = fit_k1, K_g = fit_k2, iteration = iteration,
    verbose_every = 0L
  )
  elapsed_min <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
  bic <- PBIC_from_result(
    fit, data$Y, data$V, data$covariates, fit_k1, fit_k2, burnin
  )

  exact_q_ok <- all(vapply(seq_along(fit$B_list), function(i) {
    all(fit$B_list[[i]][, -1L, drop = FALSE][fit$Q1_list[[i]] == 0L] == 0)
  }, logical(1L))) && all(vapply(seq_along(fit$L_list), function(i) {
    all(fit$L_list[[i]][, -1L, drop = FALSE][fit$Q2_list[[i]] == 0L] == 0)
  }, logical(1L)))

  keep <- seq.int(burnin + 1L, iteration)
  mean_parameter_count <- mean(
    ncol(data$Y) + ncol(data$V) + 2 * fit$active_q_count[keep] +
      fit_k1 * (fit_k2 + 1L + ncol(data$covariates)) + (2^fit_k2 - 1L)
  )

  out <- list(
    model = "EACDM",
    scenario_id = as.integer(job$scenario_id),
    replicate_id = as.integer(job$replicate_id),
    n = as.integer(job$n),
    J = as.integer(2L * job$J_block),
    J_block = as.integer(job$J_block),
    true_K1 = as.integer(job$K),
    true_K2 = as.integer(job$K),
    fit_K1 = as.integer(fit_k1),
    fit_K2 = as.integer(fit_k2),
    iteration = iteration,
    burnin = burnin,
    BIC_mod = bic,
    mean_parameter_count = mean_parameter_count,
    mean_complete_loglik = sum(rowMeans(
      fit$log_lik_complete[, keep, drop = FALSE]
    )),
    exact_Q_invariant = exact_q_ok,
    complete_likelihood_includes_class_prevalence =
      isTRUE(fit$sampler$pi2_in_complete_loglik),
    bic_includes_pi2 = TRUE,
    likelihood = fit$sampler$likelihood,
    elapsed_min = elapsed_min,
    seed = fit_seed,
    data_file = job$file,
    task_id = task_id
  )
  if (!is.finite(out$BIC_mod) || !isTRUE(out$exact_Q_invariant) ||
      !isTRUE(out$complete_likelihood_includes_class_prevalence)) {
    stop("Candidate fit failed an exact-Q/BIC invariant: ", basename(out_path))
  }

  temporary_path <- paste0(out_path, ".tmp_", Sys.getpid())
  saveRDS(out, temporary_path)
  if (!file.rename(temporary_path, out_path)) {
    stop("Could not atomically install result: ", out_path)
  }
  cat(
    "Saved", basename(out_path),
    "BIC", format(out$BIC_mod, digits = 10),
    "elapsed_min", format(elapsed_min, digits = 4), "\n"
  )
  rm(fit, out)
  invisible(gc())
}

cat("Completed replicate task", task_id, "\n")
