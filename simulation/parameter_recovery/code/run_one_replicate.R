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

make_perms <- function(x) {
  if (length(x) == 1L) return(matrix(x, nrow = 1L))
  do.call(rbind, lapply(seq_along(x), function(i) cbind(x[i], make_perms(x[-i]))))
}

binary_q <- function(Q, threshold = 0.5) {
  1L * (as.matrix(Q) >= threshold)
}

align_block <- function(Q_est, Q_true, threshold = 0.5) {
  Q_est_bin <- binary_q(Q_est, threshold)
  perms <- make_perms(seq_len(ncol(Q_true)))
  scores <- apply(perms, 1L, function(idx) sum(Q_est_bin[, idx, drop = FALSE] == Q_true))
  best <- perms[which.max(scores), ]
  list(permutation = best, Q_binary = Q_est_bin[, best, drop = FALSE])
}

adjusted_rand_index <- function(x, y) {
  x <- as.factor(x)
  y <- as.factor(y)
  tab <- table(x, y)
  choose2 <- function(z) z * (z - 1) / 2
  sum_tab <- sum(choose2(tab))
  sum_row <- sum(choose2(rowSums(tab)))
  sum_col <- sum(choose2(colSums(tab)))
  n2 <- choose2(length(x))
  expected <- sum_row * sum_col / n2
  max_index <- 0.5 * (sum_row + sum_col)
  if (max_index == expected) return(1)
  (sum_tab - expected) / (max_index - expected)
}

q_row_labels <- function(Q) {
  apply(as.matrix(Q), 1L, paste0, collapse = "")
}

blockdiag_q <- function(Q1, Q2) {
  rbind(
    cbind(Q1, matrix(0L, nrow(Q1), ncol(Q2))),
    cbind(matrix(0L, nrow(Q2), ncol(Q1)), Q2)
  )
}

rmse <- function(est, truth) {
  sqrt(mean((as.matrix(est) - as.matrix(truth))^2))
}

args <- parse_args()
code_dir <- script_dir()
project_dir <- normalizePath(file.path(code_dir, ".."), mustWork = TRUE)
launch_dir <- getwd()
resolve_path <- function(path, default_under_project) {
  if (grepl("^/", path)) return(path)
  launch_candidate <- file.path(launch_dir, path)
  if (file.exists(launch_candidate) || dir.exists(dirname(launch_candidate))) {
    return(normalizePath(launch_candidate, mustWork = FALSE))
  }
  file.path(project_dir, default_under_project)
}
data_arg <- arg_value(args, "data_dir", file.path("data", "generated"))
result_arg <- arg_value(args, "result_dir", "result")
data_dir <- resolve_path(data_arg, data_arg)
result_dir <- resolve_path(result_arg, result_arg)
setwd(project_dir)
iteration <- as.integer(arg_value(args, "iteration", 3000L))
burnin <- as.integer(arg_value(args, "burnin", 2000L))
seed <- as.integer(arg_value(args, "seed", 910000L))
task_id <- as.integer(arg_value(args, "task_id", Sys.getenv("SLURM_ARRAY_TASK_ID", "1")))

manifest <- read.csv(file.path(data_dir, "manifest.csv"))
setnum <- manifest$setnum[1L]
grid <- merge(manifest, data.frame(replicate_id = seq_len(setnum)))
grid <- grid[order(grid$scenario_id, grid$replicate_id), ]
if (task_id < 1L || task_id > nrow(grid)) stop("task_id out of range: ", task_id)
job <- grid[task_id, ]

out_dir <- file.path(result_dir, "fits",
                     sprintf("scenario_%02d_n%d_J%d_K%d",
                             job$scenario_id, job$n, job$J_block, job$K))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_path <- file.path(out_dir, sprintf("replicate_%03d.rds", job$replicate_id))
if (file.exists(out_path) && tolower(arg_value(args, "overwrite", "FALSE")) != "true") {
  existing_ok <- tryCatch({
    existing <- readRDS(out_path)
    isTRUE(existing$metrics$exact_Q_invariant) &&
      identical(as.integer(existing$fit_config$iteration), iteration) &&
      identical(as.integer(existing$fit_config$burnin), burnin) &&
      all(is.finite(unlist(existing$metrics[c(
        "ARI_Q", "RMSE_Delta", "RMSE_eta", "BIC_mod"
      )])))
  }, error = function(e) FALSE)
  if (existing_ok) {
    cat("Valid result exists, skipping:", out_path, "\n")
    quit(save = "no", status = 0L)
  }
  quarantine <- paste0(out_path, ".invalid_", format(Sys.time(), "%Y%m%d_%H%M%S"))
  if (!file.rename(out_path, quarantine)) stop("Could not quarantine invalid result: ", out_path)
  cat("Moved invalid/incompatible result to", quarantine, "\n")
}

source(file.path("code", "new_model_main.R"))
sim <- readRDS(file.path(data_dir, job$file))
dat <- sim$datasets[[job$replicate_id]]
truth <- sim$truth

fit_seed <- seed + job$scenario_id * 100000L + job$replicate_id
set.seed(fit_seed)
cat("Running scenario", job$scenario_id, "replicate", job$replicate_id,
    "n", job$n, "J_block", job$J_block, "K", job$K, "\n")
start_time <- Sys.time()
fit <- ECDM_main(
  Y = dat$Y,
  V = dat$V,
  covarites = dat$covariates,
  K_a = job$K,
  K_g = job$K,
  iteration = iteration
)
elapsed_min <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))

q1_mean <- PostMean(fit$Q1_list, burnin)
q2_mean <- PostMean(fit$Q2_list, burnin)
delta1_mean <- PostMean(fit$B_list, burnin)
delta2_mean <- PostMean(fit$L_list, burnin)
eta_mean <- PostMean(fit$Sita_list, burnin)
pi2_mean <- PostMean(fit$pi2_list, burnin)

align1 <- align_block(q1_mean, truth$Q1)
align2 <- align_block(q2_mean, truth$Q2)
q_full_true <- blockdiag_q(truth$Q1, truth$Q2)
q_full_est <- blockdiag_q(align1$Q_binary, align2$Q_binary)

delta1_aligned <- delta1_mean[, 1L + align1$permutation, drop = FALSE]
delta2_aligned <- delta2_mean[, 1L + align2$permutation, drop = FALSE]
eta_aligned <- eta_mean[c(1L, 1L + align2$permutation, nrow(eta_mean)),
                        align1$permutation, drop = FALSE]
pbic <- PBIC_from_result(fit, dat$Y, dat$V, dat$covariates, job$K, job$K, burnin)

# This invariant is the defining difference from the former SSVS code.
exact_q_ok <- all(vapply(seq_along(fit$B_list), function(i) {
  all(fit$B_list[[i]][, -1L][fit$Q1_list[[i]] == 0] == 0)
}, logical(1L))) && all(vapply(seq_along(fit$L_list), function(i) {
  all(fit$L_list[[i]][, -1L][fit$Q2_list[[i]] == 0] == 0)
}, logical(1L)))

metrics <- data.frame(
  scenario_id = job$scenario_id,
  replicate_id = job$replicate_id,
  n = job$n,
  J = 2L * job$J_block,
  J_block = job$J_block,
  K1 = job$K,
  K2 = job$K,
  ARI_Q = adjusted_rand_index(q_row_labels(q_full_true), q_row_labels(q_full_est)),
  RMSE_Delta = sqrt(mean(c(
    (delta1_aligned - truth$B[, -1L, drop = FALSE])^2,
    (delta2_aligned - truth$L[, -1L, drop = FALSE])^2
  ))),
  RMSE_beta0 = sqrt(mean(c(
    (delta1_mean[, 1L] - truth$B[, 1L])^2,
    (delta2_mean[, 1L] - truth$L[, 1L])^2
  ))),
  RMSE_eta = rmse(eta_aligned, truth$eta),
  BIC_mod = pbic,
  exact_Q_invariant = exact_q_ok,
  elapsed_min = elapsed_min,
  seed = fit_seed
)

out <- list(
  metrics = metrics,
  posterior = list(
    Q1_mean = q1_mean,
    Q2_mean = q2_mean,
    beta01_mean = delta1_mean[, 1L],
    beta02_mean = delta2_mean[, 1L],
    Delta1_mean = delta1_mean[, -1L, drop = FALSE],
    Delta2_mean = delta2_mean[, -1L, drop = FALSE],
    eta_mean = eta_mean,
    pi2_mean = pi2_mean,
    Q1_binary_aligned = align1$Q_binary,
    Q2_binary_aligned = align2$Q_binary,
    Q1_permutation = align1$permutation,
    Q2_permutation = align2$permutation
  ),
  fit_config = list(
    iteration = iteration,
    burnin = burnin,
    likelihood = "exact Q restriction",
    bic_includes_pi2 = TRUE
  )
)
saveRDS(out, out_path)
cat("Saved", out_path, "\n")
print(metrics)
