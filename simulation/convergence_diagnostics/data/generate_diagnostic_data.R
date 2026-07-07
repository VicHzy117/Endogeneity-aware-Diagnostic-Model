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

make_binary_design <- function(K) {
  states <- 0:(2^K - 1L)
  bits <- t(vapply(states, function(x) as.integer(intToBits(x))[seq_len(K)], integer(K)))
  cbind(intercept = 1L, bits)
}

make_q_matrix <- function(K, J) {
  identity_rows <- do.call(rbind, replicate(3L, diag(K), simplify = FALSE))
  pairs <- utils::combn(seq_len(K), 2L)
  pair_rows <- t(apply(pairs, 2L, function(idx) {
    row <- integer(K)
    row[idx] <- 1L
    row
  }))
  base <- rbind(identity_rows, pair_rows)
  base[rep(seq_len(nrow(base)), length.out = J), , drop = FALSE]
}

make_eta <- function(K) {
  if (K == 3L) {
    eta <- matrix(c(
       0.6, -0.7, -0.6,
       1.0, -1.6,  1.4,
      -0.8,  0.7, -0.8,
      -1.5,  1.5,  0.6,
       0.8,  0.8,  0.8
    ), nrow = 5L, ncol = 3L, byrow = TRUE)
  } else {
    intercept_pool <- c(0.6, -0.7, -0.6, 0.4)
    diag_pool <- c(1.0, 0.7, 0.6, 0.9)
    off_pool <- matrix(c(
       0.0, -1.6,  1.4, -0.6,
      -0.8,  0.0, -0.8,  1.1,
      -1.5,  1.5,  0.0, -0.9,
       0.7, -1.0,  0.8,  0.0
    ), nrow = 4L, byrow = TRUE)
    latent <- off_pool[seq_len(K), seq_len(K), drop = FALSE]
    diag(latent) <- diag_pool[seq_len(K)]
    eta <- rbind(intercept_pool[seq_len(K)], latent, rep(0.8, K))
  }
  rownames(eta) <- c("intercept", paste0("alpha2_", seq_len(K)), "z1")
  colnames(eta) <- paste0("alpha1_", seq_len(K))
  eta
}

sample_ordinal_3 <- function(eta) {
  p0 <- pnorm(-eta)
  p1 <- pnorm(1 - eta) - p0
  u <- matrix(runif(length(eta)), nrow = nrow(eta), ncol = ncol(eta))
  1L * (u >= p0) + 1L * (u >= p0 + p1)
}

row_match_class <- function(class_rows, design) {
  keys <- apply(design, 1L, paste0, collapse = "")
  rows <- apply(class_rows, 1L, paste0, collapse = "")
  match(rows, keys) - 1L
}

generate_one_dataset <- function(replicate_id, n, K, J_block, seed) {
  dataset_seed <- seed + n * 100000L + J_block * 1000L + K * 100L + replicate_id
  set.seed(dataset_seed)

  A <- make_binary_design(K)
  G <- make_binary_design(K)
  Q1 <- make_q_matrix(K, J_block)
  Q2 <- make_q_matrix(K, J_block)
  B <- cbind(intercept = rep(-0.2, J_block), Q1)
  L <- cbind(intercept = rep(-0.2, J_block), Q2)
  eta <- make_eta(K)

  covariates <- matrix(rbinom(n, 1L, 0.5), nrow = n, ncol = 1L)
  colnames(covariates) <- "z1"

  gamma_class <- sample.int(nrow(G), n, replace = TRUE) - 1L
  G_cate <- G[gamma_class + 1L, , drop = FALSE]
  G_catecov <- cbind(G_cate, covariates)
  alpha_prob <- plogis(G_catecov %*% eta)
  A_binary <- matrix(rbinom(n * K, 1L, alpha_prob), nrow = n, ncol = K)
  A_cate <- cbind(intercept = 1L, A_binary)

  list(
    replicate_id = replicate_id,
    dataset_seed = dataset_seed,
    Y = sample_ordinal_3(A_cate %*% t(B)),
    V = sample_ordinal_3(G_cate %*% t(L)),
    covariates = covariates,
    alpha_class = row_match_class(A_cate, A),
    gamma_class = gamma_class,
    truth = list(Q1 = Q1, Q2 = Q2, B = B, L = L, eta = eta)
  )
}

args <- parse_args()
project_dir <- normalizePath(file.path(script_dir(), ".."), mustWork = TRUE)
setwd(project_dir)

out_dir <- arg_value(args, "out_dir", file.path("data", "generated"))
base_seed <- as.integer(arg_value(args, "seed", 20260517L))
n_values <- as.integer(strsplit(arg_value(args, "n_values", "500,1000,2000"), ",")[[1L]])
j_values <- as.integer(strsplit(arg_value(args, "j_values", "24,36"), ",")[[1L]])
k_values <- as.integer(strsplit(arg_value(args, "k_values", "2,3,4"), ",")[[1L]])
replicate_ids <- as.integer(strsplit(arg_value(args, "replicate_ids", "1,50,100"), ",")[[1L]])

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
scenario_grid <- expand.grid(n = n_values, J_block = j_values, K = k_values)
scenario_grid$scenario_id <- seq_len(nrow(scenario_grid))
manifest <- merge(scenario_grid, data.frame(replicate_id = replicate_ids))
manifest <- manifest[order(manifest$scenario_id, manifest$replicate_id), ]
manifest$diagnostic_dataset_id <- seq_len(nrow(manifest))
manifest$file <- sprintf(
  "scenario_%02d_n%d_J%d_K%d_rep%03d.rds",
  manifest$scenario_id, manifest$n, manifest$J_block, manifest$K, manifest$replicate_id
)
manifest$base_seed <- base_seed
manifest$dataset_seed <- base_seed + manifest$n * 100000L +
  manifest$J_block * 1000L + manifest$K * 100L + manifest$replicate_id

for (i in seq_len(nrow(manifest))) {
  row <- manifest[i, ]
  cat("Generating", row$file, "with seed", row$dataset_seed, "\n")
  dat <- generate_one_dataset(
    replicate_id = row$replicate_id,
    n = row$n,
    K = row$K,
    J_block = row$J_block,
    seed = base_seed
  )
  saveRDS(dat, file.path(out_dir, row$file), compress = "xz")
}

write.csv(manifest, file.path(out_dir, "manifest.csv"), row.names = FALSE)
saveRDS(manifest, file.path(out_dir, "manifest.rds"))
cat("Generated", nrow(manifest), "diagnostic datasets in", normalizePath(out_dir), "\n")

