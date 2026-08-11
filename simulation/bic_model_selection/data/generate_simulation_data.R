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

make_binary_design <- function(K) {
  states <- 0:(2^K - 1L)
  bits <- t(vapply(states, function(x) as.integer(intToBits(x))[seq_len(K)], integer(K)))
  cbind(intercept = 1L, bits)
}

make_q_matrix <- function(K, J) {
  identity_rows <- do.call(rbind, replicate(3L, diag(K), simplify = FALSE))
  pair_rows <- if (K >= 2L) {
    pairs <- utils::combn(seq_len(K), 2L)
    t(apply(pairs, 2L, function(idx) {
      row <- integer(K)
      row[idx] <- 1L
      row
    }))
  } else {
    matrix(integer(0), nrow = 0L, ncol = K)
  }
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
    eta <- rbind(
      intercept_pool[seq_len(K)],
      latent,
      rep(0.8, K)
    )
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
  set.seed(seed + n * 100000L + J_block * 1000L + K * 100L + replicate_id)

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
  alpha_class <- row_match_class(A_cate, A)

  Y <- sample_ordinal_3(A_cate %*% t(B))
  V <- sample_ordinal_3(G_cate %*% t(L))
  colnames(Y) <- paste0("Y", seq_len(J_block))
  colnames(V) <- paste0("V", seq_len(J_block))

  list(
    replicate_id = replicate_id,
    scenario = list(n = n, J_block = J_block, K1 = K, K2 = K),
    Y = Y,
    V = V,
    covariates = covariates,
    alpha_class = alpha_class,
    gamma_class = gamma_class
  )
}

generate_scenario <- function(n, J_block, K, setnum, seed) {
  datasets <- lapply(seq_len(setnum), generate_one_dataset,
                     n = n, K = K, J_block = J_block, seed = seed)
  list(
    datasets = datasets,
    truth = list(
      K1 = K,
      K2 = K,
      J1 = J_block,
      J2 = J_block,
      Q1 = make_q_matrix(K, J_block),
      Q2 = make_q_matrix(K, J_block),
      B = cbind(intercept = rep(-0.2, J_block), make_q_matrix(K, J_block)),
      L = cbind(intercept = rep(-0.2, J_block), make_q_matrix(K, J_block)),
      eta = make_eta(K)
    ),
    metadata = list(
      setnum = setnum,
      n = n,
      J_block = J_block,
      J_total = 2L * J_block,
      K1 = K,
      K2 = K,
      response_levels = 3L,
      response_encoding = "0-based ordinal categories: 0, 1, 2",
      seed = seed,
      seed_formula = "seed + n * 100000 + J_block * 1000 + K * 100 + replicate_id"
    )
  )
}

args <- parse_args()
root <- normalizePath(file.path(script_dir(), ".."), mustWork = TRUE)
out_dir <- arg_value(args, "out_dir", file.path(root, "data", "generated"))
setnum <- as.integer(arg_value(args, "setnum", 100L))
seed <- as.integer(arg_value(args, "seed", 20260517L))
n_values <- as.integer(strsplit(arg_value(args, "n_values", "500,1000,2000"), ",")[[1L]])
j_values <- as.integer(strsplit(arg_value(args, "j_values", "24,36"), ",")[[1L]])
k_values <- as.integer(strsplit(arg_value(args, "k_values", "2,3,4"), ",")[[1L]])

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
manifest <- expand.grid(n = n_values, J_block = j_values, K = k_values)
manifest$scenario_id <- seq_len(nrow(manifest))
manifest$file <- sprintf("scenario_%02d_n%d_J%d_K%d.rds",
                         manifest$scenario_id, manifest$n, manifest$J_block, manifest$K)
manifest$setnum <- setnum
manifest$seed <- seed

for (i in seq_len(nrow(manifest))) {
  row <- manifest[i, ]
  cat("Generating", row$file, "\n")
  sim <- generate_scenario(row$n, row$J_block, row$K, setnum, seed)
  saveRDS(sim, file.path(out_dir, row$file), compress = "xz")
}

write.csv(manifest, file.path(out_dir, "manifest.csv"), row.names = FALSE)
saveRDS(manifest, file.path(out_dir, "manifest.rds"))
cat("Saved simulation data to", normalizePath(out_dir), "\n")
