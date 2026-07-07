#include <RcppArmadillo.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

namespace {

// Small numerical constants used by the Polya-Gamma sampler and log-likelihood.
constexpr double TRUNC = 0.64;
constexpr double TINY = 1e-300;

inline double square(double x) { return x * x; }

inline double log_bernoulli_eta(double eta, double y) {
  // log Bernoulli(y | logit^{-1}(eta)), stable for large |eta|.
  if (y > 0.5) {
    return (eta >= 0.0) ? -std::log1p(std::exp(-eta)) : eta - std::log1p(std::exp(eta));
  }
  return (eta >= 0.0) ? -eta - std::log1p(std::exp(-eta)) : -std::log1p(std::exp(eta));
}

inline double safe_log(double x) {
  return std::log(std::max(x, TINY));
}

uword sample_log_prob(const vec& log_prob) {
  // Sample from unnormalized log-probabilities with the log-sum-exp trick.
  // This replaces repeated calls to R's sample() inside the MCMC loops.
  const double max_log = log_prob.max();
  double total = 0.0;
  std::vector<double> weights(log_prob.n_elem);

  for (uword i = 0; i < log_prob.n_elem; ++i) {
    weights[i] = std::exp(log_prob(i) - max_log);
    total += weights[i];
  }

  const double draw = R::runif(0.0, total);
  double cumsum = 0.0;
  for (uword i = 0; i < log_prob.n_elem; ++i) {
    cumsum += weights[i];
    if (draw <= cumsum) return i;
  }
  return log_prob.n_elem - 1;
}

vec rdirichlet_vec(const vec& alpha) {
  vec out(alpha.n_elem);
  double total = 0.0;
  for (uword i = 0; i < alpha.n_elem; ++i) {
    out(i) = R::rgamma(alpha(i), 1.0);
    total += out(i);
  }
  return out / total;
}

double dtruncnorm_pos(double x, double mean, double sd) {
  const double lower_cdf = R::pnorm((0.0 - mean) / sd, 0.0, 1.0, true, false);
  const double density = R::dnorm(x, mean, sd, false);
  return density / std::max(1.0 - lower_cdf, TINY);
}

double rtruncnorm_reject(double mean, double sd, double lower, double upper) {
  double value = R_NaReal;
  do {
    value = R::rnorm(mean, sd);
  } while (value < lower || value > upper);
  return value;
}

double rbeta_gamma(double alpha, double beta) {
  const double x = R::rgamma(alpha, 1.0);
  const double y = R::rgamma(beta, 1.0);
  return x / (x + y);
}

double mass_texpon(double z) {
  // Mixture probability used by Devroye's PG(1, z) sampler.
  const double fz = square(M_PI) / 8.0 + square(z) / 2.0;
  const double b = std::sqrt(1.0 / TRUNC) * (TRUNC * z - 1.0);
  const double a = -std::sqrt(1.0 / TRUNC) * (TRUNC * z + 1.0);
  const double x0 = std::log(fz) + fz * TRUNC;
  const double xb = x0 - z + R::pnorm(b, 0.0, 1.0, true, true);
  const double xa = x0 + z + R::pnorm(a, 0.0, 1.0, true, true);
  const double qdivp = 4.0 / M_PI * (std::exp(xb) + std::exp(xa));
  return 1.0 / (1.0 + qdivp);
}

double r_tigauss(double z) {
  z = std::abs(z);
  const double mu = 1.0 / z;
  double x = TRUNC + 1.0;

  if (mu > TRUNC) {
    double alpha = 0.0;
    while (R::runif(0.0, 1.0) > alpha) {
      double e1 = R::rexp(1.0);
      double e2 = R::rexp(1.0);
      while (square(e1) > 2.0 * e2 / TRUNC) {
        e1 = R::rexp(1.0);
        e2 = R::rexp(1.0);
      }
      x = TRUNC / square(1.0 + TRUNC * e1);
      alpha = std::exp(-0.5 * square(z) * x);
    }
  } else {
    while (x > TRUNC) {
      const double y = square(R::rnorm(0.0, 1.0));
      x = mu + 0.5 * square(mu) * y - 0.5 * mu * std::sqrt(4.0 * mu * y + square(mu * y));
      if (R::runif(0.0, 1.0) > mu / (mu + x)) x = square(mu) / x;
    }
  }
  return x;
}

double a_coef(double n, double x) {
  if (x > TRUNC) {
    return M_PI * (n + 0.5) * std::exp(-square(n + 0.5) * square(M_PI) * x / 2.0);
  }
  return std::pow(2.0 / M_PI / x, 1.5) * M_PI * (n + 0.5) *
         std::exp(-2.0 * square(n + 0.5) / x);
}

double rpg_devroye_one(double z) {
  // Draw one Polya-Gamma PG(1, z) variate using Devroye's accept-reject method.
  z = std::abs(z) * 0.5;
  const double fz = square(M_PI) / 8.0 + square(z) / 2.0;
  double x = 0.0;

  while (true) {
    if (R::runif(0.0, 1.0) < mass_texpon(z)) {
      x = TRUNC + R::rexp(1.0) / fz;
    } else {
      x = r_tigauss(z);
    }

    double s = a_coef(0.0, x);
    const double y = R::runif(0.0, s);
    int n = 0;
    while (true) {
      ++n;
      if (n % 2 == 1) {
        s -= a_coef(n, x);
        if (y <= s) break;
      } else {
        s += a_coef(n, x);
        if (y > s) break;
      }
    }
    if (y <= s) break;
  }
  return 0.25 * x;
}

double dot_without_col(const mat& x, const rowvec& beta, uword row, uword skip_col) {
  double out = 0.0;
  for (uword p = 0; p < x.n_cols; ++p) {
    if (p != skip_col) out += x(row, p) * beta(p);
  }
  return out;
}

}  // namespace

// [[Rcpp::export]]
double rpg_devroye_1(double z) {
  return rpg_devroye_one(z);
}

// [[Rcpp::export]]
double rpg_devroye_R(int h, double z) {
  double x = 0.0;
  for (int j = 0; j < h; ++j) x += rpg_devroye_one(z);
  return x;
}

// [[Rcpp::export]]
double rtruncnorm1(double mean, double sd, double lower, double upper) {
  return rtruncnorm_reject(mean, sd, lower, upper);
}

// [[Rcpp::export]]
cube f_Thres(const mat& A, const mat& B_i, const mat& Tau, const mat& Y, const vec& M) {
  // Ordinal probit probabilities for each item, latent class, and response level.
  // The class-level linear predictor is computed once per call.
  const int J = B_i.n_rows;
  const int C = A.n_rows;
  const int m_max = static_cast<int>(M.max());
  const mat eta = A * B_i.t();
  cube thres(J, C, m_max, fill::zeros);

  for (int j = 0; j < J; ++j) {
    for (int c = 0; c < C; ++c) {
      for (int m = 0; m < static_cast<int>(M(j)); ++m) {
        thres(j, c, m) =
            R::pnorm(Tau(j, m + 1) - eta(c, j), 0.0, 1.0, true, false) -
            R::pnorm(Tau(j, m) - eta(c, j), 0.0, 1.0, true, false);
      }
    }
  }
  return thres;
}

// [[Rcpp::export]]
List f_alp_Ystar_pi(vec n_cate, vec i_cate, mat A_cate, const mat& A, const mat& B_i,
                    const mat& Tau, const mat& Y, const vec& M, const mat& Theta,
                    const mat& G_cate, const mat& Gcate_Covariate) {
  // Gibbs update for alpha classes and latent Y*. The optimized version keeps
  // the class probabilities in log scale and samples with pure C++ code.
  const int n = Y.n_rows;
  const int J = B_i.n_rows;
  const int K = B_i.n_cols - 1;
  const int C = A.n_rows;

  const cube thres = f_Thres(A, B_i, Tau, Y, M);
  const mat class_eta = Gcate_Covariate * Theta;
  mat y_star(n, J, fill::zeros);
  vec log_prob(C);

  for (int i = 0; i < n; ++i) {
    n_cate(static_cast<uword>(i_cate(i))) -= 1.0;

    for (int c = 0; c < C; ++c) {
      double lp = 0.0;
      for (int j = 0; j < J; ++j) {
        lp += safe_log(thres(j, c, static_cast<uword>(Y(i, j))));
      }
      for (int k = 0; k < K; ++k) {
        lp += log_bernoulli_eta(class_eta(i, k), A(c, k + 1));
      }
      log_prob(c) = lp;
    }

    const uword new_class = sample_log_prob(log_prob);
    i_cate(i) = static_cast<double>(new_class);
    n_cate(new_class) += 1.0;
    A_cate.row(i) = A.row(new_class);

    for (int j = 0; j < J; ++j) {
      const double mean = dot(A.row(new_class), B_i.row(j));
      y_star(i, j) = rtruncnorm_reject(mean, 1.0, Tau(j, static_cast<uword>(Y(i, j))),
                                       Tau(j, static_cast<uword>(Y(i, j)) + 1));
    }
  }

  const vec pi_gibbs = rdirichlet_vec(n_cate + 1.0);
  return List::create(_["n_cate"] = n_cate, _["i_cate"] = i_cate, _["A_cate"] = A_cate,
                      _["Y_star_gibbs"] = y_star, _["pi_gibbs"] = pi_gibbs);
}

// [[Rcpp::export]]
List f_gam_Vstar_pi(vec n_cate, vec i_cate, mat G_cate, const mat& G, const mat& L_i,
                    const mat& Tau, const mat& V, const vec& M, const mat& Theta,
                    const mat& A_cate, const mat& COV, const mat& G_cateCov) {
  // Gibbs update for gamma classes and latent V*. For each subject, only the
  // needed class/covariate linear predictors are formed instead of rebuilding
  // a large temporary matrix.
  const int n = V.n_rows;
  const int J = L_i.n_rows;
  const int K_g = L_i.n_cols - 1;
  const int K_a = A_cate.n_cols - 1;
  const int C = G.n_rows;
  const int p_cov = COV.n_cols;
  const int nc0 = 1;

  const cube thres = f_Thres(G, L_i, Tau, V, M);
  mat v_star(n, J, fill::zeros);
  vec log_prob(C);

  for (int i = 0; i < n; ++i) {
    n_cate(static_cast<uword>(i_cate(i))) -= 1.0;

    for (int c = 0; c < C; ++c) {
      double lp = std::log(nc0 + n_cate(c));
      for (int j = 0; j < J; ++j) {
        lp += safe_log(thres(j, c, static_cast<uword>(V(i, j))));
      }

      for (int k = 0; k < K_a; ++k) {
        double eta = 0.0;
        for (int p = 0; p < K_g + 1; ++p) eta += G(c, p) * Theta(p, k);
        for (int p = 0; p < p_cov; ++p) eta += COV(i, p) * Theta(K_g + 1 + p, k);
        lp += log_bernoulli_eta(eta, A_cate(i, k + 1));
      }
      log_prob(c) = lp;
    }

    const uword new_class = sample_log_prob(log_prob);
    i_cate(i) = static_cast<double>(new_class);
    n_cate(new_class) += 1.0;
    G_cate.row(i) = G.row(new_class);

    for (int j = 0; j < J; ++j) {
      const double mean = dot(G.row(new_class), L_i.row(j));
      v_star(i, j) = rtruncnorm_reject(mean, 1.0, Tau(j, static_cast<uword>(V(i, j))),
                                       Tau(j, static_cast<uword>(V(i, j)) + 1));
    }
  }

  const vec pi_gibbs = rdirichlet_vec(n_cate + 1.0);
  return List::create(_["n_cate"] = n_cate, _["i_cate"] = i_cate, _["G_cate"] = G_cate,
                      _["V_star_gibbs"] = v_star, _["pi_gibbs"] = pi_gibbs);
}

// [[Rcpp::export]]
List f_Sitacoef_w(const mat& A_cate, mat W, const mat& G_cate, mat Sita,
                  const mat& Means_prior, const cube& Cov_prior, const mat& Gcate_Covariate) {
  // Polya-Gamma regression update for Sita. The key speed-up is replacing
  // X' diag(W) X with X' (X * W), avoiding an n by n diagonal matrix.
  const int n = A_cate.n_rows;
  const int K_1 = Sita.n_cols;
  const int K_2 = Sita.n_rows;
  const mat A_func = A_cate.cols(1, A_cate.n_cols - 1) - 0.5;
  const mat gamma_sita = Gcate_Covariate * Sita;

  for (int k = 0; k < K_1; ++k) {
    for (int i = 0; i < n; ++i) W(k, i) = rpg_devroye_one(gamma_sita(i, k));

    const vec wk = W.row(k).t();
    const mat weighted_x = Gcate_Covariate.each_col() % wk;
    const mat prior_prec = inv_sympd(Cov_prior.slice(k));
    const mat precision = Gcate_Covariate.t() * weighted_x + prior_prec;
    const vec rhs = Gcate_Covariate.t() * A_func.col(k) + prior_prec * Means_prior.col(k);
    const mat cov_post = inv_sympd(precision);
    const vec mean_post = cov_post * rhs;
    Sita.col(k) = mean_post + chol(cov_post, "lower") * randn<vec>(K_2);
  }

  return List::create(_["Sita"] = Sita, _["W"] = W);
}

// [[Rcpp::export]]
List f_Q_Beta_omega(const mat& A_cate, mat B_i, const mat& Y_star_gibbs, int c1, int c0,
                    double omega, mat V_Q, mat Q_MH, mat Q_qta) {
  // Metropolis update for Q followed by conjugate/truncated-normal updates
  // for the item coefficients. Temporary submatrices are avoided in the inner loop.
  const int J = B_i.n_rows;
  const int K = B_i.n_cols - 1;
  const int p = K + 1;
  const double inf = std::numeric_limits<double>::infinity();
  const mat D = A_cate.t() * A_cate;

  for (int j = 0; j < J; ++j) {
    for (int k = 0; k < K; ++k) {
      const double tran1 = dtruncnorm_pos(B_i(j, k + 1), 0.0, std::sqrt(1.0 / c1));
      const double tran2 = dtruncnorm_pos(B_i(j, k + 1), 0.0, std::sqrt(1.0 / c0));
      const double trans = tran2 / tran1 * (1.0 - omega) / omega;
      const double accept = std::min(1.0, std::pow(trans, 2.0 * Q_MH(j, k) - 1.0));

      if (R::runif(0.0, 1.0) < accept) {
        Q_MH(j, k) = 1.0 - Q_MH(j, k);
        Q_qta(j, k + 1) = Q_MH(j, k);
      }
    }

    V_Q.row(j) = Q_qta.row(j) / c1 + (1.0 - Q_qta.row(j)) / c0;

    for (int l = 0; l < p; ++l) {
      double rhs = 0.0;
      const rowvec beta_j = B_i.row(j);
      for (uword i = 0; i < A_cate.n_rows; ++i) {
        rhs += A_cate(i, l) * (Y_star_gibbs(i, j) - dot_without_col(A_cate, beta_j, i, l));
      }
      const double sigma = 1.0 / (D(l, l) + 1.0 / V_Q(j, l));
      const double mean = sigma * rhs;
      B_i(j, l) = (l == 0) ? R::rnorm(mean, std::sqrt(sigma))
                            : rtruncnorm_reject(mean, std::sqrt(sigma), 0.0, inf);
    }
  }

  omega = rbeta_gamma(accu(Q_MH) + 1.0, J * K - accu(Q_MH) + 1.0);
  return List::create(_["Q_MH"] = Q_MH, _["Q_qta"] = Q_qta, _["B_i"] = B_i,
                      _["V_Q"] = V_Q, _["omega"] = omega);
}

// [[Rcpp::export]]
List fixQ_Beta_omega(const mat& A_cate, mat B_i, const mat& Y_star_gibbs, int c1, int c0,
                     double omega, mat V_Q, mat Q_MH, mat Q_qta) {
  const int J = B_i.n_rows;
  const int K = B_i.n_cols - 1;
  const int p = K + 1;
  const double inf = std::numeric_limits<double>::infinity();
  const mat D = A_cate.t() * A_cate;

  for (int j = 0; j < J; ++j) {
    for (int l = 0; l < p; ++l) {
      double rhs = 0.0;
      const rowvec beta_j = B_i.row(j);
      for (uword i = 0; i < A_cate.n_rows; ++i) {
        rhs += A_cate(i, l) * (Y_star_gibbs(i, j) - dot_without_col(A_cate, beta_j, i, l));
      }
      const double sigma = 1.0 / (D(l, l) + 1.0 / V_Q(j, l));
      const double mean = sigma * rhs;
      B_i(j, l) = (l == 0) ? R::rnorm(mean, std::sqrt(sigma))
                            : rtruncnorm_reject(mean, std::sqrt(sigma), 0.0, inf);
    }
  }

  omega = rbeta_gamma(accu(Q_MH) + 1.0, J * K - accu(Q_MH) + 1.0);
  return List::create(_["Q_MH"] = Q_MH, _["Q_qta"] = Q_qta, _["B_i"] = B_i,
                      _["V_Q"] = V_Q, _["omega"] = omega);
}

// [[Rcpp::export]]
List WAIC_y_v(const mat& Y, const mat& V, const vec& i_cate_a, const vec& i_cate_g,
              const mat& A, const mat& G, const mat& B_i, const mat& L_i,
              const vec& M_y, const vec& M_v, const mat& tau_y, const mat& tau_v,
              const mat& Theta, const mat& Gcate_Covariate, const mat& A_cate) {
  // Per-person likelihood contribution used downstream for PBIC/WAIC-style summaries.
  // Products are accumulated in log scale first to reduce underflow.
  const int n = V.n_rows;
  const int J_y = B_i.n_rows;
  const int J_v = L_i.n_rows;
  const int K_y = A.n_cols - 1;

  const cube thres_y = f_Thres(A, B_i, tau_y, Y, M_y);
  const cube thres_v = f_Thres(G, L_i, tau_v, V, M_v);
  const mat eta = Gcate_Covariate * Theta;
  vec out(n, fill::ones);

  for (int i = 0; i < n; ++i) {
    double log_lik = 0.0;
    const uword ca = static_cast<uword>(i_cate_a(i));
    const uword cg = static_cast<uword>(i_cate_g(i));

    for (int j = 0; j < J_y; ++j) log_lik += safe_log(thres_y(j, ca, static_cast<uword>(Y(i, j))));
    for (int j = 0; j < J_v; ++j) log_lik += safe_log(thres_v(j, cg, static_cast<uword>(V(i, j))));
    for (int k = 0; k < K_y; ++k) log_lik += log_bernoulli_eta(eta(i, k), A_cate(i, k + 1));

    out(i) = std::exp(log_lik);
  }

  return List::create(_["P_YVber"] = out);
}
