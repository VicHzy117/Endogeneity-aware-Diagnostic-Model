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

inline double exact_log_bf_impl(double s_xx, double s_xr, double sigma_beta2) {
  const double precision = s_xx + 1.0 / sigma_beta2;
  const double z = s_xr / std::sqrt(precision);
  return std::log(2.0) - 0.5 * std::log(sigma_beta2 * precision) +
         0.5 * s_xr * s_xr / precision +
         R::pnorm(z, 0.0, 1.0, true, true);
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

double rtruncnorm_reject(double mean, double sd, double lower, double upper) {
  // Stable sampler for N(mean, sd^2) truncated to [lower, upper].  Naive
  // rejection from the untruncated normal can effectively hang when the
  // interval lies deep in a tail, which can occur transiently in MCMC.
  if (!(sd > 0.0) || !(lower < upper)) return R_NaReal;

  double a = (lower - mean) / sd;
  double b = (upper - mean) / sd;
  double z = R_NaReal;

  if (!std::isfinite(a) && !std::isfinite(b)) {
    z = R::rnorm(0.0, 1.0);
  } else if (a > 0.0) {
    // Robert's exponentially tilted rejection sampler for a positive tail.
    const double rate = 0.5 * (a + std::sqrt(a * a + 4.0));
    do {
      z = a + R::rexp(1.0 / rate);
      if (z > b) continue;
    } while (z > b || R::runif(0.0, 1.0) > std::exp(-0.5 * square(z - rate)));
  } else if (b < 0.0) {
    // Reflect an upper-tail problem into a positive lower-tail problem.
    const double reflected_lower = -b;
    const double reflected_upper = -a;
    const double rate = 0.5 * (reflected_lower +
                               std::sqrt(reflected_lower * reflected_lower + 4.0));
    double reflected = R_NaReal;
    do {
      reflected = reflected_lower + R::rexp(1.0 / rate);
      if (reflected > reflected_upper) continue;
    } while (reflected > reflected_upper ||
             R::runif(0.0, 1.0) > std::exp(-0.5 * square(reflected - rate)));
    z = -reflected;
  } else {
    // The interval contains the normal mode, so direct rejection has at least
    // moderate acceptance for the ordinal intervals used by this model.
    do {
      z = R::rnorm(0.0, 1.0);
    } while (z < a || z > b);
  }
  return mean + sd * z;
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
double exact_q_log_bf(double s_xx, double s_xr, double sigma_beta2) {
  return exact_log_bf_impl(s_xx, s_xr, sigma_beta2);
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
                    const mat& A_cate, const mat& COV, const vec& pi_g) {
  // Gibbs update for gamma classes and latent V*. For each subject, only the
  // needed class/covariate linear predictors are formed instead of rebuilding
  // a large temporary matrix.
  const int n = V.n_rows;
  const int J = L_i.n_rows;
  const int K_g = L_i.n_cols - 1;
  const int K_a = A_cate.n_cols - 1;
  const int C = G.n_rows;
  const int p_cov = COV.n_cols;
  const cube thres = f_Thres(G, L_i, Tau, V, M);
  mat v_star(n, J, fill::zeros);
  vec log_prob(C);

  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < C; ++c) {
      // Exact full conditional under p(alpha2 | pi2).  The previous code used
      // leave-one-out class counts here, which implicitly collapsed pi2 even
      // though a separate pi2 draw was stored.  The manuscript sampler instead
      // conditions on the current Dirichlet draw.
      double lp = safe_log(pi_g(c));
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
    G_cate.row(i) = G.row(new_class);

    for (int j = 0; j < J; ++j) {
      const double mean = dot(G.row(new_class), L_i.row(j));
      v_star(i, j) = rtruncnorm_reject(mean, 1.0, Tau(j, static_cast<uword>(V(i, j))),
                                       Tau(j, static_cast<uword>(V(i, j)) + 1));
    }
  }

  n_cate.zeros();
  for (int i = 0; i < n; ++i) n_cate(static_cast<uword>(i_cate(i))) += 1.0;
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
List f_Q_Beta_omega(const mat& A_cate, mat B_i, const mat& Y_star_gibbs,
                    double sigma_beta2, double sigma_intercept2,
                    double omega, mat Q_MH) {
  // Exact spike-and-positive-slab update from manuscript Section 3.2, Step 4.
  // B_i stores the effective coefficients delta: when q_jk=0 the corresponding
  // coefficient is exactly zero and therefore absent from the likelihood.
  // q_jk is sampled after analytically integrating the positive slab magnitude.
  const int J = B_i.n_rows;
  const int K = B_i.n_cols - 1;
  const double inf = std::numeric_limits<double>::infinity();

  for (int j = 0; j < J; ++j) {
    for (int k = 0; k < K; ++k) {
      double s_xx = 0.0;
      double s_xr = 0.0;
      for (uword i = 0; i < A_cate.n_rows; ++i) {
        double fitted_without_k = B_i(j, 0);
        for (int h = 0; h < K; ++h) {
          if (h != k) fitted_without_k += A_cate(i, h + 1) * B_i(j, h + 1);
        }
        const double x = A_cate(i, k + 1);
        const double residual = Y_star_gibbs(i, j) - fitted_without_k;
        s_xx += x * x;
        s_xr += x * residual;
      }

      const double precision = s_xx + 1.0 / sigma_beta2;
      const double post_var = 1.0 / precision;
      const double post_mean = post_var * s_xr;

      // log{m_jk(1)/m_jk(0)} for a half-normal N_+(0,sigma_beta2) slab.
      const double log_bf = exact_log_bf_impl(s_xx, s_xr, sigma_beta2);
      const double log_odds = std::log(omega) - std::log1p(-omega) + log_bf;
      const double inclusion_prob = (log_odds >= 0.0)
          ? 1.0 / (1.0 + std::exp(-log_odds))
          : std::exp(log_odds) / (1.0 + std::exp(log_odds));

      if (R::runif(0.0, 1.0) < inclusion_prob) {
        Q_MH(j, k) = 1.0;
        B_i(j, k + 1) = rtruncnorm_reject(
            post_mean, std::sqrt(post_var), 0.0, inf);
      } else {
        Q_MH(j, k) = 0.0;
        B_i(j, k + 1) = 0.0;
      }
    }

    // Intercept update after the coordinate-wise partially collapsed Q sweep.
    double residual_sum = 0.0;
    for (uword i = 0; i < A_cate.n_rows; ++i) {
      double active_part = 0.0;
      for (int k = 0; k < K; ++k) active_part += A_cate(i, k + 1) * B_i(j, k + 1);
      residual_sum += Y_star_gibbs(i, j) - active_part;
    }
    const double intercept_var = 1.0 / (A_cate.n_rows + 1.0 / sigma_intercept2);
    const double intercept_mean = intercept_var * residual_sum;
    B_i(j, 0) = R::rnorm(intercept_mean, std::sqrt(intercept_var));
  }

  omega = rbeta_gamma(accu(Q_MH) + 1.0, J * K - accu(Q_MH) + 1.0);
  return List::create(_["Q_MH"] = Q_MH, _["B_i"] = B_i, _["omega"] = omega);
}

// [[Rcpp::export]]
List complete_loglik(const mat& Y, const mat& V, const vec& i_cate_a, const vec& i_cate_g,
                     const mat& A, const mat& G, const mat& B_i, const mat& L_i,
                     const vec& M_y, const vec& M_v, const mat& tau_y, const mat& tau_v,
                     const mat& Theta, const mat& Gcate_Covariate, const mat& A_cate,
                     const vec& pi_g) {
  // Manuscript equation (6): measurement likelihood, structural Bernoulli
  // likelihood, and the previously omitted pi2 profile-probability term.
  const int n = V.n_rows;
  const int J_y = B_i.n_rows;
  const int J_v = L_i.n_rows;
  const int K_y = A.n_cols - 1;

  const cube thres_y = f_Thres(A, B_i, tau_y, Y, M_y);
  const cube thres_v = f_Thres(G, L_i, tau_v, V, M_v);
  const mat eta = Gcate_Covariate * Theta;
  vec out(n, fill::zeros);

  for (int i = 0; i < n; ++i) {
    double log_lik = safe_log(pi_g(static_cast<uword>(i_cate_g(i))));
    const uword ca = static_cast<uword>(i_cate_a(i));
    const uword cg = static_cast<uword>(i_cate_g(i));

    for (int j = 0; j < J_y; ++j) log_lik += safe_log(thres_y(j, ca, static_cast<uword>(Y(i, j))));
    for (int j = 0; j < J_v; ++j) log_lik += safe_log(thres_v(j, cg, static_cast<uword>(V(i, j))));
    for (int k = 0; k < K_y; ++k) log_lik += log_bernoulli_eta(eta(i, k), A_cate(i, k + 1));

    out(i) = log_lik;
  }

  return List::create(_["log_lik_complete"] = out);
}
