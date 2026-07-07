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

constexpr double TINY = 1e-300;

inline double safe_log(double x) {
  return std::log(std::max(x, TINY));
}

double rtruncnorm_reject(double mean, double sd, double lower, double upper) {
  double value = R_NaReal;
  do {
    value = R::rnorm(mean, sd);
  } while (value < lower || value > upper);
  return value;
}

double dtruncnorm_pos(double x, double mean, double sd) {
  const double lower_cdf = R::pnorm((0.0 - mean) / sd, 0.0, 1.0, true, false);
  return R::dnorm(x, mean, sd, false) / std::max(1.0 - lower_cdf, TINY);
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

double rbeta_gamma(double alpha, double beta) {
  const double x = R::rgamma(alpha, 1.0);
  const double y = R::rgamma(beta, 1.0);
  return x / (x + y);
}

uword sample_log_prob(const vec& log_prob) {
  const double max_log = log_prob.max();
  std::vector<double> weights(log_prob.n_elem);
  double total = 0.0;

  for (uword c = 0; c < log_prob.n_elem; ++c) {
    weights[c] = std::exp(log_prob(c) - max_log);
    total += weights[c];
  }

  const double draw = R::runif(0.0, total);
  double cumulative = 0.0;
  for (uword c = 0; c < log_prob.n_elem; ++c) {
    cumulative += weights[c];
    if (draw <= cumulative) return c;
  }
  return log_prob.n_elem - 1;
}

double dot_without_col(const mat& x, const rowvec& beta, uword row, uword skip_col) {
  double out = 0.0;
  for (uword p = 0; p < x.n_cols; ++p) {
    if (p != skip_col) out += x(row, p) * beta(p);
  }
  return out;
}

cube ordinal_probabilities(const mat& design, const mat& beta, const mat& tau,
                           const vec& n_levels) {
  const int J = beta.n_rows;
  const int C = design.n_rows;
  const int m_max = static_cast<int>(n_levels.max());
  const mat eta = design * beta.t();
  cube prob(J, C, m_max, fill::zeros);

  for (int j = 0; j < J; ++j) {
    for (int c = 0; c < C; ++c) {
      for (int m = 0; m < static_cast<int>(n_levels(j)); ++m) {
        prob(j, c, m) =
            R::pnorm(tau(j, m + 1) - eta(c, j), 0.0, 1.0, true, false) -
            R::pnorm(tau(j, m) - eta(c, j), 0.0, 1.0, true, false);
      }
    }
  }

  return prob;
}

}  // namespace

// [[Rcpp::export]]
double rtruncnorm1(double mean, double sd, double lower, double upper) {
  return rtruncnorm_reject(mean, sd, lower, upper);
}

// [[Rcpp::export]]
cube f_Thres(const mat& A, const mat& B_i, const mat& Tau, const mat& Y, const vec& M) {
  return ordinal_probabilities(A, B_i, Tau, M);
}

// [[Rcpp::export]]
List f_alp_Ystar_pi(mat N_gibbs, vec alpha_gibbs, mat a_gibbs, const mat& A,
                    const mat& B_gibbs, const mat& Tau, const mat& Y, int Mj) {
  const int n = Y.n_rows;
  const int J = B_gibbs.n_rows;
  const int C = A.n_rows;
  const vec M = arma::ones<vec>(J) * Mj;
  const rowvec tau_row = Tau.t();
  const mat tau_by_item = arma::repmat(tau_row, J, 1);
  const cube theta = ordinal_probabilities(A, B_gibbs, tau_by_item, M);

  mat Y_star_gibbs(n, J, fill::zeros);
  vec log_prob(C);

  for (int i = 0; i < n; ++i) {
    const uword old_class = static_cast<uword>(alpha_gibbs(i));
    N_gibbs(0, old_class) -= 1.0;

    for (int c = 0; c < C; ++c) {
      double lp = safe_log(N_gibbs(0, c) + 1.0);
      for (int j = 0; j < J; ++j) {
        lp += safe_log(theta(j, c, static_cast<uword>(Y(i, j))));
      }
      log_prob(c) = lp;
    }

    const uword new_class = sample_log_prob(log_prob);
    alpha_gibbs(i) = static_cast<double>(new_class);
    N_gibbs(0, new_class) += 1.0;
    a_gibbs.row(i) = A.row(new_class);

    for (int j = 0; j < J; ++j) {
      const double mean = arma::dot(A.row(new_class), B_gibbs.row(j));
      Y_star_gibbs(i, j) =
          rtruncnorm_reject(mean, 1.0, Tau(static_cast<uword>(Y(i, j)), 0),
                            Tau(static_cast<uword>(Y(i, j)) + 1, 0));
    }
  }

  const vec pi_gibbs = rdirichlet_vec(N_gibbs.t() + 1.0);
  return List::create(_["N_gibbs"] = N_gibbs, _["alpha_gibbs"] = alpha_gibbs,
                      _["a_gibbs"] = a_gibbs, _["Y_star_gibbs"] = Y_star_gibbs,
                      _["pi_gibbs"] = pi_gibbs);
}

// [[Rcpp::export]]
List f_Q_Beta_omega(const mat& a_gibbs, mat B_gibbs, const mat& Y_star_gibbs,
                    int c1, int c0, double omega, mat V, mat Q_MH, mat Q_qta) {
  const int J = B_gibbs.n_rows;
  const int P = B_gibbs.n_cols;
  const int n_effects = P - 1;
  const mat D = a_gibbs.t() * a_gibbs;

  for (int j = 0; j < J; ++j) {
    for (int p = 0; p < n_effects; ++p) {
      const int beta_col = p + 1;
      const double dense = dtruncnorm_pos(B_gibbs(j, beta_col), 0.0, std::sqrt(1.0 / c1));
      const double spike = dtruncnorm_pos(B_gibbs(j, beta_col), 0.0, std::sqrt(1.0 / c0));
      const double ratio = spike / std::max(dense, TINY) * (1.0 - omega) / omega;
      const double accept = std::min(1.0, std::pow(ratio, 2.0 * Q_MH(j, p) - 1.0));

      if (R::runif(0.0, 1.0) < accept) {
        Q_MH(j, p) = 1.0 - Q_MH(j, p);
        Q_qta(j, beta_col) = Q_MH(j, p);
      }
    }

    V.row(j) = Q_qta.row(j) / c1 + (1.0 - Q_qta.row(j)) / c0;

    for (int p = 0; p < P; ++p) {
      const double sigma = 1.0 / (D(p, p) + 1.0 / V(j, p));
      double rhs = 0.0;
      for (uword i = 0; i < a_gibbs.n_rows; ++i) {
        rhs += a_gibbs(i, p) * (Y_star_gibbs(i, j) - dot_without_col(a_gibbs, B_gibbs.row(j), i, p));
      }

      const double mean = sigma * rhs;
      const double sd = std::sqrt(sigma);
      B_gibbs(j, p) = (p == 0) ? R::rnorm(mean, sd) : rtruncnorm_reject(mean, sd, 0.0, arma::datum::inf);
    }
  }

  omega = rbeta_gamma(arma::accu(Q_MH) + 1.0, J * n_effects - arma::accu(Q_MH) + 1.0);
  return List::create(_["Q_MH"] = Q_MH, _["Q_qta"] = Q_qta, _["B_gibbs"] = B_gibbs,
                      _["V"] = V, _["omega"] = omega);
}

// [[Rcpp::export]]
List BIC_y(const mat& Y, const vec& i_cate_a, const mat& A, const mat& B_i,
           const vec& M_y, const mat& tau_y) {
  const int n = Y.n_rows;
  const int J = B_i.n_rows;
  const cube theta = ordinal_probabilities(A, B_i, tau_y, M_y);
  mat P_Y(n, J, fill::ones);

  for (int i = 0; i < n; ++i) {
    const uword c = static_cast<uword>(i_cate_a(i));
    for (int j = 0; j < J; ++j) {
      P_Y(i, j) = std::max(theta(j, c, static_cast<uword>(Y(i, j))), TINY);
    }
  }

  return List::create(_["P_Y"] = prod(P_Y, 1));
}

// [[Rcpp::export]]
List f_mcmc(mat N_gibbs, vec alpha_gibbs, const mat& A, const mat& Tau, const mat& Y,
            int Mj, mat a_gibbs, mat B_gibbs, int c1, int c0, double omega, mat V,
            mat Q_MH, mat Q_qta, int iteration) {
  const int n = Y.n_rows;
  const int J = B_gibbs.n_rows;
  const int C = A.n_rows;
  const int P = B_gibbs.n_cols;
  const int n_effects = P - 1;

  vec pi_gibbs(C);
  mat Y_star_gibbs(n, J);
  mat pi_gibbs_trace(C, iteration);
  mat alpha_gibbs_trace(n, iteration);
  cube Q_MH_trace(J, n_effects, iteration);
  cube B_gibbs_trace(J, P, iteration);
  mat P_Y(n, iteration);

  const vec M_j = arma::ones<vec>(J) * Mj;
  const rowvec tau_row = Tau.t();
  const mat tau_mat = arma::repmat(tau_row, J, 1);

  for (int r = 0; r < iteration; ++r) {
    List alpha_step = f_alp_Ystar_pi(N_gibbs, alpha_gibbs, a_gibbs, A, B_gibbs, Tau, Y, Mj);
    N_gibbs = as<mat>(alpha_step["N_gibbs"]);
    alpha_gibbs = as<vec>(alpha_step["alpha_gibbs"]);
    a_gibbs = as<mat>(alpha_step["a_gibbs"]);
    Y_star_gibbs = as<mat>(alpha_step["Y_star_gibbs"]);
    pi_gibbs = as<vec>(alpha_step["pi_gibbs"]);

    List beta_step = f_Q_Beta_omega(a_gibbs, B_gibbs, Y_star_gibbs, c1, c0,
                                    omega, V, Q_MH, Q_qta);
    Q_MH = as<mat>(beta_step["Q_MH"]);
    Q_qta = as<mat>(beta_step["Q_qta"]);
    B_gibbs = as<mat>(beta_step["B_gibbs"]);
    V = as<mat>(beta_step["V"]);
    omega = as<double>(beta_step["omega"]);

    pi_gibbs_trace.col(r) = pi_gibbs;
    alpha_gibbs_trace.col(r) = alpha_gibbs;
    Q_MH_trace.slice(r) = Q_MH;
    B_gibbs_trace.slice(r) = B_gibbs;

    List likelihood_step = BIC_y(Y, alpha_gibbs, A, B_gibbs, M_j, tau_mat);
    P_Y.col(r) = as<vec>(likelihood_step["P_Y"]);
  }

  return List::create(_["pi_gibbs_trace"] = pi_gibbs_trace,
                      _["Q_MH_trace"] = Q_MH_trace,
                      _["B_gibbs_trace"] = B_gibbs_trace,
                      _["alpha_gibbs_trace"] = alpha_gibbs_trace,
                      _["P_Y"] = P_Y);
}
