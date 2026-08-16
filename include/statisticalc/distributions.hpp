// statisticalc - header-only running statistics for C++20
//
// Minimal set of continuous distributions needed for inferential statistics:
// standard normal, Student's t, chi-squared and Fisher-Snedecor F.
//
// For each distribution three functions are provided:
//   *_cdf(x, ...)       lower tail, P(X <= x)
//   *_sf(x, ...)        upper tail (survival function), P(X > x)
//   *_quantile(p, ...)  inverse cdf
// The survival functions are computed directly (rather than as 1 - cdf) so that
// p-values stay accurate deep into the tails.
#pragma once

#include <cmath>
#include <concepts>
#include <limits>

#include "detail/special_functions.hpp"

namespace statisticalc::dist {

// ---------------------------------------------------------------------------
// Standard normal
// ---------------------------------------------------------------------------

template <std::floating_point R>
[[nodiscard]] R normal_pdf(R z) {
  constexpr R inv_sqrt_2pi = R(0.3989422804014326779399460599343818684759);
  return inv_sqrt_2pi * std::exp(R(-0.5) * z * z);
}

template <std::floating_point R>
[[nodiscard]] R normal_cdf(R z) {
  constexpr R inv_sqrt2 = R(0.7071067811865475244008443621048490392848);
  return R(0.5) * std::erfc(-z * inv_sqrt2);
}

template <std::floating_point R>
[[nodiscard]] R normal_sf(R z) {
  constexpr R inv_sqrt2 = R(0.7071067811865475244008443621048490392848);
  return R(0.5) * std::erfc(z * inv_sqrt2);
}

/// Inverse standard normal cdf (Acklam's rational approximation followed by one
/// Halley refinement step, which brings the result to full double precision).
template <std::floating_point R>
[[nodiscard]] R normal_quantile(R p) {
  if (std::isnan(p) || p < R(0) || p > R(1)) return detail::quiet_nan<R>();
  if (p == R(0)) return -std::numeric_limits<R>::infinity();
  if (p == R(1)) return std::numeric_limits<R>::infinity();

  constexpr double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                           -2.759285104469687e+02, 1.383577518672690e+02,
                           -3.066479806614716e+01, 2.506628277459239e+00};
  constexpr double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                           -1.556989798598866e+02, 6.680131188771972e+01,
                           -1.328068155288572e+01};
  constexpr double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                           -2.400758277161838e+00, -2.549732539343734e+00,
                           4.374664141464968e+00,  2.938163982698783e+00};
  constexpr double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                           2.445134137142996e+00, 3.754408661907416e+00};
  constexpr double p_low = 0.02425;

  const double pp = static_cast<double>(p);
  double x = 0.0;
  if (pp < p_low) {
    const double q = std::sqrt(-2.0 * std::log(pp));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  } else if (pp <= 1.0 - p_low) {
    const double q = pp - 0.5;
    const double r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
        q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
  } else {
    const double q = std::sqrt(-2.0 * std::log1p(-pp));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }

  // One Halley step on e(x) = cdf(x) - p.
  auto z = static_cast<R>(x);
  const R e = normal_cdf(z) - p;
  const R u = e / normal_pdf(z);
  z -= u / (R(1) + R(0.5) * z * u);
  return z;
}

// ---------------------------------------------------------------------------
// Student's t
// ---------------------------------------------------------------------------

template <std::floating_point R>
[[nodiscard]] R student_t_pdf(R t, R nu) {
  if (!(nu > R(0))) return detail::quiet_nan<R>();
  const R half = R(0.5);
  const R lg = std::lgamma(half * (nu + R(1))) - std::lgamma(half * nu);
  constexpr R pi = R(3.14159265358979323846264338327950288);
  return std::exp(lg) / std::sqrt(nu * pi) *
         std::pow(R(1) + t * t / nu, -half * (nu + R(1)));
}

template <std::floating_point R>
[[nodiscard]] R student_t_cdf(R t, R nu) {
  if (!(nu > R(0)) || std::isnan(t)) return detail::quiet_nan<R>();
  const R x = nu / (nu + t * t);
  const R tail = R(0.5) * detail::beta_i(R(0.5) * nu, R(0.5), x);
  return (t > R(0)) ? R(1) - tail : tail;
}

template <std::floating_point R>
[[nodiscard]] R student_t_sf(R t, R nu) {
  return student_t_cdf(-t, nu);
}

template <std::floating_point R>
[[nodiscard]] R student_t_quantile(R p, R nu) {
  if (!(nu > R(0)) || std::isnan(p) || p < R(0) || p > R(1))
    return detail::quiet_nan<R>();
  if (p == R(0)) return -std::numeric_limits<R>::infinity();
  if (p == R(1)) return std::numeric_limits<R>::infinity();
  // The normal quantile is an excellent starting bracket for any nu.
  const R guess = (std::max)(std::abs(normal_quantile(p)), R(1));
  return detail::invert_real<R>([nu](R x) { return student_t_cdf(x, nu); }, p,
                                guess * R(2));
}

// ---------------------------------------------------------------------------
// Chi-squared
// ---------------------------------------------------------------------------

template <std::floating_point R>
[[nodiscard]] R chi_squared_pdf(R x, R k) {
  if (!(k > R(0)) || x < R(0)) return detail::quiet_nan<R>();
  if (x == R(0))
    return (k < R(2)) ? std::numeric_limits<R>::infinity()
                      : (k == R(2) ? R(0.5) : R(0));
  const R half_k = R(0.5) * k;
  return std::exp((half_k - R(1)) * std::log(x) - R(0.5) * x -
                  std::lgamma(half_k) - half_k * std::log(R(2)));
}

template <std::floating_point R>
[[nodiscard]] R chi_squared_cdf(R x, R k) {
  return detail::gamma_p(R(0.5) * k, R(0.5) * x);
}

template <std::floating_point R>
[[nodiscard]] R chi_squared_sf(R x, R k) {
  return detail::gamma_q(R(0.5) * k, R(0.5) * x);
}

template <std::floating_point R>
[[nodiscard]] R chi_squared_quantile(R p, R k) {
  if (!(k > R(0)) || std::isnan(p) || p < R(0) || p > R(1))
    return detail::quiet_nan<R>();
  if (p == R(0)) return R(0);
  if (p == R(1)) return std::numeric_limits<R>::infinity();
  // Wilson-Hilferty approximation as an initial magnitude.
  const R z = normal_quantile(p);
  const R w = R(1) - R(2) / (R(9) * k) + z * std::sqrt(R(2) / (R(9) * k));
  const R guess = (w > R(0)) ? k * w * w * w : k;
  return detail::invert_positive<R>([k](R x) { return chi_squared_cdf(x, k); },
                                    p, (std::max)(guess, R(1e-3)));
}

// ---------------------------------------------------------------------------
// Fisher-Snedecor F
// ---------------------------------------------------------------------------

template <std::floating_point R>
[[nodiscard]] R fisher_f_cdf(R x, R d1, R d2) {
  if (!(d1 > R(0)) || !(d2 > R(0)) || std::isnan(x))
    return detail::quiet_nan<R>();
  if (x <= R(0)) return R(0);
  const R y = d1 * x / (d1 * x + d2);
  return detail::beta_i(R(0.5) * d1, R(0.5) * d2, y);
}

template <std::floating_point R>
[[nodiscard]] R fisher_f_sf(R x, R d1, R d2) {
  if (!(d1 > R(0)) || !(d2 > R(0)) || std::isnan(x))
    return detail::quiet_nan<R>();
  if (x <= R(0)) return R(1);
  const R y = d2 / (d1 * x + d2);
  return detail::beta_i(R(0.5) * d2, R(0.5) * d1, y);
}

template <std::floating_point R>
[[nodiscard]] R fisher_f_quantile(R p, R d1, R d2) {
  if (!(d1 > R(0)) || !(d2 > R(0)) || std::isnan(p) || p < R(0) || p > R(1))
    return detail::quiet_nan<R>();
  if (p == R(0)) return R(0);
  if (p == R(1)) return std::numeric_limits<R>::infinity();
  return detail::invert_positive<R>(
      [d1, d2](R x) { return fisher_f_cdf(x, d1, d2); }, p, R(2));
}

}  // namespace statisticalc::dist
