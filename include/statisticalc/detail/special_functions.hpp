// statisticalc - header-only running statistics for C++20
//
// Special functions needed by the statistical distributions: the regularized
// incomplete gamma and beta functions, plus a generic bracketing/bisection
// inverter used to obtain quantiles from cumulative distribution functions.
//
// Everything here is self-contained: only <cmath> from the standard library is
// required (std::lgamma, std::erfc, ...).
#pragma once

#include <cmath>
#include <concepts>
#include <limits>
#include <stdexcept>

namespace statisticalc::detail {

/// Maximum number of iterations for the series/continued fraction expansions.
inline constexpr int max_iterations = 500;

template <std::floating_point R>
[[nodiscard]] constexpr R eps() noexcept {
  return std::numeric_limits<R>::epsilon();
}

/// Smallest number that can safely be inverted, used to protect the modified
/// Lentz algorithm against zero denominators.
template <std::floating_point R>
[[nodiscard]] constexpr R fpmin() noexcept {
  return std::numeric_limits<R>::min() / std::numeric_limits<R>::epsilon();
}

template <std::floating_point R>
[[nodiscard]] constexpr R quiet_nan() noexcept {
  return std::numeric_limits<R>::quiet_NaN();
}

// ---------------------------------------------------------------------------
// Regularized incomplete gamma function
// ---------------------------------------------------------------------------

/// Series expansion of P(a, x), convergent for x < a + 1.
template <std::floating_point R>
[[nodiscard]] R gamma_p_series(R a, R x) {
  R ap = a;
  R del = R(1) / a;
  R sum = del;
  for (int i = 0; i < max_iterations; ++i) {
    ap += R(1);
    del *= x / ap;
    sum += del;
    if (std::abs(del) < std::abs(sum) * eps<R>()) break;
  }
  return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

/// Continued fraction (modified Lentz) for Q(a, x), convergent for x > a + 1.
template <std::floating_point R>
[[nodiscard]] R gamma_q_continued_fraction(R a, R x) {
  const R tiny = fpmin<R>();
  R b = x + R(1) - a;
  R c = R(1) / tiny;
  R d = R(1) / b;
  R h = d;
  for (int i = 1; i <= max_iterations; ++i) {
    const R an = -R(i) * (R(i) - a);
    b += R(2);
    d = an * d + b;
    if (std::abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (std::abs(c) < tiny) c = tiny;
    d = R(1) / d;
    const R del = d * c;
    h *= del;
    if (std::abs(del - R(1)) <= eps<R>()) break;
  }
  return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}

/// Regularized lower incomplete gamma function P(a, x).
template <std::floating_point R>
[[nodiscard]] R gamma_p(R a, R x) {
  if (!(a > R(0)) || x < R(0) || std::isnan(a) || std::isnan(x))
    return quiet_nan<R>();
  if (x == R(0)) return R(0);
  if (x < a + R(1)) return gamma_p_series(a, x);
  return R(1) - gamma_q_continued_fraction(a, x);
}

/// Regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x).
template <std::floating_point R>
[[nodiscard]] R gamma_q(R a, R x) {
  if (!(a > R(0)) || x < R(0) || std::isnan(a) || std::isnan(x))
    return quiet_nan<R>();
  if (x == R(0)) return R(1);
  if (x < a + R(1)) return R(1) - gamma_p_series(a, x);
  return gamma_q_continued_fraction(a, x);
}

// ---------------------------------------------------------------------------
// Regularized incomplete beta function
// ---------------------------------------------------------------------------

/// Continued fraction (modified Lentz) used by beta_i().
template <std::floating_point R>
[[nodiscard]] R beta_continued_fraction(R a, R b, R x) {
  const R tiny = fpmin<R>();
  const R qab = a + b;
  const R qap = a + R(1);
  const R qam = a - R(1);
  R c = R(1);
  R d = R(1) - qab * x / qap;
  if (std::abs(d) < tiny) d = tiny;
  d = R(1) / d;
  R h = d;
  for (int m = 1; m <= max_iterations; ++m) {
    const R rm = R(m);
    const R m2 = R(2) * rm;
    // Even step of the recurrence.
    R aa = rm * (b - rm) * x / ((qam + m2) * (a + m2));
    d = R(1) + aa * d;
    if (std::abs(d) < tiny) d = tiny;
    c = R(1) + aa / c;
    if (std::abs(c) < tiny) c = tiny;
    d = R(1) / d;
    h *= d * c;
    // Odd step of the recurrence.
    aa = -(a + rm) * (qab + rm) * x / ((a + m2) * (qap + m2));
    d = R(1) + aa * d;
    if (std::abs(d) < tiny) d = tiny;
    c = R(1) + aa / c;
    if (std::abs(c) < tiny) c = tiny;
    d = R(1) / d;
    const R del = d * c;
    h *= del;
    if (std::abs(del - R(1)) <= eps<R>()) break;
  }
  return h;
}

/// Regularized incomplete beta function I_x(a, b).
template <std::floating_point R>
[[nodiscard]] R beta_i(R a, R b, R x) {
  if (!(a > R(0)) || !(b > R(0)) || std::isnan(x)) return quiet_nan<R>();
  if (x <= R(0)) return R(0);
  if (x >= R(1)) return R(1);
  const R front =
      std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) +
               a * std::log(x) + b * std::log1p(-x));
  if (x < (a + R(1)) / (a + b + R(2)))
    return front * beta_continued_fraction(a, b, x) / a;
  return R(1) - front * beta_continued_fraction(b, a, R(1) - x) / b;
}

// ---------------------------------------------------------------------------
// Generic inversion of a monotonically increasing function
// ---------------------------------------------------------------------------

/// Bisection on a monotonically increasing f with f(lo) <= 0 <= f(hi).
/// Iterates down to machine precision; the bracket must be valid.
template <std::floating_point R, typename F>
[[nodiscard]] R bisect(F &&f, R lo, R hi) {
  R flo = f(lo);
  if (flo == R(0)) return lo;
  for (int i = 0; i < 200; ++i) {
    const R mid = R(0.5) * (lo + hi);
    if (mid <= lo || mid >= hi) break;  // exhausted the representable range
    const R fm = f(mid);
    if (fm == R(0)) return mid;
    if ((fm < R(0)) == (flo < R(0))) {
      lo = mid;
      flo = fm;
    } else {
      hi = mid;
    }
    // Relative convergence: an absolute floor would stop far too early when the
    // root sits close to zero, as it does in the tails of the quantiles.
    if (std::abs(hi - lo) <=
        R(4) * eps<R>() * (std::max)(std::abs(lo), std::abs(hi)))
      break;
  }
  return R(0.5) * (lo + hi);
}

/// Invert an increasing CDF on [0, +inf): finds x such that cdf(x) == p.
template <std::floating_point R, typename F>
[[nodiscard]] R invert_positive(F &&cdf, R p, R guess = R(1)) {
  R hi = (guess > R(0)) ? guess : R(1);
  for (int i = 0; i < 200 && cdf(hi) < p; ++i) hi *= R(2);
  R lo = R(0);
  return bisect<R>([&](R x) { return cdf(x) - p; }, lo, hi);
}

/// Invert an increasing CDF on (-inf, +inf): finds x such that cdf(x) == p.
template <std::floating_point R, typename F>
[[nodiscard]] R invert_real(F &&cdf, R p, R guess = R(1)) {
  R hi = (guess > R(0)) ? guess : R(1);
  for (int i = 0; i < 200 && cdf(hi) < p; ++i) hi *= R(2);
  R lo = -hi;
  for (int i = 0; i < 200 && cdf(lo) > p; ++i) lo *= R(2);
  return bisect<R>([&](R x) { return cdf(x) - p; }, lo, hi);
}

}  // namespace statisticalc::detail
