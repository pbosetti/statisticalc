// statisticalc - header-only running statistics for C++20
//
// RunningStats<T> keeps a running window of the last N observations (or an
// unlimited one) and maintains the first four central moments incrementally,
// with Welford-style recursion formulas for both insertion and removal, so that
// every descriptive statistic is available in O(1) at any time.
//
// On top of the descriptive layer it offers inferential statistics:
//   * one-sample Student t-test on the mean, and its confidence interval;
//   * one-sample chi-squared test on the variance, and its confidence interval;
//   * two-sample tests obtained by comparing two instances (Welch or pooled
//     t-test, and the F test on the ratio of variances).
#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <ostream>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "distributions.hpp"

namespace statisticalc {

// ---------------------------------------------------------------------------
// Common types
// ---------------------------------------------------------------------------

/// Side of the alternative hypothesis H1, expressed on the true parameter.
enum class Alternative {
  two_sided,  ///< H1: parameter != null value
  less,       ///< H1: parameter <  null value
  greater     ///< H1: parameter >  null value
};

[[nodiscard]] inline const char *to_string(Alternative a) noexcept {
  switch (a) {
    case Alternative::less:
      return "less";
    case Alternative::greater:
      return "greater";
    default:
      return "two-sided";
  }
}

/// Options shared by all the tests. Designed for C++20 designated initializers:
///   s.t_test(0.0, {.alternative = Alternative::greater, .conf_level = 0.99});
struct TestOptions {
  Alternative alternative = Alternative::two_sided;
  double conf_level = 0.95;
  /// Two-sample t-test only: use the pooled variance (Student) instead of the
  /// Welch-Satterthwaite approximation, which is the default.
  bool equal_variance = false;
};

/// A closed interval, possibly unbounded on one side.
template <std::floating_point R = double>
struct Interval {
  R lower{};
  R upper{};

  [[nodiscard]] constexpr bool contains(R x) const noexcept {
    return x >= lower && x <= upper;
  }
  [[nodiscard]] constexpr R width() const noexcept { return upper - lower; }

  friend std::ostream &operator<<(std::ostream &os, const Interval &i) {
    return os << '[' << i.lower << ", " << i.upper << ']';
  }
};

/// Outcome of a hypothesis test.
template <std::floating_point R = double>
struct TestResult {
  std::string name{};  ///< human readable test name
  R statistic{};       ///< t, chi-squared or F value
  R dof1{};            ///< (first) number of degrees of freedom
  R dof2{};            ///< second dof, F test only (0 otherwise)
  R p_value{};         ///< p-value for the chosen alternative
  Alternative alternative = Alternative::two_sided;
  R estimate{};            ///< sample estimate of the parameter
  R null_value{};          ///< value of the parameter under H0
  Interval<R> conf_int{};  ///< confidence interval for the parameter
  R conf_level{};          ///< nominal coverage of conf_int

  /// True when H0 is rejected at the given significance level.
  [[nodiscard]] bool reject(R alpha = R(0.05)) const noexcept {
    return p_value < alpha;
  }
  /// Same as reject(), reads better in conditionals: if (result) { ... }
  [[nodiscard]] explicit operator bool() const noexcept { return reject(); }

  [[nodiscard]] std::string to_string() const {
    std::ostringstream os;
    os << *this;
    return os.str();
  }

  friend std::ostream &operator<<(std::ostream &os, const TestResult &r) {
    const auto flags = os.flags();
    const auto prec = os.precision();
    os << r.name << "\n  statistic = " << std::setprecision(6) << r.statistic
       << ", df = " << r.dof1;
    if (r.dof2 > R(0)) os << " and " << r.dof2;
    os << ", p-value = " << r.p_value
       << "\n  alternative hypothesis: true value is "
       << (r.alternative == Alternative::two_sided
               ? "not equal to "
               : (r.alternative == Alternative::less ? "less than "
                                                     : "greater than "))
       << r.null_value << "\n  " << std::setprecision(4) << (100 * r.conf_level)
       << "% confidence interval: " << std::setprecision(6) << r.conf_int
       << "\n  sample estimate: " << r.estimate << '\n';
    os.flags(flags);
    os.precision(prec);
    return os;
  }
};

/// Minimal description of a sample: everything the tests below need. Obtained
/// from RunningStats::summary(), it also allows tests between instances holding
/// different value types.
template <std::floating_point R = double>
struct Summary {
  std::size_t n{};
  R mean{};
  R variance{};  ///< sample variance (denominator n - 1)
  R minimum{};
  R maximum{};

  [[nodiscard]] R stddev() const { return std::sqrt(variance); }
  [[nodiscard]] R sem() const {
    return (n > 0) ? std::sqrt(variance / static_cast<R>(n))
                   : std::numeric_limits<R>::quiet_NaN();
  }
};

// ---------------------------------------------------------------------------
// p-values and confidence bounds
// ---------------------------------------------------------------------------

namespace detail {

template <std::floating_point R>
[[nodiscard]] inline R clamp_p(R p) noexcept {
  if (std::isnan(p)) return p;
  return (std::min)(R(1), (std::max)(R(0), p));
}

template <std::floating_point R>
[[nodiscard]] inline R t_p_value(R t, R df, Alternative alt) {
  switch (alt) {
    case Alternative::less:
      return dist::student_t_cdf(t, df);
    case Alternative::greater:
      return dist::student_t_sf(t, df);
    default:
      return clamp_p(R(2) * dist::student_t_sf(std::abs(t), df));
  }
}

template <std::floating_point R>
[[nodiscard]] inline R chi2_p_value(R x, R df, Alternative alt) {
  const R lower = dist::chi_squared_cdf(x, df);
  const R upper = dist::chi_squared_sf(x, df);
  switch (alt) {
    case Alternative::less:
      return lower;
    case Alternative::greater:
      return upper;
    default:
      return clamp_p(R(2) * (std::min)(lower, upper));
  }
}

template <std::floating_point R>
[[nodiscard]] inline R f_p_value(R x, R d1, R d2, Alternative alt) {
  const R lower = dist::fisher_f_cdf(x, d1, d2);
  const R upper = dist::fisher_f_sf(x, d1, d2);
  switch (alt) {
    case Alternative::less:
      return lower;
    case Alternative::greater:
      return upper;
    default:
      return clamp_p(R(2) * (std::min)(lower, upper));
  }
}

/// Confidence interval of the form estimate +/- t * standard_error.
template <std::floating_point R>
[[nodiscard]] inline Interval<R> t_interval(R estimate, R se, R df,
                                            Alternative alt, R conf) {
  constexpr R inf = std::numeric_limits<R>::infinity();
  const R alpha = R(1) - conf;
  switch (alt) {
    case Alternative::less:
      return {-inf, estimate + dist::student_t_quantile(conf, df) * se};
    case Alternative::greater:
      return {estimate - dist::student_t_quantile(conf, df) * se, inf};
    default: {
      const R t = dist::student_t_quantile(R(1) - alpha / R(2), df);
      return {estimate - t * se, estimate + t * se};
    }
  }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Tests on summaries (free functions, usable across different value types)
// ---------------------------------------------------------------------------

/// One-sample Student t-test on the mean.
template <std::floating_point R>
[[nodiscard]] TestResult<R> t_test(const Summary<R> &s, R mu0 = R(0),
                                   TestOptions o = {}) {
  TestResult<R> r;
  r.name = "One-sample t-test";
  r.alternative = o.alternative;
  r.null_value = mu0;
  r.conf_level = static_cast<R>(o.conf_level);
  r.estimate = s.mean;
  if (s.n < 2) {
    r.statistic = r.p_value = r.dof1 = std::numeric_limits<R>::quiet_NaN();
    r.conf_int = {r.statistic, r.statistic};
    return r;
  }
  const R df = static_cast<R>(s.n) - R(1);
  const R se = std::sqrt(s.variance / static_cast<R>(s.n));
  r.dof1 = df;
  r.statistic = (s.mean - mu0) / se;
  r.p_value = detail::t_p_value(r.statistic, df, o.alternative);
  r.conf_int = detail::t_interval(s.mean, se, df, o.alternative, r.conf_level);
  return r;
}

/// One-sample chi-squared test on the variance, H0: sigma^2 == sigma2_0.
template <std::floating_point R>
[[nodiscard]] TestResult<R> variance_test(const Summary<R> &s, R sigma2_0,
                                          TestOptions o = {}) {
  TestResult<R> r;
  r.name = "One-sample chi-squared test on the variance";
  r.alternative = o.alternative;
  r.null_value = sigma2_0;
  r.conf_level = static_cast<R>(o.conf_level);
  r.estimate = s.variance;
  if (s.n < 2 || !(sigma2_0 > R(0))) {
    r.statistic = r.p_value = r.dof1 = std::numeric_limits<R>::quiet_NaN();
    r.conf_int = {r.statistic, r.statistic};
    return r;
  }
  constexpr R inf = std::numeric_limits<R>::infinity();
  const R df = static_cast<R>(s.n) - R(1);
  const R scale = df * s.variance;
  r.dof1 = df;
  r.statistic = scale / sigma2_0;
  r.p_value = detail::chi2_p_value(r.statistic, df, o.alternative);
  const R alpha = R(1) - r.conf_level;
  switch (o.alternative) {
    case Alternative::less:
      r.conf_int = {R(0), scale / dist::chi_squared_quantile(alpha, df)};
      break;
    case Alternative::greater:
      r.conf_int = {scale / dist::chi_squared_quantile(R(1) - alpha, df), inf};
      break;
    default:
      r.conf_int = {scale / dist::chi_squared_quantile(R(1) - alpha / R(2), df),
                    scale / dist::chi_squared_quantile(alpha / R(2), df)};
      break;
  }
  return r;
}

/// Two-sample t-test on the difference of the means (a - b). Welch by default,
/// pooled (Student) when o.equal_variance is true.
template <std::floating_point R>
[[nodiscard]] TestResult<R> t_test(const Summary<R> &a, const Summary<R> &b,
                                   R diff0 = R(0), TestOptions o = {}) {
  TestResult<R> r;
  r.name = o.equal_variance ? "Two-sample t-test (pooled variance)"
                            : "Welch two-sample t-test";
  r.alternative = o.alternative;
  r.null_value = diff0;
  r.conf_level = static_cast<R>(o.conf_level);
  r.estimate = a.mean - b.mean;
  if (a.n < 2 || b.n < 2) {
    r.statistic = r.p_value = r.dof1 = std::numeric_limits<R>::quiet_NaN();
    r.conf_int = {r.statistic, r.statistic};
    return r;
  }
  const R na = static_cast<R>(a.n);
  const R nb = static_cast<R>(b.n);
  R se{};
  R df{};
  if (o.equal_variance) {
    df = na + nb - R(2);
    const R sp2 = ((na - R(1)) * a.variance + (nb - R(1)) * b.variance) / df;
    se = std::sqrt(sp2 * (R(1) / na + R(1) / nb));
  } else {
    const R va = a.variance / na;
    const R vb = b.variance / nb;
    se = std::sqrt(va + vb);
    df =
        (va + vb) * (va + vb) / (va * va / (na - R(1)) + vb * vb / (nb - R(1)));
  }
  r.dof1 = df;
  r.statistic = (r.estimate - diff0) / se;
  r.p_value = detail::t_p_value(r.statistic, df, o.alternative);
  r.conf_int =
      detail::t_interval(r.estimate, se, df, o.alternative, r.conf_level);
  return r;
}

/// F test on the ratio of the variances (a / b), H0: ratio == ratio0.
template <std::floating_point R>
[[nodiscard]] TestResult<R> f_test(const Summary<R> &a, const Summary<R> &b,
                                   R ratio0 = R(1), TestOptions o = {}) {
  TestResult<R> r;
  r.name = "F test on the ratio of two variances";
  r.alternative = o.alternative;
  r.null_value = ratio0;
  r.conf_level = static_cast<R>(o.conf_level);
  r.estimate = a.variance / b.variance;
  if (a.n < 2 || b.n < 2 || !(ratio0 > R(0))) {
    r.statistic = r.p_value = r.dof1 = r.dof2 =
        std::numeric_limits<R>::quiet_NaN();
    r.conf_int = {r.statistic, r.statistic};
    return r;
  }
  constexpr R inf = std::numeric_limits<R>::infinity();
  const R d1 = static_cast<R>(a.n) - R(1);
  const R d2 = static_cast<R>(b.n) - R(1);
  r.dof1 = d1;
  r.dof2 = d2;
  r.statistic = r.estimate / ratio0;
  r.p_value = detail::f_p_value(r.statistic, d1, d2, o.alternative);
  const R alpha = R(1) - r.conf_level;
  switch (o.alternative) {
    case Alternative::less:
      r.conf_int = {R(0), r.estimate / dist::fisher_f_quantile(alpha, d1, d2)};
      break;
    case Alternative::greater:
      r.conf_int = {r.estimate / dist::fisher_f_quantile(R(1) - alpha, d1, d2),
                    inf};
      break;
    default:
      r.conf_int = {
          r.estimate / dist::fisher_f_quantile(R(1) - alpha / R(2), d1, d2),
          r.estimate / dist::fisher_f_quantile(alpha / R(2), d1, d2)};
      break;
  }
  return r;
}

// ---------------------------------------------------------------------------
// RunningStats
// ---------------------------------------------------------------------------

/// Default accumulator type for a given value type: double is used for every
/// value type but long double, which keeps its own precision.
template <typename T>
using default_accumulator_t =
    std::conditional_t<std::is_same_v<T, long double>, long double, double>;

/// Running window of values with incremental descriptive and inferential
/// statistics.
///
/// @tparam T value type of the observations (any arithmetic type).
/// @tparam A floating point type used for the accumulators.
template <typename T = double, std::floating_point A = default_accumulator_t<T>>
  requires std::is_arithmetic_v<T>
class RunningStats {
 public:
  using value_type = T;
  using stat_type = A;
  using size_type = std::size_t;

  /// Capacity value meaning "keep every observation, never evict".
  static constexpr size_type unlimited = 0;

  // -- construction ---------------------------------------------------------

  /// Unlimited accumulator: no window, values are not retained.
  RunningStats() noexcept = default;

  /// Window of at most `window` values; `unlimited` (0) for no bound.
  explicit RunningStats(size_type window) : _capacity(window) {
    if (_capacity != unlimited) _buf.resize(_capacity);
  }

  /// Window of at most `window` values, pre-filled from a range.
  template <std::ranges::input_range Rng>
    requires std::convertible_to<std::ranges::range_value_t<Rng>, T>
  RunningStats(size_type window, Rng &&values) : RunningStats(window) {
    push(std::forward<Rng>(values));
  }

  // -- feeding --------------------------------------------------------------

  /// Add one observation, evicting the oldest one if the window is full.
  void push(T x) {
    if (bounded() && _n == _capacity) pop();
    if (bounded()) _buf[(_head + _n) % _capacity] = x;
    ++_n;
    add_moment(static_cast<A>(x));
    update_extrema(x);
    ++_total;
  }

  /// Add every element of a range.
  template <std::ranges::input_range Rng>
    requires std::convertible_to<std::ranges::range_value_t<Rng>, T>
  void push(Rng &&values) {
    for (auto &&v : values) push(static_cast<T>(v));
  }

  void push(std::initializer_list<T> values) {
    for (T v : values) push(v);
  }

  /// Callable form, handy with std::for_each and friends.
  void operator()(T x) { push(x); }

  /// Drop the oldest observation. Only meaningful on a bounded window, since an
  /// unlimited accumulator does not retain the values.
  void pop() {
    if (_n == 0) return;
    if (!bounded())
      throw std::logic_error(
          "statisticalc: cannot pop from an unlimited accumulator, values are "
          "not retained");
    const T x = _buf[_head];
    remove_moment(static_cast<A>(x));
    const std::uint64_t oldest = _total - _n;
    if (!_min_q.empty() && _min_q.front().first == oldest) _min_q.pop_front();
    if (!_max_q.empty() && _max_q.front().first == oldest) _max_q.pop_front();
    _head = (_head + 1) % _capacity;
    --_n;
  }

  /// Forget every observation, keeping the current window size.
  void clear() noexcept {
    _n = _head = 0;
    _total = 0;
    _mean = _m2 = _m3 = _m4 = A(0);
    _min = _max = T{};
    _min_q.clear();
    _max_q.clear();
  }

  /// Change the window size.
  /// Growing or shrinking a bounded window keeps the most recent values
  /// (the excess oldest ones are evicted); switching a bounded window to
  /// `unlimited` keeps the current moments but stops retaining values.
  /// Turning a non-empty unlimited accumulator into a bounded window is not
  /// possible, because its past values are gone: it throws std::logic_error.
  void resize(size_type window) {
    if (window == _capacity) return;
    if (window == unlimited) {
      if (_n > 0) {
        _min = min();
        _max = max();
      }
      _buf.clear();
      _buf.shrink_to_fit();
      _min_q.clear();
      _max_q.clear();
      _head = 0;
      _capacity = unlimited;
      return;
    }
    if (!bounded()) {
      if (_n != 0)
        throw std::logic_error(
            "statisticalc: cannot turn a non-empty unlimited accumulator into "
            "a bounded window, past values are not retained");
      _capacity = window;
      _buf.assign(_capacity, T{});
      return;
    }
    while (_n > window) pop();
    std::vector<T> kept = values();
    _buf.assign(window, T{});
    for (size_type i = 0; i < kept.size(); ++i) _buf[i] = kept[i];
    _head = 0;
    _capacity = window;
  }

  /// Recompute the moments from the stored values. The incremental formulas
  /// accumulate round-off over very long runs; on a bounded window this
  /// restores full accuracy. No-op on an unlimited accumulator.
  void refresh() {
    if (!bounded() || _n == 0) return;
    const std::vector<T> v = values();
    const std::uint64_t total = _total;
    const size_type cap = _capacity;
    clear();
    _capacity = cap;
    // The extrema deques index the values by their ordinal among all the values
    // ever pushed, and eviction relies on it: replay the window with the very
    // ordinals it had, so that the running count comes out unchanged.
    _total = total - static_cast<std::uint64_t>(v.size());
    for (T x : v) push(x);
  }

  // -- window state ---------------------------------------------------------

  [[nodiscard]] bool bounded() const noexcept { return _capacity != unlimited; }
  [[nodiscard]] size_type capacity() const noexcept { return _capacity; }
  [[nodiscard]] size_type size() const noexcept { return _n; }
  [[nodiscard]] bool empty() const noexcept { return _n == 0; }
  [[nodiscard]] bool full() const noexcept {
    return bounded() && _n == _capacity;
  }
  /// Number of observations pushed since construction (or the last clear()).
  [[nodiscard]] std::uint64_t total_count() const noexcept { return _total; }

  /// i-th value of the window, from the oldest (0) to the newest (size() - 1).
  [[nodiscard]] T at(size_type i) const {
    require_values("at()");
    if (i >= _n) throw std::out_of_range("statisticalc: index out of range");
    return _buf[(_head + i) % _capacity];
  }
  [[nodiscard]] T operator[](size_type i) const { return at(i); }
  [[nodiscard]] T oldest() const { return at(0); }
  [[nodiscard]] T newest() const { return at(_n - 1); }

  /// Copy of the window, from the oldest to the newest value.
  [[nodiscard]] std::vector<T> values() const {
    require_values("values()");
    std::vector<T> out;
    out.reserve(_n);
    for (size_type i = 0; i < _n; ++i)
      out.push_back(_buf[(_head + i) % _capacity]);
    return out;
  }

  // -- descriptive statistics ----------------------------------------------

  [[nodiscard]] A sum() const noexcept { return _mean * static_cast<A>(_n); }
  [[nodiscard]] A mean() const noexcept { return (_n > 0) ? _mean : nan(); }

  /// Sample variance (denominator n - 1).
  [[nodiscard]] A variance() const noexcept {
    return (_n > 1) ? _m2 / (static_cast<A>(_n) - A(1)) : nan();
  }
  /// Population variance (denominator n).
  [[nodiscard]] A population_variance() const noexcept {
    return (_n > 0) ? _m2 / static_cast<A>(_n) : nan();
  }
  [[nodiscard]] A stddev() const noexcept { return std::sqrt(variance()); }
  [[nodiscard]] A population_stddev() const noexcept {
    return std::sqrt(population_variance());
  }
  /// Standard error of the mean.
  [[nodiscard]] A sem() const noexcept {
    return (_n > 1) ? std::sqrt(variance() / static_cast<A>(_n)) : nan();
  }
  /// Coefficient of variation, stddev / mean.
  [[nodiscard]] A cv() const noexcept { return stddev() / _mean; }
  /// Root mean square.
  [[nodiscard]] A rms() const noexcept {
    return (_n > 0) ? std::sqrt(_m2 / static_cast<A>(_n) + _mean * _mean)
                    : nan();
  }
  /// Sum of the squared deviations from the mean (the M2 accumulator).
  [[nodiscard]] A sum_of_squares() const noexcept { return _m2; }

  /// Adjusted Fisher-Pearson standardised moment coefficient (sample skewness).
  [[nodiscard]] A skewness() const noexcept {
    if (_n < 3 || _m2 <= A(0)) return nan();
    const A n = static_cast<A>(_n);
    return std::sqrt(n * (n - A(1))) / (n - A(2)) * population_skewness();
  }
  /// Skewness of the window seen as a whole population.
  [[nodiscard]] A population_skewness() const noexcept {
    if (_n < 2 || _m2 <= A(0)) return nan();
    const A n = static_cast<A>(_n);
    return std::sqrt(n) * _m3 / std::pow(_m2, A(1.5));
  }
  /// Sample excess kurtosis (0 for a normal sample).
  [[nodiscard]] A kurtosis() const noexcept {
    if (_n < 4 || _m2 <= A(0)) return nan();
    const A n = static_cast<A>(_n);
    return ((n + A(1)) * population_kurtosis() + A(6)) * (n - A(1)) /
           ((n - A(2)) * (n - A(3)));
  }
  /// Excess kurtosis of the window seen as a whole population.
  [[nodiscard]] A population_kurtosis() const noexcept {
    if (_n < 2 || _m2 <= A(0)) return nan();
    const A n = static_cast<A>(_n);
    return n * _m4 / (_m2 * _m2) - A(3);
  }

  [[nodiscard]] T min() const {
    if (_n == 0) throw std::logic_error("statisticalc: min() on empty window");
    return bounded() ? _min_q.front().second : _min;
  }
  [[nodiscard]] T max() const {
    if (_n == 0) throw std::logic_error("statisticalc: max() on empty window");
    return bounded() ? _max_q.front().second : _max;
  }
  [[nodiscard]] T range() const { return static_cast<T>(max() - min()); }

  /// Quantile of type 7 (the default of R and numpy), by linear interpolation
  /// on a sorted copy of the window. Bounded windows only.
  [[nodiscard]] A quantile(double p) const {
    require_values("quantile()");
    if (_n == 0)
      throw std::logic_error("statisticalc: quantile() on empty window");
    if (!(p >= 0.0 && p <= 1.0))
      throw std::invalid_argument(
          "statisticalc: quantile probability outside [0, 1]");
    std::vector<T> v = values();
    std::sort(v.begin(), v.end());
    const double h = p * static_cast<double>(_n - 1);
    const auto lo = static_cast<size_type>(std::floor(h));
    const auto hi = static_cast<size_type>(std::ceil(h));
    const A frac = static_cast<A>(h - static_cast<double>(lo));
    return static_cast<A>(v[lo]) +
           frac * (static_cast<A>(v[hi]) - static_cast<A>(v[lo]));
  }
  [[nodiscard]] A median() const { return quantile(0.5); }
  /// Interquartile range, Q3 - Q1.
  [[nodiscard]] A iqr() const { return quantile(0.75) - quantile(0.25); }

  /// Standardised score of a value with respect to the window.
  [[nodiscard]] A zscore(T x) const noexcept {
    return (static_cast<A>(x) - _mean) / stddev();
  }

  /// Snapshot of the sample, to be fed to the free test functions.
  [[nodiscard]] Summary<A> summary() const {
    Summary<A> s;
    s.n = _n;
    s.mean = mean();
    s.variance = variance();
    s.minimum = (_n > 0) ? static_cast<A>(min()) : nan();
    s.maximum = (_n > 0) ? static_cast<A>(max()) : nan();
    return s;
  }
  /// Implicit conversion, so an instance can be passed wherever a Summary is.
  [[nodiscard]] operator Summary<A>() const { return summary(); }

  // -- inferential statistics, one sample ----------------------------------

  /// Student t-test on the mean, H0: mu == mu0.
  [[nodiscard]] TestResult<A> t_test(A mu0 = A(0), TestOptions o = {}) const {
    return statisticalc::t_test(summary(), mu0, o);
  }
  /// Chi-squared test on the variance, H0: sigma^2 == sigma2_0.
  [[nodiscard]] TestResult<A> variance_test(A sigma2_0,
                                            TestOptions o = {}) const {
    return statisticalc::variance_test(summary(), sigma2_0, o);
  }
  /// Confidence interval for the mean.
  [[nodiscard]] Interval<A> mean_ci(double conf_level = 0.95) const {
    return t_test(A(0), {.conf_level = conf_level}).conf_int;
  }
  /// Confidence interval for the variance.
  [[nodiscard]] Interval<A> variance_ci(double conf_level = 0.95) const {
    return variance_test(A(1), {.conf_level = conf_level}).conf_int;
  }
  /// Confidence interval for the standard deviation.
  [[nodiscard]] Interval<A> stddev_ci(double conf_level = 0.95) const {
    const Interval<A> v = variance_ci(conf_level);
    return {std::sqrt(v.lower), std::sqrt(v.upper)};
  }

  // -- friends: usability ---------------------------------------------------

  /// Stream an observation in: stats << 1.0 << 2.0 << 3.0;
  friend RunningStats &operator<<(RunningStats &s, T x) {
    s.push(x);
    return s;
  }
  /// Same, in accumulation form: stats += 1.0;
  friend RunningStats &operator+=(RunningStats &s, T x) {
    s.push(x);
    return s;
  }
  /// Feed a whole range: stats += std::vector{1.0, 2.0};
  template <std::ranges::input_range Rng>
    requires std::convertible_to<std::ranges::range_value_t<Rng>, T>
  friend RunningStats &operator+=(RunningStats &s, Rng &&values) {
    s.push(std::forward<Rng>(values));
    return s;
  }

  /// One line summary of the window.
  friend std::ostream &operator<<(std::ostream &os, const RunningStats &s) {
    const auto prec = os.precision();
    os << "RunningStats[n=" << s._n << '/';
    if (s.bounded())
      os << s._capacity;
    else
      os << "inf";
    if (s._n > 0) {
      os << std::setprecision(6) << ", mean=" << s.mean();
      if (s._n > 1) os << ", sd=" << s.stddev();
      os << ", min=" << +s.min() << ", max=" << +s.max();
    }
    os << ']';
    os.precision(prec);
    return os;
  }

  /// Merge two samples into a single unlimited accumulator: the moments are
  /// combined pairwise (Chan-Golub-LeVeque), the values are not retained.
  friend RunningStats merge(const RunningStats &a, const RunningStats &b) {
    if (a._n == 0) return unlimited_copy(b);
    if (b._n == 0) return unlimited_copy(a);
    RunningStats out;
    const A na = static_cast<A>(a._n);
    const A nb = static_cast<A>(b._n);
    const A n = na + nb;
    const A d = b._mean - a._mean;
    const A d2 = d * d;
    out._n = a._n + b._n;
    out._total = a._total + b._total;
    out._mean = a._mean + d * nb / n;
    out._m2 = a._m2 + b._m2 + d2 * na * nb / n;
    out._m3 = a._m3 + b._m3 + d2 * d * na * nb * (na - nb) / (n * n) +
              A(3) * d * (na * b._m2 - nb * a._m2) / n;
    out._m4 = a._m4 + b._m4 +
              d2 * d2 * na * nb * (na * na - na * nb + nb * nb) / (n * n * n) +
              A(6) * d2 * (na * na * b._m2 + nb * nb * a._m2) / (n * n) +
              A(4) * d * (na * b._m3 - nb * a._m3) / n;
    out._min = (std::min)(a.min(), b.min());
    out._max = (std::max)(a.max(), b.max());
    return out;
  }
  /// Merge operator, see merge().
  friend RunningStats operator+(const RunningStats &a, const RunningStats &b) {
    return merge(a, b);
  }

  friend void swap(RunningStats &a, RunningStats &b) noexcept {
    using std::swap;
    swap(a._capacity, b._capacity);
    swap(a._buf, b._buf);
    swap(a._head, b._head);
    swap(a._n, b._n);
    swap(a._total, b._total);
    swap(a._mean, b._mean);
    swap(a._m2, b._m2);
    swap(a._m3, b._m3);
    swap(a._m4, b._m4);
    swap(a._min_q, b._min_q);
    swap(a._max_q, b._max_q);
    swap(a._min, b._min);
    swap(a._max, b._max);
  }

  // -- friends: two sample tests -------------------------------------------

  /// Two-sample t-test on the difference of the means (a - b).
  friend TestResult<A> t_test(const RunningStats &a, const RunningStats &b,
                              A diff0 = A(0), TestOptions o = {}) {
    return statisticalc::t_test(a.summary(), b.summary(), diff0, o);
  }
  /// F test on the ratio of the variances (a / b).
  friend TestResult<A> f_test(const RunningStats &a, const RunningStats &b,
                              A ratio0 = A(1), TestOptions o = {}) {
    return statisticalc::f_test(a.summary(), b.summary(), ratio0, o);
  }
  /// Pooled (weighted) variance of the two samples.
  friend A pooled_variance(const RunningStats &a, const RunningStats &b) {
    if (a._n < 2 || b._n < 2) return nan();
    const A na = static_cast<A>(a._n);
    const A nb = static_cast<A>(b._n);
    return ((na - A(1)) * a.variance() + (nb - A(1)) * b.variance()) /
           (na + nb - A(2));
  }
  /// Cohen's d effect size of the difference of the means.
  friend A cohens_d(const RunningStats &a, const RunningStats &b) {
    return (a.mean() - b.mean()) / std::sqrt(pooled_variance(a, b));
  }
  /// True when the t-test does not reject the equality of the means.
  friend bool same_mean(const RunningStats &a, const RunningStats &b,
                        double alpha = 0.05, TestOptions o = {}) {
    return !statisticalc::t_test(a.summary(), b.summary(), A(0), o)
                .reject(static_cast<A>(alpha));
  }
  /// True when the F test does not reject the equality of the variances.
  friend bool same_variance(const RunningStats &a, const RunningStats &b,
                            double alpha = 0.05, TestOptions o = {}) {
    return !statisticalc::f_test(a.summary(), b.summary(), A(1), o)
                .reject(static_cast<A>(alpha));
  }

 private:
  static constexpr A nan() noexcept {
    return std::numeric_limits<A>::quiet_NaN();
  }

  static RunningStats unlimited_copy(const RunningStats &s) {
    RunningStats out;
    out._n = s._n;
    out._total = s._total;
    out._mean = s._mean;
    out._m2 = s._m2;
    out._m3 = s._m3;
    out._m4 = s._m4;
    if (s._n > 0) {
      out._min = s.min();
      out._max = s.max();
    }
    return out;
  }

  void require_values(const char *what) const {
    if (!bounded())
      throw std::logic_error(std::string("statisticalc: ") + what +
                             " requires a bounded window, an unlimited "
                             "accumulator does not retain the values");
  }

  /// Welford recursion, extended to the third and fourth central moments
  /// (Pebay). _n has already been incremented when this is called.
  void add_moment(A x) {
    const A n = static_cast<A>(_n);
    const A n1 = n - A(1);
    const A delta = x - _mean;
    const A delta_n = delta / n;
    const A delta_n2 = delta_n * delta_n;
    const A term = delta * delta_n * n1;
    _mean += delta_n;
    _m4 += term * delta_n2 * (n * n - A(3) * n + A(3)) + A(6) * delta_n2 * _m2 -
           A(4) * delta_n * _m3;
    _m3 += term * delta_n * (n - A(2)) - A(3) * delta_n * _m2;
    _m2 += term;
  }

  /// Exact inverse of add_moment(): removes an observation from the moments.
  /// _n still counts the value being removed when this is called.
  void remove_moment(A x) {
    const A n = static_cast<A>(_n);
    const A n1 = n - A(1);
    if (n1 <= A(0)) {
      _mean = _m2 = _m3 = _m4 = A(0);
      return;
    }
    const A delta_n = (x - _mean) / n1;
    const A delta_n2 = delta_n * delta_n;
    const A delta = n * delta_n;
    const A term = delta * delta_n * n1;
    const A m2 = _m2 - term;
    const A m3 = _m3 - term * delta_n * (n - A(2)) + A(3) * delta_n * m2;
    const A m4 = _m4 - term * delta_n2 * (n * n - A(3) * n + A(3)) -
                 A(6) * delta_n2 * m2 + A(4) * delta_n * m3;
    _mean -= delta_n;
    _m2 = m2;
    _m3 = m3;
    _m4 = m4;
  }

  /// Monotonic deques give the extrema of a sliding window in O(1) amortised;
  /// an unlimited accumulator only needs two scalars.
  void update_extrema(T x) {
    if (!bounded()) {
      if (_n == 1) {
        _min = _max = x;
      } else {
        if (x < _min) _min = x;
        if (x > _max) _max = x;
      }
      return;
    }
    while (!_min_q.empty() && !(_min_q.back().second < x)) _min_q.pop_back();
    _min_q.emplace_back(_total, x);
    while (!_max_q.empty() && !(_max_q.back().second > x)) _max_q.pop_back();
    _max_q.emplace_back(_total, x);
  }

  size_type _capacity = unlimited;
  std::vector<T> _buf{};     ///< ring buffer, bounded windows only
  size_type _head = 0;       ///< position of the oldest value in _buf
  size_type _n = 0;          ///< values currently in the window
  std::uint64_t _total = 0;  ///< values ever pushed

  A _mean = A(0);  ///< running mean
  A _m2 = A(0);    ///< sum of squared deviations
  A _m3 = A(0);    ///< third central moment accumulator
  A _m4 = A(0);    ///< fourth central moment accumulator

  std::deque<std::pair<std::uint64_t, T>> _min_q{};  ///< bounded windows
  std::deque<std::pair<std::uint64_t, T>> _max_q{};
  T _min = T{};  ///< unlimited accumulator
  T _max = T{};
};

using RunningStatsd = RunningStats<double>;
using RunningStatsf = RunningStats<float>;

}  // namespace statisticalc
