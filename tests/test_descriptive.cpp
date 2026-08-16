// Descriptive statistics computed with the recursion formulas.
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numeric>
#include <random>
#include <statisticalc/statisticalc.hpp>
#include <vector>

#include "reference_data.hpp"

using namespace statisticalc;
using ref::WithinAbs;
using ref::WithinRel;

namespace {

/// Straightforward two-pass computation, used as an independent reference.
struct TwoPass {
  double mean{}, m2{}, m3{}, m4{};
  explicit TwoPass(const std::vector<double> &v) {
    const auto n = static_cast<double>(v.size());
    mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
    for (double x : v) {
      const double d = x - mean;
      m2 += d * d;
      m3 += d * d * d;
      m4 += d * d * d * d;
    }
  }
  [[nodiscard]] double variance(std::size_t n) const {
    return m2 / (static_cast<double>(n) - 1.0);
  }
};

}  // namespace

TEST_CASE("moments of a known sample", "[descriptive]") {
  RunningStats<double> s(ref::a.size(), ref::a);

  CHECK(s.size() == 10);
  CHECK(s.full());
  CHECK(s.total_count() == 10);
  CHECK_THAT(s.mean(), WithinRel(ref::a_mean, ref::tol));
  CHECK_THAT(s.variance(), WithinRel(ref::a_var, ref::tol));
  CHECK_THAT(s.population_variance(), WithinRel(ref::a_popvar, ref::tol));
  CHECK_THAT(s.stddev(), WithinRel(ref::a_sd, ref::tol));
  CHECK_THAT(s.sem(), WithinRel(ref::a_sem, ref::tol));
  CHECK_THAT(s.skewness(), WithinRel(ref::a_skew, ref::tol));
  CHECK_THAT(s.population_skewness(), WithinRel(ref::a_popskew, ref::tol));
  CHECK_THAT(s.kurtosis(), WithinRel(ref::a_kurt, ref::tol));
  CHECK_THAT(s.population_kurtosis(), WithinRel(ref::a_popkurt, ref::tol));
  CHECK_THAT(s.rms(), WithinRel(ref::a_rms, ref::tol));
  CHECK_THAT(s.sum(), WithinRel(33.8, ref::tol));
  CHECK_THAT(s.cv(), WithinRel(ref::a_sd / ref::a_mean, ref::tol));
  CHECK(s.min() == 1.9);
  CHECK(s.max() == 5.1);
  CHECK_THAT(s.range(), WithinRel(3.2, 1e-12));
}

TEST_CASE("order statistics", "[descriptive]") {
  RunningStats<double> s(ref::a.size(), ref::a);
  CHECK_THAT(s.median(), WithinRel(ref::a_median, ref::tol));
  CHECK_THAT(s.quantile(0.25), WithinRel(ref::a_q25, ref::tol));
  CHECK_THAT(s.quantile(0.75), WithinRel(ref::a_q75, ref::tol));
  CHECK_THAT(s.iqr(), WithinRel(ref::a_q75 - ref::a_q25, ref::tol));
  CHECK(s.quantile(0.0) == s.min());
  CHECK(s.quantile(1.0) == s.max());
  CHECK_THROWS_AS(s.quantile(1.5), std::invalid_argument);

  RunningStats<double> b(ref::b.size(), ref::b);
  CHECK_THAT(b.median(), WithinRel(ref::b_median, ref::tol));
  CHECK_THAT(b.skewness(), WithinRel(ref::b_skew, ref::tol));
  CHECK_THAT(b.kurtosis(), WithinRel(ref::b_kurt, ref::tol));
}

TEST_CASE("degenerate sample sizes", "[descriptive]") {
  RunningStats<double> s;
  CHECK(s.empty());
  CHECK(std::isnan(s.mean()));
  CHECK(std::isnan(s.variance()));
  CHECK_THROWS_AS(s.min(), std::logic_error);

  s.push(4.0);
  CHECK_THAT(s.mean(), WithinRel(4.0, 1e-15));
  CHECK(std::isnan(s.variance()));  // needs 2 values
  CHECK(std::isnan(s.skewness()));  // needs 3 values
  CHECK(std::isnan(s.kurtosis()));  // needs 4 values
  CHECK(s.population_variance() == 0.0);

  s.push(6.0);
  CHECK_THAT(s.variance(), WithinRel(2.0, 1e-14));
  CHECK_THAT(s.stddev(), WithinRel(std::sqrt(2.0), 1e-14));
  CHECK(std::isnan(s.skewness()));

  s.push(8.0);
  CHECK_THAT(s.mean(), WithinRel(6.0, 1e-14));
  CHECK_THAT(s.skewness(), WithinAbs(0.0, 1e-14));  // symmetric
  CHECK(std::isnan(s.kurtosis()));

  SECTION("a constant sample has zero variance and undefined shape") {
    RunningStats<double> c;
    for (int i = 0; i < 10; ++i) c.push(3.0);
    CHECK_THAT(c.variance(), WithinAbs(0.0, 1e-15));
    CHECK(std::isnan(c.skewness()));
    CHECK(std::isnan(c.kurtosis()));
  }
}

TEST_CASE("recursion matches the two-pass formulas on random data",
          "[descriptive]") {
  std::mt19937 rng(20260816);
  std::lognormal_distribution<double> dis(0.0, 0.75);  // strongly skewed
  std::vector<double> v(500);
  std::generate(v.begin(), v.end(), [&] { return dis(rng); });

  RunningStats<double> s;  // unlimited accumulator
  s.push(v);
  const TwoPass tp(v);

  CHECK_THAT(s.mean(), WithinRel(tp.mean, 1e-12));
  CHECK_THAT(s.sum_of_squares(), WithinRel(tp.m2, 1e-10));
  CHECK_THAT(s.variance(), WithinRel(tp.variance(v.size()), 1e-10));
  const double n = static_cast<double>(v.size());
  CHECK_THAT(s.population_skewness(),
             WithinRel(std::sqrt(n) * tp.m3 / std::pow(tp.m2, 1.5), 1e-9));
  CHECK_THAT(s.population_kurtosis(),
             WithinRel(n * tp.m4 / (tp.m2 * tp.m2) - 3.0, 1e-9));
  CHECK_THAT(s.min(), WithinRel(*std::min_element(v.begin(), v.end()), 1e-15));
  CHECK_THAT(s.max(), WithinRel(*std::max_element(v.begin(), v.end()), 1e-15));
}

TEST_CASE("values far from zero do not lose precision", "[descriptive]") {
  // The textbook naive "sum of squares minus square of sum" fails here.
  RunningStats<double> s;
  for (double x : {1e9 + 4.0, 1e9 + 7.0, 1e9 + 13.0, 1e9 + 16.0}) s.push(x);
  CHECK_THAT(s.mean(), WithinRel(1e9 + 10.0, 1e-15));
  CHECK_THAT(s.variance(), WithinRel(30.0, 1e-9));
}

TEST_CASE("integral and single precision value types", "[descriptive]") {
  SECTION("int values accumulate in double") {
    RunningStats<int> s(5);
    s.push({1, 2, 3, 4, 5});
    static_assert(std::is_same_v<decltype(s)::stat_type, double>);
    CHECK_THAT(s.mean(), WithinRel(3.0, 1e-15));
    CHECK_THAT(s.variance(), WithinRel(2.5, 1e-15));
    CHECK(s.min() == 1);
    CHECK(s.max() == 5);
  }
  SECTION("float values also accumulate in double") {
    RunningStats<float> s;
    s.push({1.0f, 2.0f, 3.0f, 4.0f});
    static_assert(std::is_same_v<decltype(s)::stat_type, double>);
    CHECK_THAT(s.mean(), WithinRel(2.5, 1e-15));
  }
  SECTION("long double keeps its own precision") {
    RunningStats<long double> s;
    static_assert(std::is_same_v<decltype(s)::stat_type, long double>);
    s.push({1.0L, 2.0L, 3.0L});
    CHECK(s.mean() == 2.0L);
  }
  SECTION("the accumulator type can be chosen explicitly") {
    RunningStats<float, float> s;
    static_assert(std::is_same_v<decltype(s)::stat_type, float>);
    s.push({1.0f, 2.0f, 3.0f});
    CHECK_THAT(static_cast<double>(s.mean()), WithinRel(2.0, 1e-6));
  }
}

TEST_CASE("z-score", "[descriptive]") {
  RunningStats<double> s(ref::a.size(), ref::a);
  CHECK_THAT(s.zscore(ref::a_mean), WithinAbs(0.0, 1e-12));
  CHECK_THAT(s.zscore(ref::a_mean + ref::a_sd), WithinRel(1.0, 1e-12));
}
