// One-sample inferential statistics, checked against SciPy.
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <random>
#include <statisticalc/statisticalc.hpp>

#include "reference_data.hpp"

using namespace statisticalc;
using ref::WithinAbs;
using ref::WithinRel;

TEST_CASE("one-sample t-test", "[inference]") {
  const RunningStats<double> s(ref::a.size(), ref::a);

  SECTION("two-sided") {
    const auto r = s.t_test(3.0);
    CHECK(r.dof1 == 9.0);
    CHECK_THAT(r.statistic, WithinRel(1.0417150957346086, ref::tol));
    CHECK_THAT(r.p_value, WithinRel(0.3247174788270432, 1e-10));
    CHECK_THAT(r.estimate, WithinRel(ref::a_mean, ref::tol));
    CHECK(r.null_value == 3.0);
    CHECK_THAT(r.conf_int.lower, WithinRel(2.5548034319718465, 1e-10));
    CHECK_THAT(r.conf_int.upper, WithinRel(4.205196568028154, 1e-10));
    CHECK(r.conf_int.contains(ref::a_mean));
    CHECK_FALSE(r.reject(0.05));
    CHECK_FALSE(static_cast<bool>(r));
  }
  SECTION("confidence level") {
    const auto r = s.t_test(3.0, {.conf_level = 0.99});
    CHECK_THAT(r.conf_int.lower, WithinRel(2.194515109878348, 1e-10));
    CHECK_THAT(r.conf_int.upper, WithinRel(4.5654848901216525, 1e-10));
    CHECK_THAT(s.mean_ci(0.95).lower, WithinRel(2.5548034319718465, 1e-10));
    CHECK_THAT(s.mean_ci(0.95).upper, WithinRel(4.205196568028154, 1e-10));
  }
  SECTION("one-sided alternatives") {
    const auto lo = s.t_test(3.0, {.alternative = Alternative::less});
    CHECK_THAT(lo.p_value, WithinRel(0.8376412605864784, 1e-10));
    CHECK(lo.conf_int.lower == -std::numeric_limits<double>::infinity());
    CHECK_THAT(lo.conf_int.upper, WithinRel(4.0486885092302005, 1e-10));

    const auto hi = s.t_test(3.0, {.alternative = Alternative::greater});
    CHECK_THAT(hi.p_value, WithinRel(0.1623587394135216, 1e-10));
    CHECK_THAT(hi.conf_int.lower, WithinRel(2.7113114907698, 1e-10));
    CHECK(hi.conf_int.upper == std::numeric_limits<double>::infinity());

    CHECK_THAT(lo.p_value + hi.p_value, WithinAbs(1.0, 1e-12));
  }
  SECTION("a clearly displaced mean is detected") {
    const auto r = s.t_test(0.0);
    CHECK(r.p_value < 1e-5);
    CHECK(r.reject());
    CHECK(static_cast<bool>(r));
  }
  SECTION("too few values give a NaN result rather than throwing") {
    RunningStats<double> tiny;
    tiny.push(1.0);
    const auto r = tiny.t_test(0.0);
    CHECK(std::isnan(r.statistic));
    CHECK(std::isnan(r.p_value));
  }
}

TEST_CASE("one-sample variance test", "[inference]") {
  const RunningStats<double> s(ref::a.size(), ref::a);

  SECTION("two-sided, sigma^2 = 1") {
    const auto r = s.variance_test(1.0);
    CHECK(r.dof1 == 9.0);
    CHECK_THAT(r.statistic, WithinRel(11.975999999999997, ref::tol));
    CHECK_THAT(r.p_value, WithinRel(0.4293313906158977, 1e-10));
    CHECK_THAT(r.estimate, WithinRel(ref::a_var, ref::tol));
    CHECK_FALSE(r.reject());
  }
  SECTION("two-sided, sigma^2 = 1.3") {
    const auto r = s.variance_test(1.3);
    CHECK_THAT(r.statistic, WithinRel(9.21230769230769, ref::tol));
    CHECK_THAT(r.p_value, WithinRel(0.8358235738210881, 1e-10));
  }
  SECTION("one-sided alternatives") {
    CHECK_THAT(s.variance_test(1.0, {.alternative = Alternative::less}).p_value,
               WithinRel(0.7853343046920511, 1e-10));
    CHECK_THAT(
        s.variance_test(1.0, {.alternative = Alternative::greater}).p_value,
        WithinRel(0.21466569530794885, 1e-10));
  }
  SECTION("confidence intervals") {
    const auto ci = s.variance_ci(0.95);
    CHECK_THAT(ci.lower, WithinRel(0.6295613828002029, 1e-10));
    CHECK_THAT(ci.upper, WithinRel(4.434915777922817, 1e-10));
    CHECK(ci.contains(ref::a_var));

    const auto sd = s.stddev_ci(0.95);
    CHECK_THAT(sd.lower, WithinRel(std::sqrt(0.6295613828002029), 1e-10));
    CHECK_THAT(sd.upper, WithinRel(std::sqrt(4.434915777922817), 1e-10));
    CHECK(sd.contains(ref::a_sd));
  }
  SECTION("a variance far from the null value is detected") {
    const auto r = s.variance_test(0.01);
    CHECK(r.p_value < 1e-6);
    CHECK(r.reject());
  }
  SECTION("an invalid null variance gives NaN") {
    CHECK(std::isnan(s.variance_test(0.0).p_value));
  }
}

TEST_CASE("tests can also be run on a Summary", "[inference]") {
  const RunningStats<double> s(ref::a.size(), ref::a);
  const Summary<double> sum = s.summary();

  CHECK(sum.n == 10);
  CHECK_THAT(sum.mean, WithinRel(ref::a_mean, ref::tol));
  CHECK_THAT(sum.variance, WithinRel(ref::a_var, ref::tol));
  CHECK_THAT(sum.stddev(), WithinRel(ref::a_sd, ref::tol));
  CHECK_THAT(sum.sem(), WithinRel(ref::a_sem, ref::tol));
  CHECK(sum.minimum == 1.9);
  CHECK(sum.maximum == 5.1);

  CHECK_THAT(t_test(sum, 3.0).p_value, WithinRel(0.3247174788270432, 1e-10));
  CHECK_THAT(variance_test(sum, 1.0).p_value,
             WithinRel(0.4293313906158977, 1e-10));

  SECTION("hand made summaries need no accumulator at all") {
    const Summary<double> made{.n = 25, .mean = 10.4, .variance = 4.0};
    const auto r = t_test(made, 10.0);
    CHECK_THAT(r.statistic, WithinRel(0.4 / (2.0 / 5.0), 1e-12));
    CHECK(r.dof1 == 24.0);
  }
}

TEST_CASE("the coverage of the confidence interval is nominal", "[inference]") {
  // 95% intervals built on normal samples must miss the true mean about 5% of
  // the time: a loose but genuinely end-to-end check of quantiles and moments.
  std::mt19937 rng(20260101);
  std::normal_distribution<double> dis(2.0, 3.0);
  int covered = 0;
  constexpr int trials = 2000;
  for (int i = 0; i < trials; ++i) {
    RunningStats<double> s(12);
    for (int j = 0; j < 12; ++j) s.push(dis(rng));
    if (s.mean_ci(0.95).contains(2.0)) ++covered;
  }
  const double rate = static_cast<double>(covered) / trials;
  CHECK(rate > 0.93);
  CHECK(rate < 0.97);
}
