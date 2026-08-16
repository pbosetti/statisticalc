// Two-sample tests obtained by comparing two instances, checked against SciPy.
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <statisticalc/statisticalc.hpp>

#include "reference_data.hpp"

using namespace statisticalc;
using ref::WithinAbs;
using ref::WithinRel;

namespace {
RunningStats<double> sample_a() {
  return RunningStats<double>(ref::a.size(), ref::a);
}
RunningStats<double> sample_b() {
  return RunningStats<double>(ref::b.size(), ref::b);
}
}  // namespace

TEST_CASE("Welch two-sample t-test", "[two-sample]") {
  const auto a = sample_a();
  const auto b = sample_b();

  const auto r = t_test(a, b);  // found through ADL, Welch by default
  CHECK(r.name == "Welch two-sample t-test");
  CHECK_THAT(r.statistic, WithinRel(-5.222822518637561, 1e-10));
  CHECK_THAT(r.dof1, WithinRel(16.857609814066162, 1e-10));
  CHECK_THAT(r.p_value, WithinRel(7.081183795661777e-05, 1e-9));
  CHECK_THAT(r.estimate, WithinRel(ref::a_mean - ref::b_mean, 1e-12));
  CHECK_THAT(r.conf_int.lower, WithinRel(-3.2811961254332664, 1e-9));
  CHECK_THAT(r.conf_int.upper, WithinRel(-1.3921372079000647, 1e-9));
  CHECK(r.reject());
  CHECK_FALSE(r.conf_int.contains(0.0));

  SECTION("one-sided") {
    const auto lo = t_test(a, b, 0.0, {.alternative = Alternative::less});
    CHECK_THAT(lo.p_value, WithinRel(3.540591897830889e-05, 1e-9));
    CHECK(lo.conf_int.lower == -std::numeric_limits<double>::infinity());
  }
  SECTION("a non-zero null difference") {
    const auto r2 =
        t_test(a, b, -2.3366666666666664);  // exactly the observed gap
    CHECK_THAT(r2.statistic, WithinAbs(0.0, 1e-10));
    CHECK_THAT(r2.p_value, WithinRel(1.0, 1e-12));
  }
  SECTION("swapping the samples flips the sign") {
    const auto rev = t_test(b, a);
    CHECK_THAT(rev.statistic, WithinRel(-r.statistic, 1e-12));
    CHECK_THAT(rev.p_value, WithinRel(r.p_value, 1e-12));
  }
}

TEST_CASE("pooled two-sample t-test", "[two-sample]") {
  const auto a = sample_a();
  const auto b = sample_b();

  const auto r = t_test(a, b, 0.0, {.equal_variance = true});
  CHECK(r.name == "Two-sample t-test (pooled variance)");
  CHECK_THAT(r.statistic, WithinRel(-5.347103754833798, 1e-10));
  CHECK(r.dof1 == 20.0);
  CHECK_THAT(r.p_value, WithinRel(3.11117698969747e-05, 1e-9));
  CHECK_THAT(r.conf_int.lower, WithinRel(-3.248225798225491, 1e-9));
  CHECK_THAT(r.conf_int.upper, WithinRel(-1.42510753510784, 1e-9));

  CHECK_THAT(pooled_variance(a, b), WithinRel(1.041633333333333, 1e-11));
  CHECK_THAT(cohens_d(a, b), WithinRel(-2.2894930451031246, 1e-10));
}

TEST_CASE("F test on the ratio of the variances", "[two-sample]") {
  const auto a = sample_a();
  const auto b = sample_b();

  const auto r = f_test(a, b);
  CHECK_THAT(r.statistic, WithinRel(1.6526910048927361, 1e-11));
  CHECK(r.dof1 == 9.0);
  CHECK(r.dof2 == 11.0);
  CHECK_THAT(r.p_value, WithinRel(0.42676117787291185, 1e-10));
  CHECK_THAT(r.estimate, WithinRel(ref::a_var / ref::b_var, 1e-11));
  CHECK_THAT(r.conf_int.lower, WithinRel(0.4606292310101074, 1e-9));
  CHECK_THAT(r.conf_int.upper, WithinRel(6.465450282523634, 1e-9));
  CHECK(r.conf_int.contains(1.0));
  CHECK_FALSE(r.reject());

  SECTION("one-sided") {
    CHECK_THAT(f_test(a, b, 1.0, {.alternative = Alternative::less}).p_value,
               WithinRel(0.786619411063544, 1e-10));
    CHECK_THAT(f_test(a, b, 1.0, {.alternative = Alternative::greater}).p_value,
               WithinRel(0.21338058893645592, 1e-10));
  }
  SECTION("a large variance ratio is detected") {
    RunningStats<double> wide;
    for (double x : ref::a) wide.push(10.0 * x);  // variance x100
    CHECK(f_test(wide, a).reject());
    CHECK_THAT(f_test(wide, a).statistic, WithinRel(100.0, 1e-9));
  }
}

TEST_CASE("convenience predicates", "[two-sample]") {
  const auto a = sample_a();
  const auto b = sample_b();

  CHECK_FALSE(same_mean(a, b));  // the means clearly differ
  CHECK(same_variance(a, b));    // the variances do not
  CHECK(same_mean(a, a));
  CHECK(same_variance(a, a));

  SECTION("shifting a sample leaves the variance untouched") {
    RunningStats<double> shifted;
    for (double x : ref::a) shifted.push(x + 100.0);
    CHECK_FALSE(same_mean(a, shifted));
    CHECK(same_variance(a, shifted));
    CHECK_THAT(f_test(a, shifted).statistic, WithinRel(1.0, 1e-9));
  }
  SECTION("the significance level can be tightened") {
    // p = 3.1e-5, so the pooled test rejects at 1e-4 but not at 1e-6.
    CHECK_FALSE(same_mean(a, b, 1e-4, {.equal_variance = true}));
    CHECK(same_mean(a, b, 1e-6, {.equal_variance = true}));
  }
}

TEST_CASE("two-sample tests between different value types", "[two-sample]") {
  // Different instantiations are compared through their summaries.
  RunningStats<int> ints(8);
  ints.push({10, 12, 11, 13, 12, 14, 11, 12});
  RunningStats<float> floats;
  for (double x : ref::b) floats.push(static_cast<float>(x));

  const auto r = t_test(ints.summary(), floats.summary());
  CHECK(r.estimate > 0.0);
  CHECK(r.reject());
  CHECK_THAT(f_test(ints.summary(), floats.summary()).statistic,
             WithinRel(ints.variance() / floats.variance(), 1e-9));
}

TEST_CASE("two-sample tests degrade gracefully", "[two-sample]") {
  RunningStats<double> one;
  one.push(1.0);
  const auto a = sample_a();
  CHECK(std::isnan(t_test(a, one).p_value));
  CHECK(std::isnan(f_test(a, one).p_value));
  CHECK(std::isnan(pooled_variance(a, one)));
}
