// Special functions and distributions, checked against SciPy.
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <statisticalc/statisticalc.hpp>

#include "reference_data.hpp"

using namespace statisticalc;
using ref::WithinAbs;
using ref::WithinRel;

TEST_CASE("regularized incomplete gamma", "[special]") {
  CHECK_THAT(detail::gamma_p(3.0, 2.0), WithinRel(0.32332358381693654, 1e-13));
  CHECK_THAT(detail::gamma_p(0.5, 0.1), WithinRel(0.34527915398142317, 1e-13));
  CHECK_THAT(detail::gamma_p(20.0, 25.0), WithinRel(0.8664251659143496, 1e-13));
  CHECK_THAT(detail::gamma_q(20.0, 25.0),
             WithinRel(1.0 - 0.8664251659143496, 1e-11));

  SECTION("P + Q == 1") {
    for (double a : {0.5, 1.0, 3.5, 12.0, 80.0})
      for (double x : {0.01, 0.5, 2.0, 10.0, 100.0})
        CHECK_THAT(detail::gamma_p(a, x) + detail::gamma_q(a, x),
                   WithinAbs(1.0, 1e-13));
  }
  SECTION("edge cases") {
    CHECK(detail::gamma_p(2.0, 0.0) == 0.0);
    CHECK(detail::gamma_q(2.0, 0.0) == 1.0);
    CHECK(std::isnan(detail::gamma_p(-1.0, 1.0)));
  }
}

TEST_CASE("regularized incomplete beta", "[special]") {
  CHECK_THAT(detail::beta_i(2.0, 3.0, 0.4),
             WithinRel(0.5247999999999999, 1e-13));
  CHECK_THAT(detail::beta_i(0.5, 0.5, 0.25), WithinRel(1.0 / 3.0, 1e-12));

  SECTION("symmetry I_x(a,b) == 1 - I_(1-x)(b,a)") {
    for (double x : {0.05, 0.3, 0.5, 0.77, 0.99})
      CHECK_THAT(detail::beta_i(3.0, 7.0, x),
                 WithinAbs(1.0 - detail::beta_i(7.0, 3.0, 1.0 - x), 1e-13));
  }
  SECTION("edge cases") {
    CHECK(detail::beta_i(2.0, 2.0, 0.0) == 0.0);
    CHECK(detail::beta_i(2.0, 2.0, 1.0) == 1.0);
  }
}

TEST_CASE("standard normal distribution", "[dist]") {
  CHECK_THAT(dist::normal_cdf(-3.5), WithinRel(0.00023262907903552502, 1e-13));
  CHECK_THAT(dist::normal_cdf(-1.0), WithinRel(0.15865525393145707, 1e-13));
  CHECK_THAT(dist::normal_cdf(0.0), WithinRel(0.5, 1e-15));
  CHECK_THAT(dist::normal_cdf(0.5), WithinRel(0.6914624612740131, 1e-13));
  CHECK_THAT(dist::normal_sf(2.5), WithinRel(0.006209665325776132, 1e-13));

  CHECK_THAT(dist::normal_quantile(0.001),
             WithinRel(-3.090232306167813, 1e-12));
  CHECK_THAT(dist::normal_quantile(0.025),
             WithinRel(-1.9599639845400545, 1e-12));
  CHECK_THAT(dist::normal_quantile(0.5), WithinAbs(0.0, 1e-15));
  CHECK_THAT(dist::normal_quantile(0.975), WithinRel(1.959963984540054, 1e-12));
  CHECK_THAT(dist::normal_quantile(0.999), WithinRel(3.090232306167813, 1e-12));

  SECTION("quantile inverts the cdf") {
    for (double p : {1e-8, 0.01, 0.2, 0.5, 0.8, 0.99, 1 - 1e-8})
      CHECK_THAT(dist::normal_cdf(dist::normal_quantile(p)),
                 WithinRel(p, 1e-11));
  }
}

TEST_CASE("Student t distribution", "[dist]") {
  CHECK_THAT(dist::student_t_cdf(0.0, 1.0), WithinRel(0.5, 1e-14));
  CHECK_THAT(dist::student_t_cdf(1.0, 1.0), WithinRel(0.75, 1e-13));
  CHECK_THAT(dist::student_t_cdf(2.5, 9.0),
             WithinRel(0.9830690861585072, 1e-13));
  CHECK_THAT(dist::student_t_cdf(-2.5, 9.0),
             WithinRel(0.016930913841492867, 1e-13));
  CHECK_THAT(dist::student_t_cdf(1.96, 1000.0),
             WithinRel(0.9748634075221256, 1e-13));
  CHECK_THAT(dist::student_t_cdf(3.0, 2.0),
             WithinRel(0.9522670168666454, 1e-13));
  CHECK_THAT(dist::student_t_cdf(0.5, 4.5),
             WithinRel(0.6797252489629764, 1e-13));
  CHECK_THAT(dist::student_t_sf(2.5, 9.0),
             WithinRel(0.016930913841492867, 1e-13));

  CHECK_THAT(dist::student_t_quantile(0.975, 9.0),
             WithinRel(2.262157162798205, 1e-11));
  CHECK_THAT(dist::student_t_quantile(0.995, 4.0),
             WithinRel(4.604094871349992, 1e-11));
  CHECK_THAT(dist::student_t_quantile(0.05, 30.0),
             WithinRel(-1.6972608865939574, 1e-11));
  CHECK_THAT(dist::student_t_quantile(0.975, 1.0),
             WithinRel(12.706204736174694, 1e-11));

  SECTION("large dof converges to the normal") {
    CHECK_THAT(dist::student_t_cdf(1.5, 1e7),
               WithinAbs(dist::normal_cdf(1.5), 1e-6));
  }
}

TEST_CASE("chi-squared distribution", "[dist]") {
  CHECK_THAT(dist::chi_squared_cdf(0.5, 1.0),
             WithinRel(0.5204998778130466, 1e-13));
  CHECK_THAT(dist::chi_squared_cdf(3.0, 3.0),
             WithinRel(0.6083748237289109, 1e-13));
  CHECK_THAT(dist::chi_squared_cdf(11.976, 9.0),
             WithinRel(0.7853343046920512, 1e-13));
  CHECK_THAT(dist::chi_squared_sf(20.0, 9.0),
             WithinRel(0.017912404529843298, 1e-12));
  CHECK_THAT(dist::chi_squared_cdf(1.0, 10.0),
             WithinRel(0.00017211562995584072, 1e-12));

  CHECK_THAT(dist::chi_squared_quantile(0.025, 9.0),
             WithinRel(2.7003894999803584, 1e-11));
  CHECK_THAT(dist::chi_squared_quantile(0.975, 9.0),
             WithinRel(19.02276779864163, 1e-11));
  CHECK_THAT(dist::chi_squared_quantile(0.5, 4.0),
             WithinRel(3.3566939800333224, 1e-11));
  CHECK_THAT(dist::chi_squared_quantile(0.99, 1.0),
             WithinRel(6.6348966010212145, 1e-11));

  SECTION("quantile inverts the cdf") {
    for (double k : {1.0, 2.5, 9.0, 40.0})
      for (double p : {0.001, 0.1, 0.5, 0.9, 0.999})
        CHECK_THAT(dist::chi_squared_cdf(dist::chi_squared_quantile(p, k), k),
                   WithinRel(p, 1e-10));
  }
}

TEST_CASE("Fisher F distribution", "[dist]") {
  CHECK_THAT(dist::fisher_f_cdf(1.0, 5.0, 5.0), WithinRel(0.5, 1e-13));
  CHECK_THAT(dist::fisher_f_cdf(1.6526910048927361, 9.0, 11.0),
             WithinRel(0.786619411063544, 1e-13));
  CHECK_THAT(dist::fisher_f_cdf(3.0, 2.0, 10.0),
             WithinRel(0.904632568359375, 1e-13));
  CHECK_THAT(dist::fisher_f_cdf(0.25, 4.0, 8.0),
             WithinRel(0.0982404443767041, 1e-13));
  CHECK_THAT(dist::fisher_f_sf(0.25, 4.0, 8.0),
             WithinRel(0.9017595556232959, 1e-13));

  CHECK_THAT(dist::fisher_f_quantile(0.975, 9.0, 11.0),
             WithinRel(3.587898669106546, 1e-11));
  CHECK_THAT(dist::fisher_f_quantile(0.025, 9.0, 11.0),
             WithinRel(0.25561885602307155, 1e-11));
  CHECK_THAT(dist::fisher_f_quantile(0.95, 3.0, 20.0),
             WithinRel(3.098391212140781, 1e-11));

  SECTION("reciprocal property F(p, d1, d2) == 1 / F(1-p, d2, d1)") {
    CHECK_THAT(
        dist::fisher_f_quantile(0.025, 9.0, 11.0),
        WithinRel(1.0 / dist::fisher_f_quantile(0.975, 11.0, 9.0), 1e-10));
  }
}
