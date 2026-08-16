// Shared fixtures for the test suite.
//
// The expected values used throughout the tests were produced independently
// with SciPy 1.x (scipy.stats), and are quoted here with full double precision.
#pragma once

#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

namespace ref {

/// Sample A, n = 10.
inline const std::vector<double> a{2.1, 3.4, 1.9, 4.8, 3.3,
                                   2.7, 5.1, 3.9, 4.4, 2.2};
/// Sample B, n = 12.
inline const std::vector<double> b{5.2, 4.1, 6.3, 5.9, 4.8, 6.6,
                                   5.4, 7.1, 4.9, 6.0, 5.5, 6.8};

// Descriptive statistics of A.
inline constexpr double a_mean = 3.3800000000000003;
inline constexpr double a_var = 1.3306666666666664;     // ddof = 1
inline constexpr double a_popvar = 1.1975999999999998;  // ddof = 0
inline constexpr double a_sd = 1.15354525991253;
inline constexpr double a_sem = 0.364783040541452;
inline constexpr double a_skew = 0.16799304013870545;     // sample (bias free)
inline constexpr double a_popskew = 0.14166417010517282;  // population
inline constexpr double a_kurt = -1.4293063295213395;     // sample excess
inline constexpr double a_popkurt = -1.3539510550827778;  // population excess
inline constexpr double a_median = 3.3499999999999996;
inline constexpr double a_q25 = 2.325;
inline constexpr double a_q75 = 4.275;
inline constexpr double a_rms = 3.5527454172794313;

// Descriptive statistics of B.
inline constexpr double b_mean = 5.716666666666666;
inline constexpr double b_var = 0.8051515151515151;
inline constexpr double b_sd = 0.897302354366417;
inline constexpr double b_skew = -0.13361674290364448;
inline constexpr double b_kurt = -0.6699931398855661;
inline constexpr double b_median = 5.7;

/// Relative tolerance used for the comparisons against the reference values.
inline constexpr double tol = 1e-11;

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

}  // namespace ref
