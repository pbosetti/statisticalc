// statisticalc - header-only running statistics for C++20
//
// Umbrella header: include this one to get the whole library.
//
//   #include <statisticalc/statisticalc.hpp>
//
//   statisticalc::RunningStats<double> s(100);   // window of 100 values
//   s << 1.0 << 2.0 << 3.0;
//   auto m = s.mean();
//   auto r = s.t_test(0.0);                      // H0: mu == 0
#pragma once

#define STATISTICALC_VERSION_MAJOR 1
#define STATISTICALC_VERSION_MINOR 0
#define STATISTICALC_VERSION_PATCH 0
#define STATISTICALC_VERSION_STRING "1.0.0"

#if defined(_MSVC_LANG)
#define STATISTICALC_CPLUSPLUS _MSVC_LANG
#else
#define STATISTICALC_CPLUSPLUS __cplusplus
#endif

#if STATISTICALC_CPLUSPLUS < 202002L
#error "statisticalc requires C++20 (use /std:c++20 on MSVC)"
#endif

#include "detail/special_functions.hpp"
#include "distributions.hpp"
#include "running_stats.hpp"
