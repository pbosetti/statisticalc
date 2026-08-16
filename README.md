# statisticalc

Header-only C++20 running statistics: a rolling (or unlimited) window of
observations with **descriptive** statistics maintained by recursion formulas
and **inferential** statistics (t-test, variance test, and their two-sample
counterparts) computed on top of them.

* header only, no dependency but the standard library;
* the window holds the last *N* values, or every value if unbounded;
* mean, variance, skewness and kurtosis are updated in **O(1)** per sample, on
  insertion *and* on eviction (Welford / Pébay recursions, both directions);
* min and max of the sliding window in O(1) amortised (monotonic deques);
* Student *t*, chi-squared and Fisher *F* distributions implemented from
  scratch, so p-values and confidence intervals need no external library;
* two instances are compared directly: `t_test(a, b)`, `f_test(a, b)`,
  `same_mean(a, b)`, `a + b`;
* tested against SciPy reference values with [Catch2](https://github.com/catchorg/Catch2),
  fetched by `FetchContent`;
* builds on Linux, macOS and Windows (GCC, Clang, AppleClang, MSVC).

## Quick start

```cpp
#include <statisticalc/statisticalc.hpp>
using statisticalc::RunningStats;

RunningStats<double> window(50);   // last 50 observations
RunningStats<double> lifetime;     // unlimited, O(1) memory

window << 10.2 << 9.8 << 10.1;     // stream values in
window.push(some_vector);          // or push any range

double m  = window.mean();
double sd = window.stddev();
double q  = window.median();       // order statistics need a bounded window

// H0: the process is centred on 10.0
auto r = window.t_test(10.0);
if (r.reject(0.05)) std::cout << r;          // prints a full report

// H0: the standard deviation is still 0.2, against a one-sided alternative
auto v = window.variance_test(0.2 * 0.2, {.alternative = statisticalc::Alternative::greater});

// Two samples: Welch t-test on the means, F test on the variances
RunningStats<double> batch1(30), batch2(30);
// ...
auto tt = t_test(batch1, batch2);            // Welch by default
auto ff = f_test(batch1, batch2);
bool ok = same_mean(batch1, batch2) && same_variance(batch1, batch2);
```

See [`examples/example.cpp`](examples/example.cpp) for a complete tour.

## Using it in your project

### FetchContent (recommended)

```cmake
include(FetchContent)
FetchContent_Declare(statisticalc
  GIT_REPOSITORY https://github.com/pbosetti/statisticalc.git
  GIT_TAG main)
FetchContent_MakeAvailable(statisticalc)

target_link_libraries(my_target PRIVATE statisticalc::statisticalc)
```

### Installed package

```bash
cmake -S . -B build -DSTATISTICALC_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build --target install
```

```cmake
find_package(statisticalc REQUIRED)
target_link_libraries(my_target PRIVATE statisticalc::statisticalc)
```

### Plain copy

Add `include/` to your include path; nothing else is needed. The library
requires C++20 (`/std:c++20 /Zc:__cplusplus` on MSVC, which the CMake target
sets for you).

## The class

```cpp
template <typename T = double, std::floating_point A = default_accumulator_t<T>>
  requires std::is_arithmetic_v<T>
class RunningStats;
```

`T` is the type of the observations (any arithmetic type: `double`, `float`,
`int`, ...), `A` the type used for the accumulators. By default `A` is `double`,
except for `long double` values, which keep their own precision. `A` can be
chosen explicitly, e.g. `RunningStats<float, float>` for embedded targets.

### Feeding and window management

| member | meaning |
| --- | --- |
| `RunningStats()` | unlimited accumulator, values are not retained |
| `RunningStats(n)` | rolling window of at most `n` values |
| `RunningStats(n, range)` | as above, pre-filled from a range |
| `push(x)`, `push(range)`, `push({...})`, `operator()(x)` | add observations |
| `pop()` | drop the oldest observation (bounded windows) |
| `clear()`, `resize(n)`, `refresh()` | reset, change the window, recompute the moments |
| `size()`, `capacity()`, `empty()`, `full()`, `bounded()`, `total_count()` | state |
| `at(i)`, `operator[]`, `oldest()`, `newest()`, `values()` | access the stored values |

`refresh()` recomputes the moments from the stored values: the incremental
formulas are numerically sound but, over hundreds of millions of updates,
round-off does accumulate, and a periodic `refresh()` clears it.

### Descriptive statistics

`mean()`, `sum()`, `variance()`, `population_variance()`, `stddev()`,
`population_stddev()`, `sem()`, `cv()`, `rms()`, `sum_of_squares()`,
`skewness()`, `population_skewness()`, `kurtosis()` (excess),
`population_kurtosis()`, `min()`, `max()`, `range()`, `zscore(x)`,
and — on bounded windows, since they need the values — `median()`,
`quantile(p)` (type 7, as in R and NumPy) and `iqr()`.

`variance()`, `skewness()` and `kurtosis()` are the usual bias-corrected sample
estimators; the `population_*` variants use the plain moment ratios. Statistics
that need more data than available return `NaN` (variance below 2 values,
skewness below 3, kurtosis below 4).

### Inferential statistics

| call | test |
| --- | --- |
| `s.t_test(mu0, opts)` | one-sample Student t-test, H0: μ == mu0 |
| `s.variance_test(sigma2, opts)` | one-sample chi-squared test, H0: σ² == sigma2 |
| `s.mean_ci(level)`, `s.variance_ci(level)`, `s.stddev_ci(level)` | confidence intervals |
| `t_test(a, b, diff0, opts)` | two-sample t-test, Welch or pooled |
| `f_test(a, b, ratio0, opts)` | F test on the ratio of the variances |
| `same_mean(a, b, alpha, opts)`, `same_variance(a, b, alpha, opts)` | predicates |
| `pooled_variance(a, b)`, `cohens_d(a, b)` | supporting quantities |

Options are passed with a designated initializer:

```cpp
s.t_test(10.0, {.alternative = Alternative::greater, .conf_level = 0.99});
t_test(a, b, 0.0, {.equal_variance = true});     // pooled instead of Welch
```

Every test returns a `TestResult<A>` carrying `statistic`, `dof1`, `dof2`,
`p_value`, `estimate`, `null_value`, `conf_int`, `conf_level`, a `reject(alpha)`
predicate (also reachable through `explicit operator bool`) and a stream
operator that prints an R-like report. The confidence interval follows the
chosen alternative: two-sided for `two_sided`, one-sided otherwise.

Tests can also be run on a `Summary<A>` — the lightweight
`{n, mean, variance, minimum, maximum}` snapshot returned by `summary()`. This
is what makes comparisons between different instantiations possible:

```cpp
RunningStats<int>   counts(100);
RunningStats<float> volts;
auto r = t_test(counts.summary(), volts.summary());
```

### Friend interface

```cpp
s << 1.0 << 2.0;                 // stream values in
s += 3.0;  s += some_vector;     // accumulate values or ranges
std::cout << s;                  // RunningStats[n=3/50, mean=2, sd=1, min=1, max=3]
auto c = a + b;                  // or merge(a, b): pairwise combination of the moments
swap(a, b);
```

`merge()` combines the moments of two samples with the Chan-Golub-LeVeque
formulas; the result is an unlimited accumulator (the individual values are not
carried over).

### Distributions

The distributions used by the tests are public, in namespace
`statisticalc::dist`, each with `_cdf`, `_sf` (upper tail, computed directly so
it stays accurate in the far tail) and `_quantile`:
`normal_*`, `student_t_*`, `chi_squared_*`, `fisher_f_*`. They are built on the
regularized incomplete gamma and beta functions in `statisticalc::detail`.

## Building and testing

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

or, with the provided presets:

```bash
cmake --preset linux      # also: linux-clang, macos, windows, debug-sanitized
cmake --build --preset linux
ctest --preset linux
```

Catch2 v3 is downloaded automatically by `FetchContent` (nothing else is
needed, and no dependency is pulled in when the library is merely consumed).
Options: `STATISTICALC_BUILD_TESTS`, `STATISTICALC_BUILD_EXAMPLES`,
`STATISTICALC_INSTALL`, all defaulting to `ON` for a top-level build and `OFF`
when the project is added as a subproject.

The suite covers the special functions and distributions, the descriptive
statistics (against two-pass recomputations and against SciPy values), the
sliding window (against a recomputation at every step), the one- and two-sample
tests, and the friend interface. Continuous integration builds and runs it on
Linux (GCC and Clang), macOS and Windows (MSVC), plus an ASan/UBSan run and an
install-and-consume check.

## Implementation notes

Central moments are updated with the recursion of Welford, extended by Pébay to
the third and fourth order. The removal step is the exact algebraic inverse of
the insertion step, which is what allows a *sliding* window without recomputing
anything:

```
insertion (n is the new count)      removal (n is the count before removal)
delta   = x - mean                  delta_n = (x - mean) / (n - 1)
delta_n = delta / n                 delta   = n * delta_n
term    = delta * delta_n * (n-1)   term    = delta * delta_n * (n - 1)
mean   += delta_n                   M2 -= term  (then M3, M4, then mean)
M4     += ...                       M4 -= ...
M3     += ...                       M3 -= ...
M2     += term                      mean -= delta_n
```

This keeps the computation in a single pass and avoids the catastrophic
cancellation of the naive "sum of squares minus square of the sum" formula (see
the test on values around 1e9).

## License

Apache License 2.0, see [LICENSE](LICENSE).
