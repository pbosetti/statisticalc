// Behaviour of the running window itself: eviction, removal recursion, extrema.
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

double naive_mean(const std::vector<double> &v) {
  return std::accumulate(v.begin(), v.end(), 0.0) /
         static_cast<double>(v.size());
}

double naive_variance(const std::vector<double> &v) {
  const double m = naive_mean(v);
  double acc = 0.0;
  for (double x : v) acc += (x - m) * (x - m);
  return acc / (static_cast<double>(v.size()) - 1.0);
}

}  // namespace

TEST_CASE("the window keeps only the most recent values", "[window]") {
  RunningStats<double> s(3);
  CHECK(s.capacity() == 3);
  CHECK(s.bounded());

  s.push({1.0, 2.0, 3.0});
  CHECK(s.size() == 3);
  CHECK(s.full());
  CHECK_THAT(s.mean(), WithinRel(2.0, 1e-15));

  s.push(10.0);  // evicts 1.0
  CHECK(s.size() == 3);
  CHECK(s.total_count() == 4);
  CHECK(s.values() == std::vector<double>{2.0, 3.0, 10.0});
  CHECK_THAT(s.mean(), WithinRel(5.0, 1e-14));
  CHECK_THAT(s.variance(), WithinRel(19.0, 1e-13));
  CHECK(s.oldest() == 2.0);
  CHECK(s.newest() == 10.0);
  CHECK(s[1] == 3.0);
  CHECK_THROWS_AS(s.at(3), std::out_of_range);
}

TEST_CASE("a sliding window matches a recomputation at every step",
          "[window]") {
  constexpr std::size_t w = 17;
  std::mt19937 rng(4242);
  std::normal_distribution<double> dis(5.0, 2.0);

  RunningStats<double> s(w);
  std::vector<double> window;
  for (int i = 0; i < 400; ++i) {
    const double x = dis(rng);
    s.push(x);
    window.push_back(x);
    if (window.size() > w) window.erase(window.begin());

    REQUIRE(s.size() == window.size());
    REQUIRE_THAT(s.mean(), WithinRel(naive_mean(window), 1e-10));
    if (window.size() > 1)
      REQUIRE_THAT(s.variance(), WithinRel(naive_variance(window), 1e-9));
    REQUIRE(s.min() == *std::min_element(window.begin(), window.end()));
    REQUIRE(s.max() == *std::max_element(window.begin(), window.end()));
  }
}

TEST_CASE("higher moments survive insertion and removal", "[window]") {
  constexpr std::size_t w = 25;
  std::mt19937 rng(7);
  std::gamma_distribution<double> dis(2.0, 1.5);

  RunningStats<double> rolling(w);
  std::vector<double> window;
  for (int i = 0; i < 300; ++i) {
    const double x = dis(rng);
    rolling.push(x);
    window.push_back(x);
    if (window.size() > w) window.erase(window.begin());
  }
  // A fresh accumulator fed with the same window must agree.
  RunningStats<double> fresh(w, window);
  CHECK_THAT(rolling.mean(), WithinRel(fresh.mean(), 1e-10));
  CHECK_THAT(rolling.variance(), WithinRel(fresh.variance(), 1e-9));
  CHECK_THAT(rolling.skewness(), WithinRel(fresh.skewness(), 1e-8));
  CHECK_THAT(rolling.kurtosis(), WithinRel(fresh.kurtosis(), 1e-8));
}

TEST_CASE("extrema of a sliding window with repeated values", "[window]") {
  RunningStats<int> s(4);
  const std::vector<int> data{5, 5, 3, 3, 7, 7, 1, 9, 9, 2, 2, 2};
  std::vector<int> window;
  for (int x : data) {
    s.push(x);
    window.push_back(x);
    if (window.size() > 4) window.erase(window.begin());
    REQUIRE(s.min() == *std::min_element(window.begin(), window.end()));
    REQUIRE(s.max() == *std::max_element(window.begin(), window.end()));
  }
}

TEST_CASE("explicit pop", "[window]") {
  RunningStats<double> s(5, ref::a);  // only the last 5 values fit
  CHECK(s.size() == 5);
  CHECK(s.values() == std::vector<double>{2.7, 5.1, 3.9, 4.4, 2.2});

  s.pop();
  CHECK(s.size() == 4);
  CHECK(s.oldest() == 5.1);
  CHECK_THAT(s.mean(), WithinRel(naive_mean({5.1, 3.9, 4.4, 2.2}), 1e-13));

  while (!s.empty()) s.pop();
  CHECK(s.empty());
  CHECK(std::isnan(s.mean()));
  CHECK(s.sum_of_squares() == 0.0);
  CHECK_NOTHROW(s.pop());  // popping an empty window is a no-op
}

TEST_CASE("clear and refresh", "[window]") {
  RunningStats<double> s(10, ref::a);
  s.clear();
  CHECK(s.empty());
  CHECK(s.capacity() == 10);
  CHECK(s.total_count() == 0);

  s.push(ref::a);
  const double before = s.variance();
  s.refresh();  // recomputes from the stored values
  CHECK_THAT(s.variance(), WithinRel(before, 1e-12));
  CHECK_THAT(s.variance(), WithinRel(ref::a_var, ref::tol));
  CHECK(s.total_count() == 10);

  SECTION("the window keeps sliding correctly after a refresh") {
    RunningStats<double> r(4);
    std::vector<double> window;
    for (int i = 0; i < 40; ++i) {
      const double x = std::sin(0.7 * i) * 10.0;  // no monotonic trend
      r.push(x);
      window.push_back(x);
      if (window.size() > 4) window.erase(window.begin());
      if (i % 5 == 0) r.refresh();  // must not disturb the eviction bookkeeping
      REQUIRE(r.size() == window.size());
      REQUIRE(r.min() == *std::min_element(window.begin(), window.end()));
      REQUIRE(r.max() == *std::max_element(window.begin(), window.end()));
      REQUIRE_THAT(r.mean(), WithinRel(naive_mean(window), 1e-12));
    }
    CHECK(r.total_count() == 40);
  }
}

TEST_CASE("resizing the window", "[window]") {
  RunningStats<double> s(10, ref::a);

  SECTION("shrinking drops the oldest values") {
    s.resize(3);
    CHECK(s.capacity() == 3);
    CHECK(s.values() == std::vector<double>{3.9, 4.4, 2.2});
    CHECK_THAT(s.mean(), WithinRel(naive_mean({3.9, 4.4, 2.2}), 1e-13));
  }
  SECTION("growing keeps everything and accepts more") {
    s.resize(12);
    CHECK(s.size() == 10);
    CHECK(!s.full());
    s.push(9.0);
    CHECK(s.size() == 11);
    CHECK(s.newest() == 9.0);
    CHECK(s.oldest() == ref::a.front());
  }
  SECTION("switching to unlimited keeps the moments but not the values") {
    const double m = s.mean();
    s.resize(RunningStats<double>::unlimited);
    CHECK(!s.bounded());
    CHECK_THAT(s.mean(), WithinRel(m, 1e-15));
    CHECK(s.min() == 1.9);
    CHECK_THROWS_AS(s.values(), std::logic_error);
  }
}

TEST_CASE("unlimited accumulator", "[window]") {
  RunningStats<double> s;
  CHECK(!s.bounded());
  CHECK(s.capacity() == RunningStats<double>::unlimited);
  CHECK(!s.full());

  for (int i = 1; i <= 1000; ++i) s.push(static_cast<double>(i));
  CHECK(s.size() == 1000);
  CHECK_THAT(s.mean(), WithinRel(500.5, 1e-13));
  CHECK_THAT(s.variance(), WithinRel(1000.0 * 1001.0 / 12.0, 1e-11));
  CHECK(s.min() == 1.0);
  CHECK(s.max() == 1000.0);

  SECTION("value based queries are rejected") {
    CHECK_THROWS_AS(s.values(), std::logic_error);
    CHECK_THROWS_AS(s.median(), std::logic_error);
    CHECK_THROWS_AS(s.at(0), std::logic_error);
    CHECK_THROWS_AS(s.pop(), std::logic_error);
  }
  SECTION("it cannot be turned into a bounded window") {
    CHECK_THROWS_AS(s.resize(10), std::logic_error);
    s.clear();
    CHECK_NOTHROW(s.resize(10));
    CHECK(s.bounded());
  }
}
