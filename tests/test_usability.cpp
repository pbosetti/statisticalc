// The friend interface: streaming, accumulation, merging, printing.
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <functional>
#include <numeric>
#include <ranges>
#include <sstream>
#include <statisticalc/statisticalc.hpp>
#include <string>
#include <vector>

#include "reference_data.hpp"

using namespace statisticalc;
using ref::WithinAbs;
using ref::WithinRel;

TEST_CASE("values can be streamed in", "[usability]") {
  RunningStats<double> s(3);
  s << 1.0 << 2.0 << 3.0 << 4.0;  // the first value is evicted
  CHECK(s.values() == std::vector<double>{2.0, 3.0, 4.0});
  CHECK_THAT(s.mean(), WithinRel(3.0, 1e-14));

  s += 5.0;
  CHECK_THAT(s.mean(), WithinRel(4.0, 1e-14));

  s += ref::a;  // a whole range at once
  CHECK(s.size() == 3);
  CHECK(s.values() == std::vector<double>{3.9, 4.4, 2.2});
}

TEST_CASE("the accumulator is a callable", "[usability]") {
  RunningStats<double> s;
  std::for_each(ref::a.begin(), ref::a.end(), std::ref(s));
  CHECK(s.size() == ref::a.size());
  CHECK_THAT(s.mean(), WithinRel(ref::a_mean, ref::tol));
}

TEST_CASE("it consumes any range, views included", "[usability]") {
  RunningStats<double> s(100);
  auto squares = std::views::iota(1, 11) |
                 std::views::transform([](int i) { return double(i) * i; });
  s.push(squares);
  CHECK(s.size() == 10);
  CHECK_THAT(s.mean(), WithinRel(38.5, 1e-13));  // mean of 1,4,9,...,100
  CHECK(s.max() == 100.0);
}

TEST_CASE("merging two samples", "[usability]") {
  RunningStats<double> a(ref::a.size(), ref::a);
  RunningStats<double> b(ref::b.size(), ref::b);

  const auto m = merge(a, b);
  std::vector<double> all = ref::a;
  all.insert(all.end(), ref::b.begin(), ref::b.end());
  const RunningStats<double> direct(all.size(), all);

  CHECK(m.size() == all.size());
  CHECK_THAT(m.mean(), WithinRel(direct.mean(), 1e-12));
  CHECK_THAT(m.variance(), WithinRel(direct.variance(), 1e-11));
  CHECK_THAT(m.skewness(), WithinRel(direct.skewness(), 1e-10));
  CHECK_THAT(m.kurtosis(), WithinRel(direct.kurtosis(), 1e-10));
  CHECK(m.min() == direct.min());
  CHECK(m.max() == direct.max());
  CHECK(!m.bounded());  // the merged result keeps moments, not values

  SECTION("operator+ is the same thing") {
    const auto p = a + b;
    CHECK_THAT(p.mean(), WithinRel(m.mean(), 1e-15));
    CHECK_THAT(p.variance(), WithinRel(m.variance(), 1e-15));
  }
  SECTION("merging with an empty sample is a no-op") {
    const RunningStats<double> empty;
    CHECK_THAT(merge(a, empty).mean(), WithinRel(a.mean(), 1e-15));
    CHECK_THAT(merge(empty, a).variance(), WithinRel(a.variance(), 1e-15));
    CHECK(merge(empty, RunningStats<double>()).empty());
  }
  SECTION("merging keeps feeding") {
    auto p = a + b;
    p.push(100.0);
    CHECK(p.size() == all.size() + 1);
    CHECK(p.max() == 100.0);
  }
}

TEST_CASE("swap", "[usability]") {
  RunningStats<double> a(4, ref::a);
  RunningStats<double> b;
  b.push(ref::b);
  const double ma = a.mean(), mb = b.mean();

  swap(a, b);
  CHECK_THAT(a.mean(), WithinRel(mb, 1e-15));
  CHECK_THAT(b.mean(), WithinRel(ma, 1e-15));
  CHECK(!a.bounded());
  CHECK(b.capacity() == 4);
}

TEST_CASE("printing", "[usability]") {
  RunningStats<double> s(5);
  std::ostringstream os;

  os << s;
  CHECK(os.str() == "RunningStats[n=0/5]");

  os.str("");
  s.push({1.0, 2.0, 3.0});
  os << s;
  const std::string out = os.str();
  CHECK(out.find("n=3/5") != std::string::npos);
  CHECK(out.find("mean=2") != std::string::npos);
  CHECK(out.find("min=1") != std::string::npos);
  CHECK(out.find("max=3") != std::string::npos);

  SECTION("an unlimited accumulator says so") {
    RunningStats<double> u;
    std::ostringstream uos;
    uos << u;
    CHECK(uos.str() == "RunningStats[n=0/inf]");
  }
  SECTION("test results print a readable report") {
    RunningStats<double> a(ref::a.size(), ref::a);
    const std::string report = a.t_test(3.0).to_string();
    CHECK(report.find("One-sample t-test") != std::string::npos);
    CHECK(report.find("p-value") != std::string::npos);
    CHECK(report.find("95% confidence interval") != std::string::npos);

    std::ostringstream tos;
    tos << t_test(a, a);
    CHECK(tos.str().find("Welch") != std::string::npos);
  }
  SECTION("alternatives have names") {
    CHECK(std::string(to_string(Alternative::two_sided)) == "two-sided");
    CHECK(std::string(to_string(Alternative::less)) == "less");
    CHECK(std::string(to_string(Alternative::greater)) == "greater");
  }
}

TEST_CASE("an instance converts implicitly to its summary", "[usability]") {
  const RunningStats<double> s(ref::a.size(), ref::a);
  const Summary<double> sum = s;  // implicit conversion
  CHECK(sum.n == 10);
  CHECK_THAT(sum.mean, WithinRel(ref::a_mean, ref::tol));
}

TEST_CASE("copy and assignment preserve the state", "[usability]") {
  RunningStats<double> s(4, ref::a);
  const RunningStats<double> copy = s;
  CHECK(copy.values() == s.values());
  CHECK_THAT(copy.variance(), WithinRel(s.variance(), 1e-15));

  s.push(42.0);  // the copy is independent
  CHECK(copy.max() != 42.0);
  CHECK(s.max() == 42.0);
}
