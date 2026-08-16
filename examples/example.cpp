// A short tour of statisticalc: a simulated measurement stream is monitored
// through a rolling window, then two production batches are compared.
#include <iostream>
#include <random>
#include <statisticalc/statisticalc.hpp>

using statisticalc::Alternative;
using statisticalc::RunningStats;

int main() {
  std::mt19937 rng(20260816);

  // ---------------------------------------------------------------------
  // 1. Rolling window over a stream of measurements
  // ---------------------------------------------------------------------
  std::normal_distribution<double> process(10.0, 0.2);
  RunningStats<double> window(50);  // last 50 samples
  RunningStats<double> lifetime;    // unlimited, O(1) memory

  for (int i = 0; i < 500; ++i) {
    const double x = process(rng);
    window << x;
    lifetime << x;
  }

  std::cout << "rolling  " << window << '\n'
            << "lifetime " << lifetime << '\n'
            << "  median      = " << window.median() << '\n'
            << "  IQR         = " << window.iqr() << '\n'
            << "  skewness    = " << window.skewness() << '\n'
            << "  kurtosis    = " << window.kurtosis() << '\n'
            << "  95% CI mean = " << window.mean_ci() << "\n\n";

  // Is the process still centred on the 10.0 nominal value?
  std::cout << window.t_test(10.0) << '\n';

  // Is the spread still within the 0.2 nominal standard deviation?
  std::cout << window.variance_test(0.2 * 0.2,
                                    {.alternative = Alternative::greater})
            << '\n';

  // ---------------------------------------------------------------------
  // 2. Comparing two batches
  // ---------------------------------------------------------------------
  std::normal_distribution<double> batch1(10.00, 0.20);
  std::normal_distribution<double> batch2(10.12, 0.35);

  RunningStats<double> first(30), second(30);
  for (int i = 0; i < 30; ++i) {
    first << batch1(rng);
    second << batch2(rng);
  }

  std::cout << "batch 1 " << first << '\n' << "batch 2 " << second << "\n\n";
  std::cout << t_test(first, second) << '\n';  // Welch, means
  std::cout << f_test(first, second) << '\n';  // variances
  std::cout << "Cohen's d      = " << cohens_d(first, second) << '\n'
            << "same mean?     = " << (same_mean(first, second) ? "yes" : "no")
            << '\n'
            << "same variance? = "
            << (same_variance(first, second) ? "yes" : "no") << '\n';

  // The two batches can also be pooled into a single sample.
  const auto both = first + second;
  std::cout << "\npooled   " << both << '\n';

  return 0;
}
