#include "feature_engine.hpp"

#include <algorithm>
#include <numeric>

double rolling_mean(const std::vector<double>& values, int window) {
  if (values.empty() || window <= 0) return 0.0;
  const int n = static_cast<int>(values.size());
  const int start = std::max(0, n - window);
  const int len = n - start;
  if (len <= 0) return 0.0;
  double sum = std::accumulate(values.begin() + start, values.end(), 0.0);
  return sum / len;
}
