#include "risk_engine.hpp"

#include <algorithm>

double clamp_daily_prediction(double pred, double last_price, double max_abs_jump) {
  double lo = last_price - max_abs_jump;
  double hi = last_price + max_abs_jump;
  return std::clamp(pred, lo, hi);
}
