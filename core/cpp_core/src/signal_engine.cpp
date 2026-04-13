#include "signal_engine.hpp"

#include <cmath>

SignalLevel spread_signal(double market_price, double contract_price, double warn_thr, double critical_thr) {
  double spread = std::fabs(market_price - contract_price);
  if (spread >= critical_thr) return SignalLevel::CRITICAL;
  if (spread >= warn_thr) return SignalLevel::WARN;
  return SignalLevel::NORMAL;
}
