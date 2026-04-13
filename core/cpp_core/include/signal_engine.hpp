#pragma once

enum class SignalLevel { NORMAL = 0, WARN = 1, CRITICAL = 2 };
SignalLevel spread_signal(double market_price, double contract_price, double warn_thr, double critical_thr);
