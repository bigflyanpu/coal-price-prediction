#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "feature_engine.hpp"
#include "io_bridge.hpp"
#include "risk_engine.hpp"
#include "signal_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(coal_cpp_core, m) {
    m.doc() = "Coal price optional C++ acceleration module";

    py::enum_<SignalLevel>(m, "SignalLevel")
        .value("NORMAL", SignalLevel::NORMAL)
        .value("WARN", SignalLevel::WARN)
        .value("CRITICAL", SignalLevel::CRITICAL)
        .export_values();

    m.def("rolling_mean", &rolling_mean, py::arg("values"), py::arg("window"));
    m.def("clamp_daily_prediction", &clamp_daily_prediction, py::arg("pred"), py::arg("last_price"), py::arg("max_abs_jump"));
    m.def("spread_signal", &spread_signal, py::arg("market_price"), py::arg("contract_price"), py::arg("warn_thr"), py::arg("critical_thr"));
    m.def("io_bridge_version", &io_bridge_version);
}
