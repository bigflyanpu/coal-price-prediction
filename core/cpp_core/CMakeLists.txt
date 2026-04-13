cmake_minimum_required(VERSION 3.16)
project(coal_price_cpp_core)
set(CMAKE_CXX_STANDARD 17)

add_library(core STATIC
  src/feature_engine.cpp
  src/signal_engine.cpp
  src/risk_engine.cpp
  src/io_bridge.cpp
)

target_include_directories(core PUBLIC include)
