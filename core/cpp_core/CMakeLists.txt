cmake_minimum_required(VERSION 3.16)
project(coal_price_cpp_core)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(core STATIC
  src/feature_engine.cpp
  src/signal_engine.cpp
  src/risk_engine.cpp
  src/io_bridge.cpp
)

target_include_directories(core PUBLIC include)

option(BUILD_PYTHON_MODULE "Build pybind11 python extension" ON)
if(BUILD_PYTHON_MODULE)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
  endif()

  pybind11_add_module(coal_cpp_core bindings/pybind_module.cpp)
  target_link_libraries(coal_cpp_core PRIVATE core)
  target_include_directories(coal_cpp_core PRIVATE include)
endif()
