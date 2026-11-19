cmake_minimum_required(VERSION 3.16)
project(swiglu_mlp_parallel LANGUAGES CXX)

# C++17 is enough (integer std::from_chars), and it is what a bare
# `g++ code/*.cpp` invocation defaults to on Miyabi. Staying there means the
# course submission builds with the same one-liner the README shows.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# Single-threaded baseline only. No OpenMP, no threading library, no external
# dependency of any kind: the course requires code that links nothing beyond
# what the compiler links by itself. No -ffast-math either — the FP64
# correctness check would become meaningless.
add_compile_options(-O3 -Wall -Wextra -Wno-unused-parameter)

if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
  add_compile_options(-xHost)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # GCC 11 (the Miyabi-G system compiler) does not know aarch64 -march=neoverse-v2,
  # so native tuning on that partition has to go through -mcpu.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    add_compile_options(-mcpu=native)
  else()
    add_compile_options(-march=native)
  endif()
endif()

# One binary, one main(): the submission is compiled as a whole, so the
# correctness check is `swiglu_bench --self-check`, not a second executable.
add_executable(swiglu_bench
  code/swiglu_ref.cpp
  code/swiglu_serial.cpp
  code/self_check.cpp
  code/bench_main.cpp)
target_include_directories(swiglu_bench PRIVATE code)

enable_testing()
add_test(NAME correctness COMMAND swiglu_bench --self-check)
