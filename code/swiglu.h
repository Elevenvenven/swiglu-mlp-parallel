// SwiGLU feed-forward layer (single layer, inference, FP32).
//
//   gate = W_gate @ x        [d_ff]
//   up   = W_up   @ x        [d_ff]
//   h    = SiLU(gate) * up   [d_ff]
//   out  = W_down @ h        [d_model]
//
// with SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g)).
// x is [tokens][d_model], out is [tokens][d_model].
#pragma once

#include <cstddef>
#include <vector>

namespace swiglu {

// TinyLlama-1.1B feed-forward dimensions (hidden 2048, intermediate 5632).
struct Dims {
  int tokens = 16;
  int d_model = 2048;
  int d_ff = 5632;
};

// Row-major weights, as PyTorch nn.Linear stores them:
//   gate/up: [d_ff][d_model]   down: [d_model][d_ff]
struct Weights {
  std::vector<float> gate;
  std::vector<float> up;
  std::vector<float> down;
};

template <typename T>
struct Matrix {
  int rows = 0;
  int cols = 0;
  std::vector<T> data;

  Matrix() = default;
  Matrix(int r, int c) : rows(r), cols(c), data(std::size_t(r) * c, T{}) {}
  T& at(int r, int c) { return data[std::size_t(r) * cols + c]; }
  const T& at(int r, int c) const { return data[std::size_t(r) * cols + c]; }
  T* row(int r) { return data.data() + std::size_t(r) * cols; }
  const T* row(int r) const { return data.data() + std::size_t(r) * cols; }
};

using MatrixF32 = Matrix<float>;
using MatrixF64 = Matrix<double>;

// Deterministic synthetic data so both developers benchmark identical inputs.
void init_weights(Weights& w, const Dims& d, unsigned seed = 1);
MatrixF32 make_input(const Dims& d, unsigned seed = 2);

// Double-precision, unfused reference (allocates separate gate/up/h buffers).
// Slower on purpose; it exists only to define the expected output.
void swiglu_reference(const MatrixF64& x, const Weights& w, MatrixF64& out);

// Single-threaded FP32 baseline: fused per output channel (README's "fused").
// Every later parallel/optimized version is compared against this: same
// FP accumulation order, so mismatches are visible as bit differences.
void swiglu_serial(const MatrixF32& x, const Weights& w, MatrixF32& out);

// Correctness gate: swiglu_serial vs the FP64 reference on a few shapes.
// Returns the number of failed checks (0 = pass). Lives in self_check.cpp and
// is called by `swiglu_bench --self-check`; the submission must stay a single
// binary, so there is deliberately no second main().
int self_check();

// Effective throughput in GFLOP/s: 3 projections x 2 FLOPs per MAC.
inline double gflops(const Dims& d, double seconds) {
  // 3 projections x 2 FLOPs per MAC x tokens * d_model * d_ff
  const double flops =
      6.0 * d.tokens * d.d_model * d.d_ff;
  return flops / seconds / 1e9;
}

}  // namespace swiglu
