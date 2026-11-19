#include "swiglu.h"

#include <cmath>
#include <cstdint>

namespace swiglu {
namespace {

// Small deterministic PRNG (xorshift32): avoids <random> implementation drift
// between platforms and keeps synthetic inputs reproducible.
struct Rng {
  std::uint32_t s;
  explicit Rng(std::uint32_t seed) : s(seed ? seed : 0x9e3779b9u) {}
  std::uint32_t next() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
  // Uniform in [-1, 1).
  float uniform() {
    return (static_cast<float>(next() >> 8) / 8388608.0f) - 1.0f;
  }
};

// Scale keeps the projections in a range where SiLU neither saturates nor
// vanishes, which is what we want to compare numerics against.
constexpr float kWeightScale = 0.04f;
constexpr float kInputScale = 0.5f;

}  // namespace

void init_weights(Weights& w, const Dims& d, unsigned seed) {
  const std::size_t iface = std::size_t(d.d_ff) * d.d_model;
  const std::size_t down = std::size_t(d.d_model) * d.d_ff;
  w.gate.resize(iface);
  w.up.resize(iface);
  w.down.resize(down);

  Rng rng(seed);
  for (float& v : w.gate) v = rng.uniform() * kWeightScale;
  for (float& v : w.up) v = rng.uniform() * kWeightScale;
  for (float& v : w.down) v = rng.uniform() * kWeightScale;
}

MatrixF32 make_input(const Dims& d, unsigned seed) {
  MatrixF32 x(d.tokens, d.d_model);
  Rng rng(seed);
  for (float& v : x.data) v = rng.uniform() * kInputScale;
  return x;
}

void swiglu_reference(const MatrixF64& x, const Weights& w, MatrixF64& out) {
  const int T = x.rows;
  const int H = x.cols;
  const int I = static_cast<int>(w.gate.size() / std::size_t(H));

  out = MatrixF64(T, H);

  // Deliberately unfused: whole gate array, then whole up array, then h.
  std::vector<double> gate(I), up(I), h(I);
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < I; ++i) {
      double g = 0.0, u = 0.0;
      const float* wg = w.gate.data() + std::size_t(i) * H;
      const float* wu = w.up.data() + std::size_t(i) * H;
      for (int k = 0; k < H; ++k) {
        const double xk = x.at(t, k);
        g += static_cast<double>(wg[k]) * xk;
        u += static_cast<double>(wu[k]) * xk;
      }
      gate[i] = g;
      up[i] = u;
    }
    for (int i = 0; i < I; ++i) {
      const double g = gate[i];
      h[i] = (g / (1.0 + std::exp(-g))) * up[i];
    }
    for (int r = 0; r < H; ++r) {
      const float* wd = w.down.data() + std::size_t(r) * I;
      double acc = 0.0;
      for (int i = 0; i < I; ++i) {
        acc += static_cast<double>(wd[i]) * h[i];
      }
      out.at(t, r) = acc;
    }
  }
}

}  // namespace swiglu
