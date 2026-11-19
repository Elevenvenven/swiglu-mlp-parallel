#include "swiglu.h"

#include <cmath>

namespace swiglu {

// Single-threaded FP32 baseline. Fused: SiLU is applied to each gate value the
// moment it is produced, so no [tokens][d_ff] gate/up arrays are materialised.
void swiglu_serial(const MatrixF32& x, const Weights& w, MatrixF32& out) {
  const int T = x.rows;
  const int H = x.cols;
  const int I = static_cast<int>(w.gate.size() / std::size_t(H));

  out = MatrixF32(T, H);
  std::vector<float> h(static_cast<std::size_t>(I));

  for (int t = 0; t < T; ++t) {
    const float* xt = x.row(t);
    for (int i = 0; i < I; ++i) {
      const float* wg = w.gate.data() + std::size_t(i) * H;
      const float* wu = w.up.data() + std::size_t(i) * H;
      float g = 0.f, u = 0.f;
      for (int k = 0; k < H; ++k) {
        const float xk = xt[k];
        g += wg[k] * xk;
        u += wu[k] * xk;
      }
      h[std::size_t(i)] = (g / (1.f + std::exp(-g))) * u;
    }

    float* ot = out.row(t);
    for (int r = 0; r < H; ++r) {
      const float* wd = w.down.data() + std::size_t(r) * I;
      const float* hh = h.data();
      float acc = 0.f;
      for (int i = 0; i < I; ++i) {
        acc += wd[i] * hh[i];
      }
      ot[r] = acc;
    }
  }
}

}  // namespace swiglu
