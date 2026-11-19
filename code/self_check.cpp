// Correctness gate: FP32 baseline vs FP64 reference.
//
// No main() here on purpose. The course submission is compiled as one program
// from all of code/*.cpp, so the check hangs off `swiglu_bench --self-check`
// instead of being a second executable.
#include "swiglu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct Stats {
  double max_abs = 0.0;
  double max_rel = 0.0;  // max|a-b| / max(1, |ref|)
  double mean_abs = 0.0;
};

Stats compare(const swiglu::MatrixF32& got, const swiglu::MatrixF64& ref) {
  Stats s;
  double sum = 0.0;
  const std::size_t n = ref.data.size();
  for (std::size_t i = 0; i < n; ++i) {
    const double a = static_cast<double>(got.data[i]);
    const double b = ref.data[i];
    const double d = std::fabs(a - b);
    s.max_abs = std::max(s.max_abs, d);
    s.max_rel = std::max(s.max_rel, d / std::max(1.0, std::fabs(b)));
    sum += d;
  }
  s.mean_abs = sum / static_cast<double>(n);
  return s;
}

}  // namespace

namespace swiglu {

int self_check() {
  // Tolerance for FP32 accumulation over d_model / d_ff terms against an FP64
  // reference; measured values sit ~3 orders of magnitude below this.
  constexpr double kMaxAbsTol = 2e-3;
  constexpr double kMaxRelTol = 1e-5;

  int failures = 0;
  const Dims dims[] = {
      {1, 2048, 5632},   // single token (decode)
      {16, 2048, 5632},  // default
      {128, 2048, 5632}, // batch case
      {4, 256, 512},     // small, fast sanity case
  };

  for (const auto& d : dims) {
    std::printf("[tokens=%d d_model=%d d_ff=%d]\n", d.tokens, d.d_model,
                d.d_ff);

    Weights w;
    init_weights(w, d);
    MatrixF32 x = make_input(d);

    MatrixF64 xd(d.tokens, d.d_model);
    for (std::size_t i = 0; i < x.data.size(); ++i) {
      xd.data[i] = static_cast<double>(x.data[i]);
    }

    MatrixF64 ref;
    swiglu_reference(xd, w, ref);

    MatrixF32 serial;
    swiglu_serial(x, w, serial);

    const Stats ss = compare(serial, ref);
    std::printf("  serial vs fp64 ref: max_abs=%.3e max_rel=%.3e mean_abs=%.3e\n",
                ss.max_abs, ss.max_rel, ss.mean_abs);

    if (ss.max_abs > kMaxAbsTol || ss.max_rel > kMaxRelTol) {
      std::printf("  FAIL: serial out of tolerance\n");
      ++failures;
    }
    std::printf("\n");
  }

  std::printf("%s\n", failures == 0 ? "ALL CHECKS PASSED" : "CHECKS FAILED");
  return failures;
}

}  // namespace swiglu
