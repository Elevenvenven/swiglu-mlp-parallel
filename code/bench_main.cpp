// Benchmark harness: median wall-clock latency + GFLOPS, CSV on stdout.
//
//   ./swiglu_bench --tokens 16 --warmup 2 --repeat 20
//   ./swiglu_bench --self-check                 # correctness gate
//   ./swiglu_bench --tokens 4 --dump out.txt    # output for external checks
//
// Single-threaded baselines only; sweeps are driven by
// scripts/run_baseline.sh.

#include "swiglu.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

struct Args {
  std::string impl = "serial";  // serial | reference
  swiglu::Dims dims{};
  int warmup = 2;
  int repeat = 20;
  bool header = true;
  unsigned seed = 1;
  bool self_check = false;
  std::string dump;  // dump the output matrix here (cross-check vs numpy)
};

void usage() {
  std::printf(
      "usage: swiglu_bench [--impl serial|reference] [--tokens T]\n"
      "                    [--d-model H] [--d-ff I] [--warmup N] [--repeat N]\n"
      "                    [--seed N] [--no-header] [--self-check]\n"
      "                    [--dump FILE]\n");
}

int to_int(const char* s) {
  int v = 0;
  auto [p, ec] = std::from_chars(s, s + std::strlen(s), v);
  if (ec != std::errc()) {
    std::fprintf(stderr, "invalid integer: %s\n", s);
    std::exit(2);
  }
  return v;
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", k.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (k == "--impl") a.impl = next();
    else if (k == "--tokens") a.dims.tokens = to_int(next());
    else if (k == "--d-model") a.dims.d_model = to_int(next());
    else if (k == "--d-ff") a.dims.d_ff = to_int(next());
    else if (k == "--warmup") a.warmup = to_int(next());
    else if (k == "--repeat") a.repeat = to_int(next());
    else if (k == "--seed") a.seed = static_cast<unsigned>(to_int(next()));
    else if (k == "--no-header") a.header = false;
    else if (k == "--self-check") a.self_check = true;
    else if (k == "--dump") a.dump = next();
    else if (k == "-h" || k == "--help") { usage(); std::exit(0); }
    else { std::fprintf(stderr, "unknown flag: %s\n", k.c_str()); std::exit(2); }
  }
  return a;
}

std::string compiler_id() {
#if defined(__NVCOMPILER)
  return "nvc++";
#elif defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
  return "icpx";
#elif defined(__clang__)
  return "clang++";
#elif defined(__GNUC__)
  return "g++";
#else
  return "unknown";
#endif
}

std::string host_name() {
  char buf[256] = {0};
  if (gethostname(buf, sizeof(buf) - 1) != 0) return "unknown";
  std::string s(buf);
  const auto dot = s.find('.');
  return dot == std::string::npos ? s : s.substr(0, dot);
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  if (n % 2 == 1) return v[n / 2];
  return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);

  if (a.self_check) return swiglu::self_check() == 0 ? 0 : 1;

  swiglu::Weights w;
  swiglu::init_weights(w, a.dims, a.seed);
  swiglu::MatrixF32 x = swiglu::make_input(a.dims, a.seed + 1);
  swiglu::MatrixF32 out;
  swiglu::MatrixF64 xd, outd;

  const bool reference = a.impl == "reference";
  if (reference) {
    xd = swiglu::MatrixF64(a.dims.tokens, a.dims.d_model);
    for (std::size_t i = 0; i < x.data.size(); ++i) {
      xd.data[i] = static_cast<double>(x.data[i]);
    }
  }

  auto run_once = [&]() {
    if (reference) {
      swiglu::swiglu_reference(xd, w, outd);
    } else if (a.impl == "serial") {
      swiglu::swiglu_serial(x, w, out);
    } else {
      std::fprintf(stderr, "unknown impl: %s\n", a.impl.c_str());
      std::exit(2);
    }
  };

  run_once();  // also warms the page cache for the weight arrays
  for (int i = 0; i < a.warmup; ++i) run_once();

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(a.repeat));
  for (int i = 0; i < a.repeat; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    run_once();
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double>(t1 - t0).count());
  }

  const double med = median(samples);
  const double best = *std::min_element(samples.begin(), samples.end());

  // Verifiability: a checksum lets any two runs (impl, thread count, compiler,
  // node) be compared without shipping matrices around, and --dump writes the
  // raw output so it can be diffed against an independent implementation.
  const swiglu::MatrixF32& y = out;
  double checksum = 0.0;
  double abs_sum = 0.0;
  if (!reference) {
    for (const float v : y.data) {
      checksum += static_cast<double>(v);
      abs_sum += std::fabs(static_cast<double>(v));
    }
  }

  if (!a.dump.empty() && !reference) {
    std::FILE* f = std::fopen(a.dump.c_str(), "w");
    if (!f) {
      std::fprintf(stderr, "cannot write %s\n", a.dump.c_str());
      return 2;
    }
    std::fprintf(f, "%d %d\n", y.rows, y.cols);
    for (int r = 0; r < y.rows; ++r) {
      for (int c = 0; c < y.cols; ++c) {
        std::fprintf(f, "%s%.9g", c ? "," : "", static_cast<double>(y.at(r, c)));
      }
      std::fputc('\n', f);
    }
    std::fclose(f);
  }

  if (a.header) {
    std::printf(
        "impl,host,compiler,tokens,d_model,d_ff,repeat,median_ms,min_ms,"
        "gflops,checksum,abs_sum\n");
  }
  std::printf("%s,%s,%s,%d,%d,%d,%d,%.4f,%.4f,%.2f,%.9g,%.9g\n",
              a.impl.c_str(), host_name().c_str(), compiler_id().c_str(),
              a.dims.tokens, a.dims.d_model, a.dims.d_ff, a.repeat, med * 1e3,
              best * 1e3, swiglu::gflops(a.dims, med), checksum, abs_sum);
  return 0;
}
