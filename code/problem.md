# SwiGLU feed-forward layer of a transformer LLM (single layer, inference)

## 1. The problem

A transformer LLM (GPT/Llama/Qwen family) is a stack of identical blocks, and
each block ends with a *feed-forward network* (FFN) that does most of the
floating-point work. In Llama-style models that FFN is called **SwiGLU** and
consists of three matrix-vector products plus a nonlinearity:

```
gate = W_gate · x        # [d_ff]
up   = W_up   · x        # [d_ff]
h    = SiLU(gate) ⊙ up   # elementwise, SiLU(g) = g / (1 + exp(-g))
out  = W_down · h        # [d_model]
```

where `x` is one token's activation vector of length `d_model`, and the weights
are `W_gate, W_up ∈ R^{d_ff × d_model}`, `W_down ∈ R^{d_model × d_ff}`. For the
Llama-2/TinyLlama shape used here, `d_model = 2048`, `d_ff = 5632`, so one layer
holds 3 · 2048 · 5632 = 34.6 M weights (138 MB in FP32).

Inference is batched: `T` tokens are pushed through the layer at once, which is
just the same three matrix products with an `T × d_model` input. Nothing here is
approximate: the output must match a straightforward reference computation.

Why this is a reasonable parallel target:

- The three projections are independent output channels per token: with
  `d_ff = 5632` and 72+ cores there is abundant parallelism (issue #6).
- The arithmetic is simple enough to be checked exactly (see §3), so speed can
  be studied without arguing about correctness.
- It is genuinely memory-heavy: for `T = 16` the layer touches 2.2 GB of weights
  (138 MB read 16 times) but does only 1.107 GFLOP, so a naive parallel version
  is limited by memory bandwidth, not by FLOPs. Measuring *where* the time goes
  is therefore part of the exercise, not a side note.

This baseline deliberately does **one layer, one forward pass, synthetic
weights**. Training a real LLM is out of scope; extracting the real weights of
one TinyLlama layer is an optional extra, not part of the submitted code.

## 2. The baseline code (how it solves the problem)

Files (all C++, no external libraries — only the standard library that the
compiler links by itself):

| file | content |
|---|---|
| `code/swiglu.h` | shapes (`Dims`), weight/activation containers, FP32/FP64 matrix type, `gflops()` |
| `code/swiglu_ref.cpp` | deterministic synthetic data + FP64 unfused reference implementation |
| `code/swiglu_serial.cpp` | the baseline: single-threaded FP32, fused |
| `code/self_check.cpp` | correctness gate, `swiglu::self_check()` |
| `code/bench_main.cpp` | the `main()`: timing, CSV output, `--self-check`, `--dump` |

Build (one command, no build system required):

```sh
g++ -O3 -std=c++17 code/*.cpp -o swiglu_bench
```

Run:

```sh
./swiglu_bench --tokens 16 --repeat 20      # one CSV row: latency + GFLOP/s
./swiglu_bench --self-check                 # exit 0 iff within tolerance
./swiglu_bench --tokens 4 --dump out.txt    # output matrix for external checks
```

Everything is configurable from the command line: `--tokens`, `--d-model`,
`--d-ff`, `--repeat`, `--warmup`, `--seed`. The default (`tokens=16`,
`repeat=20`) finishes in a few seconds; the largest sweep value (128 tokens)
takes about 1.4 s per run on one core, which is why sweeps are driven by
`scripts/run_baseline.sh`.

Implementation notes:

- **Fused, not materialised.** Gate and up are accumulated together, and SiLU
  is applied immediately, so no `T × d_ff` gate/up arrays are ever written to
  memory. Only the `d_ff` vector `h` exists per token.
- **Row-major weights**, stored exactly as PyTorch's `nn.Linear` does
  (`W[r][c]` for output channel `r`), so real weights can be dropped in later
  without a transposition.
- **Deterministic synthetic input** (xorshift32, fixed seed) so two people
  benchmarking get the same numbers; weights are scaled to 0.04 and inputs to
  0.5 so SiLU neither saturates nor vanishes and FP32 error stays meaningful.
- **No `-ffast-math`.** Reordering the sums would make the FP32-vs-FP64
  comparison meaningless, and the baseline is the reference every later version
  is judged against.

## 3. How to verify the result

1. **Self-check (in the submitted code).** `code/self_check.cpp` recomputes the
   same layer in FP64 with a deliberately naive, unfused formulation (separate
   `gate`, `up`, `h` buffers) and compares against the FP32 baseline for
   `tokens = 1, 16, 128, 4`. It fails if `max_abs > 2e-3` or
   `max_rel > 1e-5`; measured values are ~1e-7, i.e. four orders of magnitude
   inside the tolerance (FP32 accumulation error over 2048 and 5632 terms).

   ```
   $ ./swiglu_bench --self-check
   [tokens=16 d_model=2048 d_ff=5632]
     serial vs fp64 ref: max_abs=8.638e-07 max_rel=8.638e-07 mean_abs=1.102e-07
   ALL CHECKS PASSED
   ```

2. **Checksum column.** Every CSV row carries `checksum` and `abs_sum` of the
   output matrix. Any two runs (different compiler, node, thread count later)
   must agree to ~1e-6 relative; that catches "fast but wrong" variants without
   shipping matrices around.

3. **External cross-check.** `--dump out.txt` writes `rows cols` followed by the
   matrix as comma-separated values. Recomputing that in Python (`tokens`,
   `d_model`, `d_ff` are in the CSV row; the same xorshift32 weights can be
   regenerated in a dozen lines) gives an independent check of the
   implementation, not just of its internal consistency.

## 4. What is *not* done yet

This is the baseline only. Multicore parallelisation, thread scaling, SIMD and
blocking are the next steps, tracked as issues #6–#8 in `docs/ISSUES.md`.
