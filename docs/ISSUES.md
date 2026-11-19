# Issues — single-thread baseline first

Board: `Backlog → Ready → In Progress → Review → Done → Blocked`
Labels: `epic/baseline`, `epic/parallel`, `epic/eval`, `epic/model`, `prio/P0`, `prio/P1`, `owner/A`, `owner/B`

Current repository state: the parallel implementation has been removed. What
remains is the single-threaded baseline and its correctness gate. The issues
below are written to be pasted into GitHub (`gh issue create`) once the team
agrees on the wording; they are the source of truth for "what to do next" so
the shared branch is never a half-finished feature.

## Submission constraints (pd02_app) — always true, not an issue

From the assignment page:

- `code/*.cpp`, `code/*.h`, `code/*.cc` only — **self-contained, no external
  library** beyond what the compiler links by itself. `code/` currently holds
  `swiglu.h`, `swiglu_ref.cpp`, `swiglu_serial.cpp`, `self_check.cpp`,
  `bench_main.cpp`, `problem.md` and nothing else.
- **One `main()` only.** Everything under `code/` is compiled as one program, so
  a second `main()` (e.g. a separate test binary) is a link error — that is why
  the correctness gate is `swiglu_bench --self-check` in `code/self_check.cpp`.
- Verdict check: `g++ -O3 -std=c++17 code/*.cpp -o swiglu_bench` must succeed.
- Run time must be controllable (CLI flags `--tokens/--d-model/--d-ff/--repeat`);
  the default run is seconds, the largest sweep value ~1.4 s per run.
- `code/problem.md` must be self-contained (problem + baseline explanation +
  how to verify).
- `code/` must contain *only* what is submitted — no notebooks, no results.

## #1 Freeze the spec (shapes / precision / I-O) — P0

Acceptance criteria
- `README.md` states `d_model=2048`, `d_ff=5632`, FP32 weights, FP32 I/O.
- Names the target hardware and what "one measurement" means (median of N,
  same seed, page cache warm).

## #2 Single-thread SwiGLU MLP (done) — P0

Acceptance criteria
- `code/swiglu_serial.cpp` fuses gate/up/down per output channel.
- `./build/swiglu_bench --self-check` (or `ctest --test-dir build`) passes.
- Status: **Done** in this tree; this issue exists only so the baseline is
  explicitly signed off before parallel work starts.

## #3 Benchmark harness with CSV output (done) — P0

Acceptance criteria
- `swiglu_bench --impl serial --tokens T --warmup W --repeat N` prints one CSV
  row, including `checksum`/`abs_sum` so results are comparable across nodes.
- `scripts/run_baseline.sh` sweeps token counts and emits a header once.

## #4 Correctness gate against FP64 reference (done) — P0

Acceptance criteria
- `code/self_check.cpp` compares `swiglu_serial` to `swiglu_reference`
  for tokens 1/16/128 and fails above `max_abs=2e-3` or `max_rel=1e-5`.
- Depends on #2.

## #5 Run the baseline on Miyabi (G done, C open) — P0

Status (2026-11-17): **G done**, C not started.

- Submitted `qsub -q lecture-g scripts/miyabi/run_baseline_g.pbs` on `miyabi-g3`
  (job `4054624`), ran on compute node `mg0030`, ctest passed in 1.39 s.
- Artifacts: `results/baseline_mg0030_nvc++.csv`, `results/env_miyabi-g.txt`.
- Numbers: 16 tokens 61.85 ms = 17.90 GFLOP/s (nvc++ 25.9); the same sources
  under g++ 11.4.1 give 275.8 ms = 4.02 GFLOP/s (login node). Throughput is flat
  at 14-18 GFLOP/s across 1-128 tokens, i.e. one Grace core is compute-bound,
  14x above the 512 GB/s memory floor. See README "Results on Miyabi-G".

Acceptance criteria
- `results/baseline_<host>_<compiler>.csv` in the tree, with `host`,
  `compiler` populated, and `results/env_<host>.txt` next to it.
- `qsub scripts/miyabi/run_baseline_c.pbs` also completes on Miyabi-C (x86_64,
  2x Xeon Max 9480); module/compiler names there are still unverified, because
  the course states -c is rarely used and this project reports G as the main
  table.
- Depends on #2, #3, #4, and on knowing the AccountingGroup (`group_list=gt81`).

Gotchas found while doing this
- `lecture-g` requests `nvnodes-g` resources; when they are exhausted the job
  sits in `Q` with `comment = Not Running: Insufficient amount of resource:
  nvnodes-g`. Check `qstat -f <jid> | grep -E "comment|estimated.start_time"`.
- The default module set on a fresh Miyabi-G login is `nvidia/25.9(default)` +
  `nv-hpcx/25.9`, not the 25.3 that `scripts/miyabi/env_g.sh` used to pin.
- `qstat -Q` is not supported by this PBS (2024.1.5); use `qstat --limit` for
  project limits and `qstat -f` for one job.
- The cluster stops on **2026-11-25 09:00** (shown by `qstat`), so anything that
  needs a node has to happen before then.

## #6 Parallelize one layer with OpenMP (after #5) — P0

Acceptance criteria
- Parallel form keeps the weight-load term O(1) in token count: parallelize
  over `d_ff`/output channels, loop over tokens inside, not `collapse(2)`
  over `(tokens, i)`.
- Output matches the FP64 reference within the #4 tolerances; bit-identity
  with `swiglu_serial` is *not* required once the accumulation order changes.
- Thread sweep 1/2/4/8/16/32/72 (G) and /112 (C), CSV + speedup figure.

## #7 One further optimization (SIMD or blocking) — P1

Acceptance criteria
- ≥10% faster than #6 for at least one (tokens, threads) pair, without failing #4.

## #8 One-command reproduction — P1

Acceptance criteria
- `bash scripts/run_all.sh` rebuilds, re-tests and regenerates every CSV/figure
  in `results/` from a clean checkout.

## #9 Real weights from one TinyLlama layer — P1, optional

Acceptance criteria
- Extracted `W_gate/W_up/W_down` of layer 0 are loaded and forward-passed
  through `swiglu_serial` at least once.
- If weight export is blocked, move to `Blocked` and keep the synthetic path
  as the reported result.

## #10 (stretch) Same check with a Qwen 0.8B layer — optional

Acceptance criteria
- One layer forward-passes; speed is not required.

## Known documentation drift to fix under #1

- `README.md` previously implied a parallel implementation and a "fused vs
  separate kernel" comparison that no longer exists in this tree.
- `README.md` thread counts (1/2/4/8) are far below Miyabi-C's 112 cores and
  Miyabi-G's 72 cores; the sweep in #6 is the replacement.
