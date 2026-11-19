#!/bin/bash
# Create the labels and the #1-#10 issue list from docs/ISSUES.md.
# Run once from the repo root; safe to re-run (labels use --force, issues are
# created only if the title does not already exist).
set -euo pipefail

create_label() { gh label create "$1" --color "$2" --description "$3" --force >/dev/null; }

create_label "epic/baseline" "0e8a16" "Baseline: spec, single-thread kernel, Miyabi reference numbers"
create_label "epic/parallel" "1d76db" "Parallelisation of the layer"
create_label "epic/eval"     "5319e7" "Harness, correctness gate, figures, reproduction"
create_label "epic/model"    "d93f0b" "Real model weights (TinyLlama / Qwen)"
create_label "prio/P0"       "b60205" "Must have for the course deadline"
create_label "prio/P1"       "fbca04" "Should have"
create_label "owner/A"       "c2e0c6" "Kernel owner"
create_label "owner/B"       "bfdadc" "Eval/infra owner"

issue() { # title labels body
  local title=$1 labels=$2 body=$3
  if gh issue list --state all --search "\"$title\" in:title" --json title -q '.[].title' 2>/dev/null | grep -Fxq "$title"; then
    echo "skip (exists): $title"
  else
    gh issue create --title "$title" --body "$body" --label "$labels" | tail -1
  fi
}

issue "Freeze the spec: shapes, precision, I/O" "epic/baseline,prio/P0,owner/A,owner/B" 'Part of `epic/baseline`. The spec has to be one fixed thing before any number is comparable across machines or teammates.

Acceptance criteria
- [ ] `README.md` states `d_model=2048`, `d_ff=5632`, FP32 weights, FP32 I/O.
- [ ] Names the target hardware and what "one measurement" means (median of N, same seed, page cache warm).
- [ ] The measured numbers in `README.md` match `results/*.csv` (they are copied from there, not retyped).

Reference: `code/swiglu.h:16-21` holds the defaults; `README.md` "What is measured".'

issue "Single-thread SwiGLU MLP" "epic/baseline,prio/P0,owner/A" 'Part of `epic/baseline`.

Acceptance criteria
- [ ] `code/swiglu_serial.cpp` fuses gate/up/down per output channel.
- [ ] `./build/swiglu_bench --self-check` (or `ctest --test-dir build`) passes.
- [ ] Exactly one `main()` in `code/` (the submission is compiled as one program).

Status: implemented in this tree; this issue is the explicit sign-off before anything else builds on it.'

issue "Benchmark harness with CSV output" "epic/eval,prio/P0,owner/B" 'Part of `epic/eval`.

Acceptance criteria
- [ ] `swiglu_bench --impl serial --tokens T --warmup W --repeat N` prints one CSV row with `impl,host,compiler,tokens,d_model,d_ff,repeat,median_ms,min_ms,gflops,checksum,abs_sum`.
- [ ] `scripts/run_baseline.sh` sweeps token counts and emits the header once.
- [ ] `checksum`/`abs_sum` are stable enough to compare runs across machines without moving matrices.

Status: implemented (`code/bench_main.cpp`, `scripts/run_baseline.sh`).'

issue "Correctness gate against FP64 reference" "epic/eval,prio/P0,owner/B" 'Part of `epic/eval`. Depends on #2.

Acceptance criteria
- [ ] `code/self_check.cpp` compares `swiglu_serial` against `swiglu_reference` for tokens 1/16/128 and fails above `max_abs=2e-3` or `max_rel=1e-5`.
- [ ] `ctest --test-dir build` passes on Miyabi (it does: 1.39 s on `mg0030`).
- [ ] `tools/xcheck.py` re-derives a dumped matrix in plain Python as an independent check.

Measured: `max_abs` 1.3e-06 (nvc++: 2.8e-07) at 128 tokens; `xcheck` agrees to 5e-12.'

issue "Run the baseline on Miyabi (G done, C open)" "epic/baseline,prio/P0,owner/B" 'Part of `epic/baseline`. Depends on #2, #3, #4.

Status (2026-11-17): **Miyabi-G done, Miyabi-C not started.**

- `qsub -q lecture-g scripts/miyabi/run_baseline_g.pbs` ran on compute node `mg0030` (job `4054624`).
- Artifacts: `results/baseline_mg0030_nvc++.csv`, `results/env_miyabi-g.txt`, `results/baseline_curve.png`.
- Numbers: 16 tokens 61.85 ms = 17.90 GFLOP/s with nvc++ 25.9; the same sources under g++ 11.4.1 give 4.02 GFLOP/s. Throughput is flat at 14-18 GFLOP/s for 1-128 tokens, so one Grace core is compute-bound, 14x above the 512 GB/s memory floor.

Acceptance criteria
- [ ] `results/baseline_<host>_<compiler>.csv` in the tree, with `host` and `compiler` populated, plus `results/env_<host>.txt`.
- [ ] `qsub scripts/miyabi/run_baseline_c.pbs` also completes on Miyabi-C (x86_64, 2x Xeon Max 9480) — module names there are still unverified.

Gotchas
- `lecture-g` requests `nvnodes-g`; when they are exhausted the job sits in `Q` with `comment = Not Running: Insufficient amount of resource: nvnodes-g`. Inspect with `qstat -f <jid> | grep -E "comment|estimated.start_time"`.
- This PBS (2024.1.5) does not support `qstat -Q`; use `qstat --limit`.
- The cluster stops on **2026-11-25 09:00**, so node work has to finish before that.'

issue "Parallelize one layer with OpenMP" "epic/parallel,prio/P0,owner/A" 'Part of `epic/parallel`. Depends on #5.

**Currently in Backlog**: the parallel variants were removed from `code/` to satisfy the submission constraint (one program, no external libraries), so this issue is where they come back if the team decides to report scaling.

Acceptance criteria
- [ ] The parallel form keeps the weight-load term O(1) in token count: parallelize over `d_ff`/output channels and loop over tokens *inside*, not `collapse(2)` over `(tokens, i)`.
- [ ] Output matches the FP64 reference within the #4 tolerances; bit-identity with `swiglu_serial` is *not* required once the accumulation order changes.
- [ ] Thread sweep 1/2/4/8/16/32/72 (G) and /112 (C), CSV plus a speedup figure.

Baseline to beat: 16 tokens, 61.85 ms / 17.90 GFLOP/s (`results/baseline_mg0030_nvc++.csv`).'

issue "One further optimization (SIMD or blocking)" "epic/parallel,prio/P1,owner/A" 'Part of `epic/parallel`. Depends on #6.

Acceptance criteria
- [ ] At least 10% faster than #6 for at least one (tokens, threads) pair, without failing #4.
- [ ] The compiler gap is already large (nvc++ 4.4x g++ at -O3), so state which flags each measurement used.'

issue "One-command reproduction" "epic/eval,prio/P1,owner/B" 'Part of `epic/eval`. Depends on #3, #4, #5.

Acceptance criteria
- [ ] `bash scripts/run_all.sh` rebuilds, re-tests and regenerates every CSV/figure in `results/` from a clean checkout.
- [ ] `results/baseline_curve.png` is regenerated by `tools/plot_baseline.py` (needs matplotlib only, e.g. `uv run --with matplotlib`).'

issue "Real weights from one TinyLlama layer" "epic/model,prio/P1,owner/A" 'Part of `epic/model`. Optional.

Acceptance criteria
- [ ] `W_gate`/`W_up`/`W_down` of layer 0 are extracted and forward-passed through `swiglu_serial` at least once.
- [ ] If weight export is blocked, move this to `Blocked` and keep the synthetic path as the reported result. Do not close it.

TinyLlama-1.1B dimensions are what `code/swiglu.h` already uses (2048 / 5632).'

issue "Same check with a Qwen 0.8B layer (stretch)" "epic/model,prio/P1,owner/B" 'Part of `epic/model`. Stretch goal.

Acceptance criteria
- [ ] One layer forward-passes; speed is not required.
- [ ] Drop it without ceremony if the week runs out — the synthetic path is the reported result.'
