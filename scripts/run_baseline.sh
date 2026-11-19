#!/bin/bash
# Baseline sweep: one process per token count, single thread, CSV to stdout.
#
#   scripts/run_baseline.sh > results/baseline.csv
#   TOKENS="1 16" scripts/run_baseline.sh        # shorter sweep
#
# Everything here is single-threaded on purpose: this file records the number
# that future parallel versions are scored against.
set -euo pipefail

BUILD=${BUILD:-build}
BENCH=${BENCH:-$BUILD/swiglu_bench}
TOKENS=${TOKENS:-"1 2 4 8 16 32 64 128"}
REPEAT=${REPEAT:-20}

if [[ ! -x "$BENCH" ]]; then
  echo "missing $BENCH (configure with cmake -B $BUILD -S . && cmake --build $BUILD)" >&2
  exit 1
fi

# Keep BLAS/OpenMP-linked runtimes from spawning helpers behind our back.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

first=1
for t in $TOKENS; do
  if [[ $first == 1 ]]; then
    "$BENCH" --impl serial --tokens "$t" --repeat "$REPEAT"
    first=0
  else
    "$BENCH" --impl serial --tokens "$t" --repeat "$REPEAT" --no-header
  fi
done
