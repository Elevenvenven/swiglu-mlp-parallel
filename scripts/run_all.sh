#!/bin/bash
# One command: build, test, sweep, plot. Issue #8.
#
#   bash scripts/run_all.sh
#
# Regenerates results/baseline_<hostname>.csv and results/baseline_curve.png
# from a clean checkout. No network, no Python needed for the measurement
# itself; matplotlib is only pulled in by `uv` for the figure, and if `uv` is
# absent the figure step is skipped with a warning rather than failing the run.
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD=${BUILD:-build}
TOKENS=${TOKENS:-"1 2 4 8 16 32 64 128"}
REPEAT=${REPEAT:-20}

cmake -B "$BUILD" -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$BUILD" -j
ctest --test-dir "$BUILD" --output-on-failure

mkdir -p results
csv="results/baseline_$(hostname -s).csv"
BUILD="$BUILD" TOKENS="$TOKENS" REPEAT="$REPEAT" scripts/run_baseline.sh | tee "$csv"
echo "wrote $csv"

if command -v uv >/dev/null 2>&1; then
  uv run --quiet --with matplotlib python3 tools/plot_baseline.py results/baseline_curve.png
elif python3 -c 'import matplotlib' >/dev/null 2>&1; then
  python3 tools/plot_baseline.py results/baseline_curve.png
else
  echo "skip figure: neither uv nor matplotlib available" >&2
fi
