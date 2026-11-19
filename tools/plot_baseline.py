#!/usr/bin/env python3
"""Plot every results/baseline_*.csv as latency and throughput curves.

    uv run --with matplotlib python3 tools/plot_baseline.py results/baseline_curve.png

Two panels: median wall-clock latency (log-log) and GFLOP/s, one line per
(host, compiler) so a run on Grace, one on Apple silicon and one under a
different compiler are directly comparable without reading the CSVs.

Requires matplotlib only (`uv run --with matplotlib ...`); the benchmark itself
still has no Python dependency.
"""
import csv
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(paths):
    series = {}
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = f"{row['host']} / {row['compiler']}"
                series.setdefault(key, []).append(
                    (int(row["tokens"]), float(row["median_ms"]), float(row["gflops"]))
                )
    for rows in series.values():
        rows.sort()
    return series


def main() -> int:
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/baseline_curve.png")
    paths = sorted(pathlib.Path("results").glob("baseline_*.csv"))
    if not paths:
        print("no results/baseline_*.csv", file=sys.stderr)
        return 1
    series = load(paths)

    fig, (latency, throughput) = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, rows in series.items():
        tokens = [r[0] for r in rows]
        latency.plot(tokens, [r[1] for r in rows], marker="o", label=label)
        throughput.plot(tokens, [r[2] for r in rows], marker="s", label=label)

    latency.set(xlabel="tokens (d_model=2048, d_ff=5632)", ylabel="median latency [ms]",
                title="one SwiGLU layer, single thread")
    latency.set_xscale("log")
    latency.set_yscale("log")
    latency.grid(True, which="both", alpha=0.3)
    throughput.set(xlabel="tokens", ylabel="GFLOP/s",
                   title="6*T*d_model*d_ff / seconds")
    throughput.set_xscale("log")
    throughput.grid(True, which="both", alpha=0.3)
    throughput.legend(fontsize=8)

    latency.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
