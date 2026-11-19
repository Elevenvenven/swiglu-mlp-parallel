#!/usr/bin/env python3
"""Independent cross-check of `swiglu_bench --dump` without numpy.

Re-implements the layer in plain Python doubles, with the synthetic inputs
rounded to FP32 exactly as the C++ side does, and compares against a dumped
matrix. Agreement to ~1e-8 absolute means the C++ code is not just
self-consistent but actually computes the SwiGLU layer.

    ./swiglu_bench --tokens 4 --d-model 8 --d-ff 8 --dump /tmp/out.txt --repeat 2
    python3 tools/xcheck.py /tmp/out.txt --tokens 4 --d-model 8 --d-ff 8
"""
from __future__ import annotations

import argparse
import math
import struct


def f32(v: float) -> float:
    """Round a Python float to IEEE binary32, like the C++ side does."""
    return struct.unpack("<f", struct.pack("<f", v))[0]


class Rng:
    """xorshift32, identical to Rng in code/swiglu_ref.cpp."""

    def __init__(self, seed: int) -> None:
        self.s = seed if seed else 0x9E3779B9

    def next(self) -> int:
        s = self.s
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        self.s = s & 0xFFFFFFFF
        return self.s

    def uniform(self) -> float:
        # (float(n>>8) / 8388608.0f) - 1.0f, all in binary32.
        return f32(f32((self.next() >> 8) / 8388608.0) - 1.0)


def weights(tokens: int, H: int, I: int, seed: int) -> tuple[list[float], ...]:
    rng = Rng(seed)
    gate = [f32(rng.uniform() * 0.04) for _ in range(I * H)]
    up = [f32(rng.uniform() * 0.04) for _ in range(I * H)]
    down = [f32(rng.uniform() * 0.04) for _ in range(H * I)]
    return gate, up, down


def forward(T: int, H: int, I: int, seed: int) -> list[list[float]]:
    gate_w, up_w, down_w = weights(T, H, I, seed)
    rng = Rng(seed + 1)
    x = [[f32(rng.uniform() * 0.5) for _ in range(H)] for _ in range(T)]

    out = []
    for t in range(T):
        h = []
        for i in range(I):
            g = 0.0
            u = 0.0
            for k in range(H):
                xk = x[t][k]
                g += gate_w[i * H + k] * xk
                u += up_w[i * H + k] * xk
            h.append((g / (1.0 + math.exp(-g))) * u)
        row = []
        for r in range(H):
            acc = 0.0
            for i in range(I):
                acc += down_w[r * I + i] * h[i]
            row.append(acc)
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    with open(a.dump) as f:
        rows, cols = (int(v) for v in f.readline().split())
        got = [[float(v) for v in line.split(",")] for line in f if line.strip()]

    if (rows, cols) != (a.tokens, a.d_model) or len(got) != rows:
        print(f"shape mismatch: header {rows}x{cols}, rows {len(got)}")
        return 1

    want = forward(a.tokens, a.d_model, a.d_ff, a.seed)
    worst = max(abs(got[r][c] - want[r][c])
                for r in range(rows) for c in range(cols))
    scale = max(1.0, max(abs(v) for row in want for v in row))
    print(f"max |diff| = {worst:.3e}  (relative to max |value| {scale:.3e}: "
          f"{worst / scale:.3e})")
    return 0 if worst / scale < 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
