# SwiGLU MLP inference — single-thread baseline, Miyabi

[English](#english) · [日本語](#日本語)

Current state: **single-threaded baseline only**. The OpenMP variants were
removed; parallel work resumes from issue #6 in [docs/ISSUES.md](docs/ISSUES.md)
so the shared tree always builds and always has a trustworthy reference number.

## English

A course project on one SwiGLU feed-forward layer used in LLM inference. We
implement the three linear projections and the SiLU gate in C++ and establish a
single-threaded baseline that later parallel versions are measured against.

### What is measured

- Synthetic FP32 tensors with TinyLlama-1.1B dimensions: hidden 2048,
  intermediate 5632, weights and inputs FP32 (`code/swiglu.h:16-21`).
- Median wall-clock latency of one full layer (gate, up, SiLU product, down)
  for 1/2/4/8/16/32/64/128 input tokens, single thread, one seed.
- Throughput in GFLOP/s computed as `6 * tokens * d_model * d_ff / seconds`.
- Correctness against an independent FP64, unfused reference
  (`code/swiglu_ref.cpp`): `swiglu_bench --self-check` prints `max_abs`/`max_rel`
  and exits non-zero on failure; `tools/xcheck.py` re-derives one dumped matrix
  in plain Python as an external check.
- Every CSV row carries `checksum` and `abs_sum` of the output matrix, so runs
  from different compilers/nodes are comparable without moving matrices.

### Build and run

```sh
g++ -O3 -std=c++17 code/*.cpp -o build/swiglu_bench    # no external library
./build/swiglu_bench --self-check
scripts/run_baseline.sh > results/baseline.csv
```

`cmake -B build -S . && cmake --build build -j && ctest --test-dir build` is the
same thing with warnings enabled; the bare `g++` line is the course's
"self-contained, no external library" requirement, and it is what must keep
working.

On Miyabi the same two steps are wrapped by
`scripts/miyabi/run_baseline_{c,g}.pbs`, or run from a notebook:
`notebooks/miyabi_baseline.ipynb` writes these sources on the Miyabi login node
and submits the sweep with `%%sub`. It is generated from the repo by
`tools/gen_notebook.py` (regenerate after editing sources). Access, module sets
and queue names: [docs/miyabi.md](docs/miyabi.md).

### Results on Miyabi-G (2026-11-17)

One node of queue `lecture-g`, host `mg0030`, nvc++ 25.9 (`-O3`, default
`-tp neoverse-v2`), 20 repeats, median of `results/baseline_mg0030_nvc++.csv`:

| tokens | median ms | GFLOP/s |
| ---: | ---: | ---: |
| 1 | 4.896 | 14.14 |
| 2 | 8.368 | 16.54 |
| 4 | 16.405 | 16.87 |
| 8 | 31.368 | 17.65 |
| 16 | 61.849 | 17.90 |
| 32 | 122.872 | 18.02 |
| 64 | 245.249 | 18.06 |
| 128 | 495.254 | 17.89 |

Two things to read off this table:

- **Time is exactly linear in tokens** (4.9 ms at 1 token, 495 ms at 128), so
  GFLOP/s is flat at 14–18. The kernel is therefore limited by how fast one
  core issues FMAs, not by the token count.
- **The compiler dominates on Grace.** The same sources built with g++ 11.4.1
  (`-O3 -march=native`) give 275.8 ms / 4.02 GFLOP/s at 16 tokens on the login
  node `miyabi-g3`; nvc++ is 4.4× faster because it vectorizes the
  gather/scatter form of the per-channel dot products that gcc leaves scalar.

Cross-machine check: `checksum` for 16 tokens is `23.8034274` and `abs_sum`
`2112.68559` on `mg0030`, versus `23.8034219` / `2112.68556` for g++ on an Apple
M5. Different compiler, different vectorization, different accumulation order —
still the same value to six significant digits, which is what those two columns
exist to prove without shipping matrices around.

Environment capture: `results/env_miyabi-g.txt` (module list, compiler
versions, `uname`, `/proc/cpuinfo`).

### Where the time goes

For 16 tokens the layer touches 138 MB of weights (2.21 GB if the weights are
re-read once per token) but does only 1.107 GFLOP. Read at Grace's 512 GB/s,
that floor is 4.3 ms — the measured 61.8 ms sits 14× above it, so on **one**
core this kernel is compute-bound, not memory-bound. The memory floor only
becomes the real limit once threads widen the arithmetic, and then Miyabi-C
(3.2 TB/s, 112 cores) and Miyabi-G (512 GB/s, 72 cores) land in very different
places. That is why the next issue is "loop over tokens *inside* the parallel
region" (re-read weights once, not once per token) rather than "add threads".

## 日本語

LLM 推論のフィードフォワード層（SwiGLU）を 1 層取り上げ、3 つの線形射影と
SiLU ゲートを C++ で実装し、**単一スレッドの baseline** を確定させます。
以降の並列版はこの baseline を基準に評価します。並列実装はいったん削除して
おり、再開は [docs/ISSUES.md](docs/ISSUES.md) の issue #6 から行います。

### 計測内容

- TinyLlama-1.1B と同じ次元（隠れ層 2048、中間層 5632）の人工 FP32 データ。
- 1 層分（gate, up, SiLU 積, down）の中央実行時間を、トークン数
  1/2/4/8/16/32/64/128 で単一スレッド計測。
- GFLOP/s は `6 * tokens * d_model * d_ff / 秒`。
- FP64 の非融合参照実装（`code/swiglu_ref.cpp`）との `max_abs` / `max_rel` を
  `./build/swiglu_bench --self-check` で検証（許容 `2e-3` / `1e-5`）。
  `tools/xcheck.py` は dump した行列を Python で独立に再計算する。

### ビルドと実行

上記 English 節と同じ 3 コマンド。Miyabi 上では
`scripts/miyabi/run_baseline_{c,g}.pbs` を `qsub` するか、ノートブック
`notebooks/miyabi_baseline.ipynb` を使う（Miyabi のログインノードにこれらの
ソースを `%%writefile` で書き、`%%sub` でスイープを投入する）。ノートブックは
`tools/gen_notebook.py` がリポジトリから生成するので、ソースを直したら再生成する。
手順は [docs/miyabi.md](docs/miyabi.md)。

### Miyabi-G での実測値（2026-11-17）

`lecture-g` の 1 ノード、ホスト `mg0030`、nvc++ 25.9（`-O3`、既定の
`-tp neoverse-v2`）、20 回の中央値（`results/baseline_mg0030_nvc++.csv`）:

| トークン | 中央値 ms | GFLOP/s |
| ---: | ---: | ---: |
| 1 | 4.896 | 14.14 |
| 16 | 61.849 | 17.90 |
| 128 | 495.254 | 17.89 |

- **時間はトークン数にちょうど比例する**（1 トークン 4.9 ms → 128 トークン
  495 ms）。GFLOP/s が 14〜18 で平坦なので、1 コアの FMA 発行速度が律速。
- **Grace ではコンパイラの差が大きい。** 同じソースを g++ 11.4.1
  （`-O3 -march=native`）でビルドすると 16 トークンで 275.8 ms / 4.02 GFLOP/s
  （ログインノード `miyabi-g3`）。nvc++ が 4.4 倍速いのは、チャネルごとの
  内積（gather/scatter 形）を gcc がスカラのままにするのに対し、nvc++ が
  ベクトル化するため。
- 機械をまたいだ照合: 16 トークンの `checksum` は `mg0030` で `23.8034274`、
  M5 の g++ で `23.8034219`（`abs_sum` は `2112.68559` / `2112.68556`）。
  コンパイラもベクトル化も加算順も違うのに 6 桁一致するのが、行列を運ばずに
  結果を比較するための `checksum` / `abs_sum` 列の役目。

環境の記録: `results/env_miyabi-g.txt`（module list、コンパイラ版、`uname`、
`/proc/cpuinfo`）。

### 時間はどこで消えているか

16 トークンでは重み 138 MB（トークンごとに読み直すと 2.21 GB）に対して計算は
1.107 GFLOP しかない。Grace の 512 GB/s で読んだ場合の下限は 4.3 ms なのに
実測は 61.8 ms で、**下限の 14 倍上**にある。つまり **1 コアでは** このカー
ネルは演算律速であって帯域律速ではない。帯域の下限が本当に効いてくるのは
スレッドで演算を広げてからで、そのとき Miyabi-C（3.2 TB/s, 112 コア）と
Miyabi-G（512 GB/s, 72 コア）では行き先が大きく変わる。だから次の issue は
「スレッドを増やす」ではなく「並列領域の内側でトークンを回す（重みの再読み
込みをトークン回数分やめる）」である。
