# Parallel SwiGLU MLP Inference with OpenMP

[English](#english) · [日本語](#日本語)

## English

A course project on parallelizing a single SwiGLU feed-forward layer used in LLM inference. We implement the three linear projections and the SiLU gate in C++ and compare a single-threaded baseline with an OpenMP implementation.

### Experiment plan

- Start with synthetic FP32 tensors using TinyLlama's dimensions (hidden size 2048, intermediate size 5632).
- Compare the outputs with an independent reference and measure median latency for 1, 16, and 128 input tokens across 1, 2, 4, and 8 OpenMP threads.
- Compare separate and fused gate/up computation. In the fused version, apply the SiLU gate and multiply immediately after computing each pair of projection outputs, avoiding two intermediate arrays.
- If time permits, validate one layer with pretrained TinyLlama weights.

## 日本語

LLM の推論で使われる SwiGLU 付きフィードフォワード層を 1 層取り上げ、並列化の効果を調べる授業プロジェクトです。3 つの線形射影と SiLU を用いたゲート処理を C++ で実装し、単一スレッド版と OpenMP 版を比較します。

### 実験計画

- まずは TinyLlama と同じ次元（隠れ層 2048、中間層 5632）の人工データを FP32 で用意します。
- 独立した参照実装と出力を照合します。入力を 1、16、128 トークンとし、OpenMP のスレッド数を 1、2、4、8 に変えて、実行時間の中央値を測ります。
- gate と up の射影を別々に計算する実装と、各出力の組を計算した直後にゲート処理と要素積を実行する融合版を比較します。融合版では 2 つの中間配列への書き出しを省きます。
- 時間に余裕があれば、学習済みの TinyLlama の重みを使って 1 層分の出力を検証します。
