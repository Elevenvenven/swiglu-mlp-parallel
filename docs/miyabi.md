# Miyabi での作業環境（pd00_intro.ipynb で確認した事実）

情報源: 授業ノートブック `pd00_intro.ipynb`（`~/Downloads/pd00_intro.ipynb`）と
Miyabi 仕様ページ。ノートブック側に書かれていることは実測済みの運用情報として
扱う。

## 1. どこで何が動くか

- ブラウザで開くのは **taulec**（`https://taulec.zapto.org:8000/`、東大 Google
  SSO）。ここは JupyterHub のフロントエンド。
- ノートブックのカーネルは **`Python (Miyabi G)`**。つまり `%%bash` セルは
  **Miyabi-G のログインノード `miyabi-g{1,2,..}` 上で動く**。taulec ではない。
- ターミナル（File → New → Terminal）は **taulec 上**で動く。Miyabi 側の
  ファイルは `~/miyabi` に sshfs でマウントされている（`mount-miyabi`）。
- 「Miyabi kernel: Not logged in to Miyabi」が出たら、ターミナルで
  `mount-miyabi` → Kernel → Restart Kernel。taulec→Miyabi の SSH が
  パスフレーズ / 確認コード無しで通っている必要がある。

帰結: **この Mac から ssh 鍵を登録しなくても、JupyterHub の中だけで全部できる。**
ログインノードで `git clone` して、`%%sub` で計算ノードに投げればよい。

## 2. ジョブ（PBS Professional、Slurm ではない）

| キュー | 中身 |
|---|---|
| `lecture-mig` / `lecture1-mig` | ARM CPU + GH200 を MIG で分割（小さい実験向け） |
| `lecture-g` / `lecture1-g` | ARM CPU + GH200（そのまま） |
| `lecture-c` / `lecture1-c` | Intel CPU（Xeon Max） |

**授業時間中（月 16:50–18:35）は `lecture1-*`、それ以外は `lecture-*`。**
CPU の実験も基本 ARM (Grace) 側で行う（`-c` はほとんど使わない）。
授業用の walltime 上限は 15 分。

最小ジョブスクリプト（ノートブックの記述そのまま）:

```sh
#!/bin/bash
#PBS -l select=1
#PBS -l walltime=5:00
#PBS -W group_list=gt81
#PBS -q lecture1-g
#PBS -j oe

cd "${PBS_O_WORKDIR}"
hostname
```

`group_list=gt81` は**固定**（「変更してはいけない」と明記されている）。
`-j oe` で stdout/stderr が `minjob.sh.oXXXXXXX` にまとまる。

## 3. `%%sub` cell magic（qsub の代わり）

```python
%load_ext miyabi
```

以降、セル先頭に `%%sub` を書くとそのセルが**ジョブとして投入され、終了まで
追跡されて出力が流れてくる**。`qsub` スクリプトを毎回書かなくてよい。

- `%%sub -n` … 投入されるスクリプトとオプションを表示するだけ（dry run）
- `%%sub --no` … 投入せずログインノードでそのまま実行（`%%bash` 相当）
- セル内で `#PBS -q lecture1-c` のように書けばオプションを上書きできる

このリポジトリの baseline を計算ノードで回すセル:

```python
%%sub
#PBS -q lecture1-g
set -e
cd ~/swiglu-mlp-parallel
cmake -B build -S . && cmake --build build -j
ctest --test-dir build --output-on-failure
scripts/run_baseline.sh | tee results/baseline_$(hostname -s).csv
```

## 4. この Mac から直接 ssh する場合（設定済み）

`~/.ssh/config` に `Host miyabig miyabi-g` / `Host miyabi-c` を追記済み
（ユーザ `t81030`、鍵 `~/.ssh/miyabi/ed25519`）。

`IdentitiesOnly yes` は必須。このマシンの `~/.ssh` には鍵が数十個あるので、
これが無いと ssh が全部差し出して `Too many authentication failures` になる。

Miyabi は **2 要素認証を keyboard-interactive で要求する**（実測: 正しい
ユーザ名にするとサーバの残り手段が `keyboard-interactive` になり、
`BatchMode=yes` や `PreferredAuthentications=publickey` 単独では
`Permission denied` になる）。したがって:

**確認コードを毎回避けられるのは `ControlMaster` + `ControlPersist` の
セッション使い回し**（コースの get_started.html が taulec 上で同じ設定をして
いる。`ControlPath ~/.ssh/cm_%r@%h:%p` / `ControlPersist 6h` の3行で、
6時間はコード無しでログインできる）。ただし**最初の1回だけは必ずコードが要る**:
`ssh miyabig` を初めて実行すると確認コード（初回は QR コードのスキャン）を
求められる。

- `ssh miyabig` を**対話的に一度だけ**実行し、確認コードを入力する
- 以降は `ControlMaster` / `ControlPersist 6h` のソケット経由で、コード無しに
  コマンドが通る（`ssh -o BatchMode=yes miyabig hostname` で確認できる）
- 6 時間でソケットが切れたら、また一度だけ対話ログインする

ジョブにコンパイラを渡すときの注意: サイトの環境が **`CC`/`CXX` を
nvc/nvc++ の絶対パスで先に export している**（module load 前から既に設定済み）。
そのため `qsub -v CXX=g++` は無視されて nvc++ でビルドされる（実測: 2 回とも
nvc++ になった）。`scripts/miyabi/env_g.sh` は `TOOLCHAIN` を見て CC/CXX を
決めるので、`qsub -q lecture-g -v TOOLCHAIN=g++ scripts/miyabi/run_baseline_g.pbs`
の形で切り替えること。`-v TOOLCHAIN=g++` が効いていないように見えたら、まず
`grep -c TOOLCHAIN scripts/miyabi/env_g.sh` で Miyabi 側のコピーが最新か確認する。
旧スクリプトのままだと `TOOLCHAIN` を知らないので nvc++ のまま走る（実測で
つまずいた箇所）。なお `rsync -az scripts/ ... miyabig:repo/` のようにディレクトリを
複数ソースに混ぜると、`scripts/` の中身がリポジトリ直下に展開されるので、
`rsync -az --exclude .git --exclude build ./ miyabig:swiglu-mlp-parallel/` を使う。CMake 側も、GCC 11 が aarch64 の `-march=neoverse-v2` を
知らないので、aarch64 では `-mcpu=native` を使っている（`CMakeLists.txt`）。

鍵がまだ通らない場合の確認順:

1. 公開鍵 `~/.ssh/miyabi/ed25519.pub`（`SHA256:J+gtLmDv0bxaZOwid1q8VOGu9jOzX4JiqFi/ZPNcBgI`）
   を Miyabi User Portal に登録したか。反映に時間がかかることがある。
2. `ssh -v miyabig` の `Offering public key:` と `Server accepts key` の行を見る。
3. それでもだめなら `ssh -o PreferredAuthentications=keyboard-interactive miyabig`
   で確認コードを入れ、通ったら `~/.ssh/config` の `ControlPersist` で再利用する。

教材・リポジトリを取る: `rsync -av miyabig:notebooks/pd/ ./course/pd/`
（ホームからの相対パスで指定する。絶対パスは `/work/gt81/share/home/t81030/` のはずだが未確認）
（`course/` は `.gitignore` 済み）

## 5. ハードウェアの目安（数字は仕様ページ由来）

| | Miyabi-G (Grace 側) | Miyabi-C |
|---|---|---|
| CPU | Grace 72c @3.0GHz (aarch64) | Xeon Max 9480 ×2 = 112c @1.9GHz |
| メモリ | 120 GB LPDDR5X / 512 GB/s | 128 GiB HBM2e / 3.2 TB/s |
| FP32 | 3.456 TF/s | 6.8 TF/s |
| コンパイラ | `nvc++`（`nvidia/25.3`, `nv-hpcx/25.3`） | `icpx`（`intel/2025.0.0`） |
| 起動 | `mpirun` | `mpiexec.hydra` |

16 トークンで重み 138 MB を 16 回読む実装（現 baseline をそのまま並列化した
場合）は 2.2 GB の読み出しになり、Grace で約 4.3 ms / Xeon Max で約 0.69 ms。
演算はそれぞれ 0.32 ms / 0.081 ms しかない。**G と C の数字は同じ表に混ぜない。**
