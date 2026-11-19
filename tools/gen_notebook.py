#!/usr/bin/env python3
"""Generate notebooks/miyabi_baseline.ipynb from the files in this repo.

The course runs code from Jupyter notebooks with `%%writefile` + `%%bash` +
`%%sub` cells (see ~/Downloads/pd00_intro.ipynb), so the notebook is just a
transport for the sources that already live in the repo. Generating it from the
repo files is the only way to keep the two from drifting: re-run this after any
source change.

    python3 tools/gen_notebook.py
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "miyabi_baseline.ipynb"
DEST = "~/swiglu-mlp-parallel"

# (path in repo, is binary-ish skipped) in dependency order.
FILES = [
    "CMakeLists.txt",
    "code/swiglu.h",
    "code/swiglu_ref.cpp",
    "code/swiglu_serial.cpp",
    "code/self_check.cpp",
    "code/bench_main.cpp",
    "code/problem.md",
    "scripts/run_baseline.sh",
    "tools/xcheck.py",
    "scripts/miyabi/env_g.sh",
    "scripts/miyabi/env_c.sh",
    "scripts/miyabi/run_baseline_g.pbs",
    "scripts/miyabi/run_baseline_c.pbs",
]


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def code(*lines: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def main() -> None:
    cells = [
        md("# Miyabi baseline (single-thread SwiGLU layer)",
           "",
           "Kernel: **Python (Miyabi G)**. Everything below is generated from the",
           "repo by `tools/gen_notebook.py`; edit the repo, not this notebook.",
           "",
           "Order: write the sources, build on the login node, then run the sweep on",
           "a compute node with `%%sub`.",
           "",
           "Before anything else, make sure the kernel can reach Miyabi (see",
           "pd00_intro.ipynb section 1): if a cell reports \"Not logged in to",
           "Miyabi\", open a Terminal and run `mount-miyabi`, then restart the",
           "kernel."),
        code("%load_ext miyabi"),
        md("## 1. Place the sources",
           "",
           "One directory per run; safe to re-execute (files are overwritten)."),
        code("%%bash",
             f"mkdir -p {DEST}/src {DEST}/scripts/miyabi {DEST}/results",
             f"ls -d {DEST}"),
    ]

    for rel in FILES:
        src = ROOT / rel
        text = src.read_text().rstrip("\n")
        cells.append(md(f"### `{rel}`"))
        lines = [f"%%writefile {DEST}/{rel}"] + text.split("\n")
        cells.append(code(*lines))

    cells += [
        md("## 2. Build and test on the login node",
           "",
           "Small and short enough to run on the login node. The build is the",
           "assignment's one-liner (no external library, one `main()`), and",
           "`--self-check` is the gate: the FP32 baseline must match the FP64",
           "reference to 2e-3 / 1e-5."),
        code("%%bash", "set -e",
             f"cd {DEST}",
             "chmod +x scripts/run_baseline.sh",
             "mkdir -p build",
             "g++ -O3 -std=c++17 -Wall code/*.cpp -o build/swiglu_bench",
             "./build/swiglu_bench --self-check"),
        md("## 3. Single-thread baseline on a compute node",
           "",
           "`%%sub` submits this cell as a PBS job, follows it until it exits, and",
           "streams stdout/stderr. Change `lecture1-g` to `lecture-g` outside the",
           "lecture slot (Mon 16:50-18:35). `group_list=gt81` is fixed by the",
           "course. The walltime cap for coursework is 15 minutes, hence 14:00.",
           "",
           "`TOKENS` is deliberately short here: 128 tokens x 20 repeats at",
           "~1.4 s/run already eats a few minutes on one core."),
        code("%%sub",
             "#PBS -q lecture1-g",
             "#PBS -l walltime=00:14:00",
             "set -e",
             f"cd {DEST}",
             'echo "host=$(hostname) user=$(id -un) pwd=$(pwd)"',
             'nproc; cc --version | head -2',
             "TOKENS=\"1 16 128\" REPEAT=5 scripts/run_baseline.sh | tee results/baseline_$(hostname -s).csv"),
        md("## 4. Cross-check one matrix outside the C++ code",
           "",
           "`--dump` writes the output matrix; `tools/xcheck.py` recomputes it in",
           "plain Python (no numpy) from the same xorshift32 seeds. This is the",
           "check that the C++ code is correct, not merely self-consistent."),
        code("%%bash", "set -e",
             f"cd {DEST}",
             "./build/swiglu_bench --tokens 4 --d-model 8 --d-ff 8 --repeat 2 --dump /tmp/out.txt",
             "python3 tools/xcheck.py /tmp/out.txt --tokens 4 --d-model 8 --d-ff 8"),
        md("## 5. Copy `code/` into the assignment directory",
           "",
           "The submission takes **all and only files under `code/`**, so before",
           "submitting, copy the five sources plus `problem.md` into the `code/`",
           "directory that sits next to pd02_app (no notebooks, no results, no",
           "`scripts/`):",
           "",
           "```sh",
           "cp ~/swiglu-mlp-parallel/code/* <assignment-dir>/code/",
           "```",
           "",
           "`code/` must contain exactly: `swiglu.h`, `swiglu_ref.cpp`,",
           "`swiglu_serial.cpp`, `self_check.cpp`, `bench_main.cpp`, `problem.md`.",
           "",
           "## 6. Reading the result",
           "",
           "One CSV row per token count:",
           "",
           "```",
           "impl,host,compiler,tokens,d_model,d_ff,repeat,median_ms,min_ms,gflops,checksum,abs_sum",
           "```",
           "",
           "`checksum`/`abs_sum` let you compare a compute-node run with a Mac run",
           "without shipping matrices.",
           "",
           "Keep `host` and `compiler` — Grace (512 GB/s) and Xeon Max (3.2 TB/s)",
           "numbers must never be mixed in one table. Next steps (parallel version,",
           "thread sweep) are issues #6-#8 in `docs/ISSUES.md`; by then the",
           "`OMP_NUM_THREADS` / `select=1:mpiprocs=1:ompthreads=N` variants belong in",
           "their own cells, not this one."),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python (Miyabi G)",
                           "language": "python", "name": "miyabi"},
            "language_info": {"name": "python", "version": "3.9.25"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
