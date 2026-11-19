#!/bin/bash
# Miyabi-G environment (aarch64): NVIDIA GH200 — Grace CPU 72c @3.0 GHz,
# LPDDR5X 120 GB at 512 GB/s — plus an H100. Login host: miyabi-g.jcahpc.jp
#
# Verified on miyabi-g3 (2026-11-17): `module list` on a fresh login already
# shows nvidia/25.9(default) + nv-hpcx/25.9, and nvc/nvc++ live under
# /work/opt/local/aarch64/cores/nvidia/25.9/. Pin 25.9 to match that default.
#
# This project times the CPU path only. Grace (512 GB/s) and Miyabi-C HBM2e
# (3.2 TB/s) are different memory regimes, so -c and -g numbers must be
# reported separately and never mixed in one scaling table.
module purge
module load nvidia/25.9 nv-hpcx/25.9

# TOOLCHAIN selects the compiler, not $CXX: the site environment already exports
# CC/CXX as absolute nvc paths before any module load, so `qsub -v CXX=g++`
# silently loses to it (verified on miyabi-g3). `qsub -v TOOLCHAIN=g++` works.
TOOLCHAIN=${TOOLCHAIN:-nvc++}
case "$TOOLCHAIN" in
  nvc++|nvc) CC=nvc;    CXX=nvc++ ;;
  g++|gcc)   CC=gcc;    CXX=g++ ;;
  *) echo "unknown TOOLCHAIN=$TOOLCHAIN (want nvc++ or g++)" >&2; return 1 2>/dev/null || exit 1 ;;
esac
export CC CXX
