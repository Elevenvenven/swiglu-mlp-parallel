#!/bin/bash
# Miyabi-C environment (x86_64): 2x Intel Xeon Max 9480, 112 cores,
# 128 GiB HBM2e at 3.2 TB/s. Login host: miyabi-c.jcahpc.jp
#
# Sourced by scripts/miyabi/*.pbs and usable interactively on the login node.
module purge
module load intel/2025.0.0

# Same rule as env_g.sh: pick the compiler through TOOLCHAIN, never through
# $CC/$CXX, because the site environment pre-sets those.
TOOLCHAIN=${TOOLCHAIN:-intel}
case "$TOOLCHAIN" in
  intel)     CC=icx;  CXX=icpx ;;
  g++|gcc)   CC=gcc;  CXX=g++  ;;
  *) echo "unknown TOOLCHAIN=$TOOLCHAIN (want intel or g++)" >&2; return 1 2>/dev/null || exit 1 ;;
esac
export CC CXX
