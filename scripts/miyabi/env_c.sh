#!/bin/bash
# Miyabi-C environment (x86_64): 2x Intel Xeon Max 9480, 112 cores,
# 128 GiB HBM2e at 3.2 TB/s. Login host: miyabi-c.jcahpc.jp
#
# Sourced by scripts/miyabi/*.pbs and usable interactively on the login node.
module purge
module load intel/2025.0.0

export CC=icx
export CXX=icpx
