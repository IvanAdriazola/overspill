#!/bin/bash
# (inside WSL) Convert the DeepSeek-V4-Flash REAP-150B HF checkpoint to FTW (per-layer, 4096-aligned expert banks).
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda
export TVM_FFI_CUDA_ARCH_LIST=8.6 TORCH_CUDA_ARCH_LIST=8.6 FLASHINFER_CUDA_ARCH_LIST=8.6
export PYTHONPATH=$HOME/src/freetoken-exp/python
cd ~/ft
SRC=${DSV4_SRC:-$HOME/models/dsv4_reap150b}; OUT=${DSV4_OUT:-$HOME/models/dsv4_reap150b_ftw}
echo "convert $SRC -> $OUT"
exec .venv/bin/ft checkpoint --model "$SRC" --out "$OUT" "$@"
