#!/bin/bash
# (inside WSL) Convert the Qwen3.6-35B-A3B NVFP4 HF checkpoint to FTW (per-layer, 4096-aligned expert banks).
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda
export TVM_FFI_CUDA_ARCH_LIST=8.6 TORCH_CUDA_ARCH_LIST=8.6 FLASHINFER_CUDA_ARCH_LIST=8.6
export PYTHONPATH=$HOME/src/freetoken-exp/python
cd ~/ft
exec .venv/bin/ft checkpoint --model ~/models/Qwen3.6-35B-A3B-NVFP4 --out ~/models/qwen36_nvfp4_ftw "$@"
