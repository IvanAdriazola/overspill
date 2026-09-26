#!/bin/bash
# (inside WSL) Serve Qwen3.6-35B-A3B from THIS tree with CUDA graphs on (normal speed).
# Usage: serve_exp.sh <lookahead_k> [extra ft serve args]   (k=0 = unmodified behaviour)
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda
export TVM_FFI_CUDA_ARCH_LIST=8.6 TORCH_CUDA_ARCH_LIST=8.6 FLASHINFER_CUDA_ARCH_LIST=8.6
export PYTHONPATH=$HOME/src/freetoken-exp/python
export FT_LOOKAHEAD_K=$1; shift
cd ~/ft
exec .venv/bin/ft serve --model ~/models/Qwen3.6-35B-A3B-NVFP4 --served-model-name Qwen3.6-35B-A3B \
  --host 127.0.0.1 --port 1919 --text-model-only "$@"
