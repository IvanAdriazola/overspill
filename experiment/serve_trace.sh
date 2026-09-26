#!/bin/bash
# Serve Qwen3.6-35B-A3B from THIS tree (not ~/ft's install) with routing traces on.
# Reuses ~/ft/.venv; PYTHONPATH puts ~/src/freetoken-exp/python ahead of the editable ~/src/FreeToken.
# --graph 0: CUDA-graph replays skip Python, so the tracer would see nothing.
# Usage (inside WSL): serve_trace.sh <trace_dir> [extra ft serve args]
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda
export TVM_FFI_CUDA_ARCH_LIST=8.6 TORCH_CUDA_ARCH_LIST=8.6 FLASHINFER_CUDA_ARCH_LIST=8.6
export PYTHONPATH=$HOME/src/freetoken-exp/python
export FT_ROUTE_TRACE=$1; shift
mkdir -p "$FT_ROUTE_TRACE"
cd ~/ft
exec .venv/bin/ft serve --model ~/models/Qwen3.6-35B-A3B-NVFP4 --served-model-name Qwen3.6-35B-A3B \
  --host 127.0.0.1 --port 1919 --text-model-only --graph 0 "$@"
