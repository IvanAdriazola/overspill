#!/bin/bash
# (inside WSL) Serve DeepSeek-V4-Flash REAP-150B (FTW) with file-mapped expert banks:
# page cache = RAM tier, NVMe = the rest; expert math on the CPU executor, the rest on the GPU.
# Usage: serve_dsv4.sh [extra ft serve args]
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda
export TVM_FFI_CUDA_ARCH_LIST=8.6 TORCH_CUDA_ARCH_LIST=8.6 FLASHINFER_CUDA_ARCH_LIST=8.6
export PYTHONPATH=$HOME/src/freetoken-exp/python
export FT_FILE_BANKS=1
export FT_EMBED_HOST=1
export FT_HEAD_HOST=1
cd ~/ft
exec .venv/bin/ft serve --model ~/models/dsv4_reap150b_ftw --served-model-name dsv4-reap \
  --host 127.0.0.1 --port 1919 --moe-strategy cpu "$@"
