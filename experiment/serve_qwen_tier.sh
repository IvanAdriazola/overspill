#!/bin/bash
# (inside WSL) Qwen3.6 FTW on the CPU-expert strategy; $1 = stock | tier | tier_nohead
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda
export TVM_FFI_CUDA_ARCH_LIST=8.6 TORCH_CUDA_ARCH_LIST=8.6 FLASHINFER_CUDA_ARCH_LIST=8.6
export PYTHONPATH=$HOME/src/freetoken-exp/python
case "$1" in
  tier) export FT_FILE_BANKS=1 FT_EMBED_HOST=1 FT_HEAD_HOST=1 ;;
  tier_nohead) export FT_FILE_BANKS=1 FT_EMBED_HOST=1 ;;
esac
shift
cd ~/ft
exec .venv/bin/ft serve --model ~/models/qwen36_nvfp4_ftw --served-model-name Qwen3.6-35B-A3B \
  --host 127.0.0.1 --port 1919 --text-model-only --moe-strategy cpu --max-running-requests 1 "$@"
