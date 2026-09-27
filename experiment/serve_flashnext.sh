#!/bin/bash
# (inside WSL) Serve Qwen3.8-Flash-Next (FTW) with Overspill. $1 = cpu | mixed
#   cpu:   every expert layer file-mapped + computed on the CPU executor (the DeepSeek setup)
#   mixed: FreeToken pins what fits (GPU offload path); the other layers are file-mapped for the CPU
# The PLE n-gram table stays on disk (FreeToken's default --ple-backend disk).
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda
export TVM_FFI_CUDA_ARCH_LIST=8.6 TORCH_CUDA_ARCH_LIST=8.6 FLASHINFER_CUDA_ARCH_LIST=8.6
export PYTHONPATH=$HOME/src/freetoken-exp/python
export FT_FILE_BANKS=1 FT_EMBED_HOST=${FT_EMBED_HOST:-1} FT_HEAD_HOST=${FT_HEAD_HOST:-1} FT_HEAD_I8=${FT_HEAD_I8:-1}
export FT_CPU_PREFILL_MAX=${FT_CPU_PREFILL_MAX:-256} FT_WILLNEED=${FT_WILLNEED:-1} FREETOKEN_PIN_BUDGET_GB=${FREETOKEN_PIN_BUDGET_GB:-}
[ -z "$FREETOKEN_PIN_BUDGET_GB" ] && unset FREETOKEN_PIN_BUDGET_GB
MODE=$1; shift
case "$MODE" in
  cpu)   STRATEGY=(--moe-strategy cpu) ;;
  mixed) STRATEGY=(--moe-strategy offload --moe-cpu-layers auto) ;;
  *) echo "mode must be cpu|mixed"; exit 2 ;;
esac
cd ~/ft
exec .venv/bin/ft serve --model ~/models/flashnext_ftw --served-model-name flashnext --text-model-only \
  --host 127.0.0.1 --port 1919 "${STRATEGY[@]}" --max-running-requests 1 --cuda-graph-max-bs 1 "$@"
