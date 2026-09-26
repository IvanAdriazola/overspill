#!/usr/bin/env bash
# Serve a GGUF with llama.cpp's OpenAI-compatible server (run inside WSL as user ivan).
#
#   Usage: serve_llamacpp.sh /path/to/model.gguf [extra llama-server args...]
#   From Git Bash:
#     MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u ivan -- bash \
#       /mnt/c/GIT/Freetoken-colibri-experiment/experiment/llamacpp/serve_llamacpp.sh /path/model.gguf
#   For split GGUFs pass the *-00001-of-0000N.gguf file.
#
# Layout: all layers on GPU (-ngl 99), but every MoE expert tensor
# (ffn_{gate,up,down}_exps) stays in CPU memory (--cpu-moe == --n-cpu-moe <all layers>).
# The model is mmap'd (--load-mode mmap; --no-mmap / load-mode none is NOT used),
# so the expert weights can be larger than RAM and are paged in from disk on demand.
# Tip: keep the GGUF on the WSL ext4 filesystem (not /mnt/c) - 9P page-in is far slower.
#
# Env overrides: PORT (1920), HOST (127.0.0.1), CTX (8192), THREADS (11),
#                LLAMA_BIN (~/src/llama.cpp/build/bin), ALIAS (basename of the gguf)
set -euo pipefail

MODEL="${1:?usage: $0 /path/to/model.gguf [extra llama-server args]}"
shift || true
[ -f "$MODEL" ] || { echo "model not found: $MODEL" >&2; exit 1; }

LLAMA_BIN="${LLAMA_BIN:-$HOME/src/llama.cpp/build/bin}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-1920}"
CTX="${CTX:-8192}"
THREADS="${THREADS:-11}"
ALIAS="${ALIAS:-$(basename "$MODEL" .gguf)}"

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

# --cuda-unified-memory (consumed here): set GGML_CUDA_ENABLE_UNIFIED_MEMORY=1, as the
# pixi-llm-recipes setup does, so CUDA may spill allocations into host memory.
ARGS=()
for a in "$@"; do
  if [ "$a" = "--cuda-unified-memory" ]; then export GGML_CUDA_ENABLE_UNIFIED_MEMORY=1; else ARGS+=("$a"); fi
done
set -- "${ARGS[@]}"

# Experts on the CPU for every layer, unless the caller passes --n-cpu-moe N (then only the first N).
EXPERT_PLACEMENT="--cpu-moe"
case " $* " in *" --n-cpu-moe "*) EXPERT_PLACEMENT="" ;; esac

set -x
exec "$LLAMA_BIN/llama-server" \
  --model "$MODEL" \
  --alias "$ALIAS" \
  --host "$HOST" --port "$PORT" \
  --ctx-size "$CTX" \
  --parallel 1 \
  --n-gpu-layers 99 \
  $EXPERT_PLACEMENT \
  --flash-attn on \
  --load-mode mmap \
  --threads "$THREADS" --threads-batch "$THREADS" \
  --jinja \
  --metrics \
  "$@"
