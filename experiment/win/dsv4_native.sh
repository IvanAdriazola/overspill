#!/bin/bash
# (Git Bash) Native Windows, DeepSeek-V4-Flash REAP-150B (85 GB, bigger than the 64 GB of RAM): Overspill vs llama.cpp.
# Both models on G: (same NVMe). Warm runs: each config gets a discarded warm-up, then a measured run.
# Overspill = the published README settings (all-CPU, SWA ratio 0.7, short prefill on the CPU) + native-Windows port.
# llama.cpp = a small sweep so it runs at its best: all experts on CPU at -ub 512 / 2048, and --n-cpu-moe 42.
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=C:/GIT/chatbot/.venv/Scripts/python.exe
LOCK="C:/GIT/chatbot/benchmark/_with_gpu_lock.py"
BASH_EXE="C:\Program Files\Git\bin\bash.exe"
SUMMARY="$HERE/results/dsv4_native.log"
export GPU_LOCK_MAX_MINUTES=150
run() { local label=$1; shift; MSYS_NO_PATHCONV=1 "$PY" "$LOCK" "dsv4-$label" -- "$BASH_EXE" "$@" > /dev/null 2>&1; }
summ() {
  { echo "$1 ($(basename "$2"))"; grep -E "died|not ready|ready after" "$2";
    grep -E '^\{' "$2" | sed -E 's/.*"prompt_tokens": ([0-9]+).*"ttft_s": ([0-9.]+).*"decode_tok_s": ([0-9.]+).*/  prompt=\1 ttft=\2 dec=\3/'; } | tee -a "$SUMMARY"
}
cd "$HERE"
echo "=== DeepSeek-V4-Flash REAP-150B native $(date '+%F %H:%M')" | tee -a "$SUMMARY"

# Overspill (native)
export FT_MODEL_DIR='G:\AIModels\dsv4_reap150b_ftw' FT_FILE_BANKS=1 FT_EMBED_HOST=1 FT_HEAD_HOST=1 FT_SWA_RATIO=0.7 FT_CPU_PREFILL_MAX=256
run os-warm win/bench_flashnext_win.sh dsv4_os_warmup cpu --num-tokens 8192
run os      win/bench_flashnext_win.sh dsv4_os cpu --num-tokens 8192
summ "overspill all-CPU (README settings)" flashnext_win_dsv4_os.log
unset FT_MODEL_DIR FT_FILE_BANKS FT_EMBED_HOST FT_HEAD_HOST FT_SWA_RATIO FT_CPU_PREFILL_MAX

# llama.cpp (native), same GGUF as the README's llama.cpp row
export LLAMA_GGUF_WIN="G:/AIModels/dsv4/DeepSeek-V4-Flash-0731-reap-150b-MXFP4_MOE.gguf"
run ll-warm bench_llamacpp_win.sh dsv4_warmup --cpu-moe
# the head-to-head first (-ub 2048 was llama.cpp's clear best on Flash-Next); the other two are confirmation runs
for cfg in "cpumoe_ub2048:--cpu-moe -ub 2048" "cpumoe_ub512:--cpu-moe -ub 512" "ncm42_ub2048:--n-cpu-moe 42 -ub 2048"; do
  name=${cfg%%:*}; args=${cfg#*:}
  run "ll-$name" bench_llamacpp_win.sh "dsv4_$name" $args
  summ "llama.cpp $args" "llamacpp_win_dsv4_$name.log"
done
echo "=== done $(date '+%F %H:%M')" | tee -a "$SUMMARY"
