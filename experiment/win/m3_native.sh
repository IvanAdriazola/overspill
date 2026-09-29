#!/bin/bash
# (Git Bash) Native Windows, MiniMax-M3 (FTW 233 GiB / GGUF UD-IQ4_XS 194 GiB - ~3.5x the 64 GB of RAM): Overspill vs
# llama.cpp. Both on the same Kingston NVMe (FTW on C:, GGUF on D:, two partitions of one disk). Warm runs: each config
# gets a discarded warm-up (QUICK, decode only), then a measured run (the full prompt set, incl. the long prompt).
# Overspill = the published README settings (all-CPU experts, file banks, short prefill on the CPU) + native port.
# llama.cpp = its best MoE configs from Flash-Next/DeepSeek (all experts on the CPU, -ub 2048 first, then -ub 512).
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=C:/GIT/chatbot/.venv/Scripts/python.exe
LOCK="C:/GIT/chatbot/benchmark/_with_gpu_lock.py"
BASH_EXE="C:\Program Files\Git\bin\bash.exe"
SUMMARY="$HERE/results/m3_native.log"
export GPU_LOCK_MAX_MINUTES=240
run() { local label=$1; shift; MSYS_NO_PATHCONV=1 "$PY" "$LOCK" "m3-$label" -- "$BASH_EXE" "$@" > /dev/null 2>&1; }
summ() {
  { echo "$1 ($(basename "$2"))"; grep -E "died|not ready|ready after" "$2";
    grep -E '^\{' "$2" | sed -E 's/.*"prompt_tokens": ([0-9]+).*"ttft_s": ([0-9.]+).*"decode_tok_s": ([0-9.]+).*/  prompt=\1 ttft=\2 dec=\3/'; } | tee -a "$SUMMARY"
}
cd "$HERE"
echo "=== MiniMax-M3 native $(date '+%F %H:%M')" | tee -a "$SUMMARY"

# Overspill (native)
export FT_MODEL_DIR='C:\AIModels\m3_ftw' FT_FILE_BANKS=1 FT_EMBED_HOST=1 FT_HEAD_HOST=1 FT_CPU_PREFILL_MAX=256
QUICK=1 run os-warm win/bench_flashnext_win.sh m3_os_warmup cpu
run os win/bench_flashnext_win.sh m3_os cpu
summ "overspill all-CPU (README settings)" flashnext_win_m3_os.log
unset FT_MODEL_DIR FT_FILE_BANKS FT_EMBED_HOST FT_HEAD_HOST FT_CPU_PREFILL_MAX

# llama.cpp (native)
export LLAMA_GGUF_WIN="D:/AIModels/m3_gguf/UD-IQ4_XS/MiniMax-M3-UD-IQ4_XS-00001-of-00006.gguf"
run ll-warm bench_llamacpp_win.sh m3_warmup --cpu-moe -ub 2048
for cfg in "cpumoe_ub2048:--cpu-moe -ub 2048" "cpumoe_ub512:--cpu-moe -ub 512"; do
  name=${cfg%%:*}; args=${cfg#*:}
  run "ll-$name" bench_llamacpp_win.sh "m3_$name" $args
  summ "llama.cpp $args" "llamacpp_win_m3_$name.log"
done
echo "=== done $(date '+%F %H:%M')" | tee -a "$SUMMARY"
