#!/bin/bash
# (Git Bash) Overspill vs llama.cpp, both inside WSL, both reading from the same ext4 disk on G: (gmodels helper
# distro), at several WSL RAM caps. Each engine gets a discarded warm-up run first, then measured runs.
# Usage: matrix_wsl.sh [caps...]   (default: 52 48 32 16)
# Changes C:\Users\Asus\.wslconfig memory= per cap (restores the starting value at the end) and restarts WSL.
HERE=$(cd "$(dirname "$0")" && pwd)
PY=C:/GIT/chatbot/.venv/Scripts/python.exe
LOCK="C:/GIT/chatbot/benchmark/_with_gpu_lock.py"
BASH_EXE="C:\Program Files\Git\bin\bash.exe"
CFG=/c/Users/Asus/.wslconfig
ORIG=$(grep -E '^memory=' "$CFG")
SUMMARY="$HERE/results/matrix_wsl.log"
CAPS=("$@"); [ ${#CAPS[@]} -eq 0 ] && CAPS=(52 48 32 16)
export FT_MODEL_DIR=/mnt/wsl/gmodels/flashnext_ftw
export LLAMA_GGUF=/mnt/wsl/gmodels/gguf/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export WSLENV=FT_MODEL_DIR:LLAMA_GGUF
export GPU_LOCK_MAX_MINUTES=150
LLAMA_ARGS=(--n-cpu-moe 48 -ub 2048)   # best native-Windows config (sweep: n-cpu-moe 40-48, ub 512/2048, threads 8-12)

run() {  # run <label> <script> [args] -- under the GPU lock, then append its numbers to the summary
  local label=$1; shift
  MSYS_NO_PATHCONV=1 "$PY" "$LOCK" "matrix-$label" -- "$BASH_EXE" "$@" > /dev/null 2>&1
}
summ() {  # summ <cap> <engine/mode> <log>
  { echo "cap=${1}GB $2 ($(basename "$3"))"; grep -E "died|not ready|ready after" "$3";
    grep -E '^\{' "$3" | sed -E 's/.*"prompt_tokens": ([0-9]+).*"ttft_s": ([0-9.]+).*"decode_tok_s": ([0-9.]+).*/  prompt=\1 ttft=\2 dec=\3/'; } | tee -a "$SUMMARY"
}

echo "=== matrix $(date '+%F %H:%M') caps: ${CAPS[*]} llama: ${LLAMA_ARGS[*]}" | tee -a "$SUMMARY"
for cap in "${CAPS[@]}"; do
  sed -i "s/^memory=.*/memory=${cap}GB/" "$CFG"
  wsl.exe --shutdown; sleep 8
  bash "$HERE/gmodels/start_gmodels.sh" > /dev/null 2>&1
  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash -c 'free -g | sed -n 2p' | tee -a "$SUMMARY"
  cd "$HERE"
  # Overspill: warm-up (mixed), then mixed and cpu measured
  run "os-warm-$cap"  bench_flashnext.sh "M${cap}_os_warmup" mixed
  run "os-mixed-$cap" bench_flashnext.sh "M${cap}_os_mixed" mixed;  summ "$cap" "overspill mixed" "flashnext_M${cap}_os_mixed.log"
  run "os-cpu-$cap"   bench_flashnext.sh "M${cap}_os_cpu" cpu;      summ "$cap" "overspill cpu"   "flashnext_M${cap}_os_cpu.log"
  # llama.cpp: warm-up, then measured
  run "ll-warm-$cap"  bench_llamacpp_wsl.sh "M${cap}_warmup" "${LLAMA_ARGS[@]}"
  run "ll-$cap"       bench_llamacpp_wsl.sh "M${cap}" "${LLAMA_ARGS[@]}"; summ "$cap" "llama.cpp ${LLAMA_ARGS[*]}" "llamacpp_wsl_M${cap}.log"
done
sed -i "s/^memory=.*/$ORIG/" "$CFG"
wsl.exe --shutdown; sleep 8
bash "$HERE/gmodels/start_gmodels.sh" > /dev/null 2>&1
echo "=== matrix done $(date '+%F %H:%M'), .wslconfig back to $ORIG" | tee -a "$SUMMARY"
