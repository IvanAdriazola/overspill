#!/bin/bash
# (Git Bash, under the GPU lock) llama.cpp INSIDE WSL on Qwen3.8-Flash-Next UD-IQ4_XS: same VM, RAM cap, disk and
# prompts as bench_flashnext.sh, so the two engines are compared like for like.
# Usage: bench_llamacpp_wsl.sh <label> [extra llama-server args, e.g. --n-cpu-moe 48 -ub 2048]
# Env: LLAMA_GGUF (WSL path of the *-00001-of-0000N.gguf), passed via WSLENV.
HERE=$(cd "$(dirname "$0")" && pwd)
PY=C:/GIT/chatbot/.venv/Scripts/python.exe
BENCH=$(cygpath -m "$HERE/bench_openai.py")
GGUF=${LLAMA_GGUF:-/mnt/wsl/gmodels/gguf/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}
LABEL=$1; shift
LOG="$HERE/llamacpp_wsl_$LABEL.log"
echo "=== llama.cpp (WSL) $LABEL $(date +%H:%M:%S) gguf: $GGUF args: $*" | tee "$LOG"
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u ivan -- env ALIAS=flashnext PORT=1920 \
  bash /mnt/c/GIT/Freetoken-colibri-experiment/experiment/llamacpp/serve_llamacpp.sh "$GGUF" --no-mmproj "$@" \
  > "$HERE/serve_llamacpp_wsl_$LABEL.log" 2>&1 &
SRV=$!
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/c/GIT/Freetoken-colibri-experiment/experiment/monitor_io.sh > "$HERE/monitor_llamacpp_wsl_$LABEL.csv" 2>&1 &
MON=$!
t0=$(date +%s)
until [ "$(curl -s -m 1200 -o /dev/null -w '%{http_code}' http://127.0.0.1:1920/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"flashnext","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')" = 200 ]; do
  kill -0 $SRV 2>/dev/null || { echo "server died" | tee -a "$LOG"; break; }
  [ $(( $(date +%s) - t0 )) -ge 2400 ] && { echo "not ready" | tee -a "$LOG"; break; }
  sleep 10
done
echo "ready after $(( $(date +%s) - t0 ))s" | tee -a "$LOG"
if kill -0 $SRV 2>/dev/null; then
  "$PY" "$BENCH" http://127.0.0.1:1920 flashnext 1 '{"max_tokens": 128, "_full_text": true}' 2>&1 | tee -a "$LOG"
  "$PY" "$BENCH" http://127.0.0.1:1920 flashnext 1 '{"max_tokens": 400, "_full_text": true, "_prompt": "Write a Python function merge_intervals(intervals) that takes a list of [start, end] pairs and returns the merged, sorted list of non-overlapping intervals. Give only the code and one sentence on its time complexity."}' 2>&1 | tee -a "$LOG"
  "$PY" "$BENCH" http://127.0.0.1:1920 flashnext 1 '{"_long_prompt": true, "_full_text": true}' 2>&1 | tee -a "$LOG"
fi
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash -c 'pkill -INT -x llama-server; for i in $(seq 1 20); do pgrep -x llama-server >/dev/null || break; sleep 1; done; pkill -KILL -x llama-server; pkill -f monitor_io.s[h]; true'
kill $SRV $MON 2>/dev/null
grep -E '^\{' "$LOG" | sed -E 's/.*"prompt_tokens": ([0-9]+).*"ttft_s": ([0-9.]+).*"decode_tok_s": ([0-9.]+).*/prompt=\1 ttft=\2 dec=\3/'
