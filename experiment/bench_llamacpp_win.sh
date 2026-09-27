#!/bin/bash
# (Git Bash, under the GPU lock) Native-Windows llama.cpp on Qwen3.8-Flash-Next UD-IQ4_XS, reading the GGUF straight
# from D: (no WSL). Same prompts as bench_flashnext.sh. Logs Windows disk-read MB/s every 5 s.
# Usage: bench_llamacpp_win.sh <label> [extra llama-server args, e.g. --n-cpu-moe 45 -ub 2048]
HERE=$(cd "$(dirname "$0")" && pwd)
PY=C:/GIT/chatbot/.venv/Scripts/python.exe
BENCH=$(cygpath -m "$HERE/bench_openai.py")
SRV_EXE="D:/tools/llama.cpp-b11205/llama-server.exe"
GGUF="D:/AIModels/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
LABEL=$1; shift
LOG="$HERE/llamacpp_win_$LABEL.log"
echo "=== llama.cpp (native Windows b11205) $LABEL $(date +%H:%M:%S) args: $*" | tee "$LOG"
"$SRV_EXE" --model "$GGUF" --alias flashnext --host 127.0.0.1 --port 1920 --ctx-size 8192 --parallel 1 \
  --n-gpu-layers 99 --flash-attn on --load-mode mmap --threads 11 --threads-batch 11 --jinja --no-mmproj "$@" \
  > "$HERE/serve_llamacpp_win_$LABEL.log" 2>&1 &
SRV=$!
typeperf "\PhysicalDisk(_Total)\Disk Read Bytes/sec" -si 5 > "$HERE/monitor_llamacpp_win_$LABEL.csv" 2>&1 &
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
taskkill //F //PID $(tasklist //FI "IMAGENAME eq llama-server.exe" //FO CSV //NH | head -1 | cut -d, -f2 | tr -d '"') >/dev/null 2>&1
kill $SRV $MON 2>/dev/null
grep -E '^\{' "$LOG" | sed -E 's/.*"prompt_tokens": ([0-9]+).*"ttft_s": ([0-9.]+).*"decode_tok_s": ([0-9.]+).*/prompt=\1 ttft=\2 dec=\3/'
