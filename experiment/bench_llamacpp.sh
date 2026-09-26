#!/bin/bash
# (Git Bash, under the GPU lock) llama.cpp on the same DSV4 REAP-150B model (MXFP4_MOE GGUF, experts mmap'd
# on the CPU, the rest on the GPU): same three prompts as bench_dsv4.sh / Colibri.
# Usage: bench_llamacpp.sh <label> <gguf path in WSL> [extra llama-server args]
HERE=$(cd "$(dirname "$0")" && pwd)
PY=C:/GIT/chatbot/.venv/Scripts/python.exe
BENCH=C:/GIT/chatbot/benchmark/moe_engines/bench_openai.py
LABEL=$1; GGUF=$2; shift 2
LOG="$HERE/llamacpp_$LABEL.log"
echo "=== llama.cpp $LABEL $(date +%H:%M:%S) $GGUF args: $*" | tee "$LOG"
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u ivan -- bash /mnt/c/GIT/Freetoken-colibri-experiment/experiment/llamacpp/serve_llamacpp.sh "$GGUF" "$@" > "$HERE/serve_llamacpp_$LABEL.log" 2>&1 &
SRV=$!
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/c/GIT/Freetoken-colibri-experiment/experiment/monitor_io.sh > "$HERE/monitor_llamacpp_$LABEL.csv" 2>&1 &
MON=$!
MODEL=$(basename "$GGUF" .gguf)
t0=$(date +%s)
until [ "$(curl -s -m 1200 -o /dev/null -w '%{http_code}' http://127.0.0.1:1920/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}")" = 200 ]; do
  kill -0 $SRV 2>/dev/null || { echo "server died" | tee -a "$LOG"; break; }
  [ $(( $(date +%s) - t0 )) -ge 2400 ] && { echo "not ready" | tee -a "$LOG"; break; }
  sleep 10
done
echo "ready after $(( $(date +%s) - t0 ))s" | tee -a "$LOG"
if kill -0 $SRV 2>/dev/null; then
  "$PY" "$BENCH" http://127.0.0.1:1920 "$MODEL" 1 '{"max_tokens": 128, "_full_text": true}' 2>&1 | tee -a "$LOG"
  "$PY" "$BENCH" http://127.0.0.1:1920 "$MODEL" 1 '{"max_tokens": 400, "_full_text": true, "_prompt": "Write a Python function merge_intervals(intervals) that takes a list of [start, end] pairs and returns the merged, sorted list of non-overlapping intervals. Give only the code and one sentence on its time complexity."}' 2>&1 | tee -a "$LOG"
  "$PY" "$BENCH" http://127.0.0.1:1920 "$MODEL" 1 '{"_long_prompt": true, "_full_text": true}' 2>&1 | tee -a "$LOG"
fi
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash -c 'pkill -INT -f llama-serve[r]; sleep 5; pkill -KILL -f llama-serve[r]; pkill -f monitor_io.s[h]; true'
kill $SRV $MON 2>/dev/null
