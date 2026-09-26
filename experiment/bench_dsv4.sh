#!/bin/bash
# (Git Bash, under the GPU lock) Start the DSV4 server, run Colibri's exact bench plan, stop.
# Colibri cold results for comparison: short 1.17 tok/s (ttft 25.4s), coding 1.20 (17.5s), long 6446 tok ttft 1565s / 1.12 tok/s.
# Usage: bench_dsv4.sh <label> [extra ft serve args]
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/c/GIT/chatbot/.venv/Scripts/python.exe
BENCH=/c/GIT/chatbot/benchmark/moe_engines/bench_openai.py
LABEL=$1; shift
LOG="$HERE/dsv4_$LABEL.log"
echo "=== $LABEL $(date +%H:%M:%S) args: $*" | tee "$LOG"
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u ivan -- bash /mnt/c/GIT/Freetoken-colibri-experiment/experiment/serve_dsv4.sh "$@" > "$HERE/serve_dsv4_$LABEL.log" 2>&1 &
SRV=$!
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/c/GIT/Freetoken-colibri-experiment/experiment/monitor_io.sh > "$HERE/monitor_dsv4_$LABEL.csv" 2>&1 &
MON=$!
t0=$(date +%s)
until [ "$(curl -s -m 1200 -o /dev/null -w '%{http_code}' http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"dsv4-reap","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')" = 200 ]; do
  kill -0 $SRV 2>/dev/null || { echo "server died" | tee -a "$LOG"; break; }
  [ $(( $(date +%s) - t0 )) -ge 2400 ] && { echo "not ready" | tee -a "$LOG"; break; }
  sleep 10
done
echo "ready after $(( $(date +%s) - t0 ))s" | tee -a "$LOG"
if kill -0 $SRV 2>/dev/null; then
  "$PY" "$BENCH" http://127.0.0.1:1919 dsv4-reap 1 '{"max_tokens": 128, "_full_text": true}' 2>&1 | tee -a "$LOG"
  "$PY" "$BENCH" http://127.0.0.1:1919 dsv4-reap 1 '{"max_tokens": 400, "_full_text": true, "_prompt": "Write a Python function merge_intervals(intervals) that takes a list of [start, end] pairs and returns the merged, sorted list of non-overlapping intervals. Give only the code and one sentence on its time complexity."}' 2>&1 | tee -a "$LOG"
  "$PY" "$BENCH" http://127.0.0.1:1919 dsv4-reap 1 '{"_long_prompt": true, "_full_text": true}' 2>&1 | tee -a "$LOG"
fi
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/c/GIT/Freetoken-colibri-experiment/experiment/stop_server.sh
kill $SRV $MON 2>/dev/null
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash -c 'pkill -f monitor_io.s[h]; true'
