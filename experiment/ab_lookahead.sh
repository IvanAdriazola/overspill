#!/bin/bash
# (Git Bash, under the GPU lock) A/B the lookahead prefetch: for each k, start the server,
# run the decode/prefill bench (greedy, full text kept), stop the server.
# Usage: ab_lookahead.sh <out_prefix> k [k ...]
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/c/GIT/chatbot/.venv/Scripts/python.exe
BENCH=$(cygpath -m "$HERE/bench_openai.py")
OUT=$1; shift
for K in "$@"; do
  LOG="$HERE/${OUT}_k$K.log"
  echo "=== k=$K $(date +%H:%M:%S)" | tee "$LOG"
  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u ivan -- bash /mnt/c/GIT/Freetoken-colibri-experiment/experiment/serve_exp.sh "$K" > "$HERE/serve_exp_k$K.log" 2>&1 &
  SRV=$!
  t0=$(date +%s)
  until [ "$(curl -s -m 600 -o /dev/null -w '%{http_code}' http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')" = 200 ]; do
    kill -0 $SRV 2>/dev/null || { echo "server died" | tee -a "$LOG"; break; }
    [ $(( $(date +%s) - t0 )) -ge 1500 ] && { echo "not ready" | tee -a "$LOG"; break; }
    sleep 10
  done
  "$PY" "$BENCH" http://127.0.0.1:1919 Qwen3.6-35B-A3B 3 '{"_full_text": true, "chat_template_kwargs": {"enable_thinking": false}}' >> "$LOG" 2>&1
  "$PY" "$BENCH" http://127.0.0.1:1919 Qwen3.6-35B-A3B 2 '{"_long_prompt": true, "_full_text": true, "chat_template_kwargs": {"enable_thinking": false}}' >> "$LOG" 2>&1
  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/c/GIT/Freetoken-colibri-experiment/experiment/stop_server.sh
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  grep '^{' "$LOG"
done
