#!/bin/bash
# (Git Bash, under the GPU lock) Correctness A/B of the tier layer on Qwen3.6: same strategy, flags on/off.
HERE=$(cd "$(dirname "$0")" && pwd)
PY=C:/GIT/chatbot/.venv/Scripts/python.exe
BENCH=C:/GIT/chatbot/benchmark/moe_engines/bench_openai.py
for MODE in "$@"; do
  LOG="$HERE/qwentier_$MODE.log"
  echo "=== $MODE $(date +%H:%M:%S)" | tee "$LOG"
  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u ivan -- bash /mnt/c/GIT/Freetoken-colibri-experiment/experiment/serve_qwen_tier.sh "$MODE" > "$HERE/serve_qwentier_$MODE.log" 2>&1 &
  SRV=$!
  t0=$(date +%s)
  until [ "$(curl -s -m 600 -o /dev/null -w '%{http_code}' http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')" = 200 ]; do
    kill -0 $SRV 2>/dev/null || { echo "server died" | tee -a "$LOG"; break; }
    grep -q -E "worker exited|worker is gone" "$HERE/serve_qwentier_$MODE.log" && { echo "backend died" | tee -a "$LOG"; break; }
    [ $(( $(date +%s) - t0 )) -ge 1500 ] && break
    sleep 10
  done
  "$PY" "$BENCH" http://127.0.0.1:1919 Qwen3.6-35B-A3B 3 '{"_full_text": true, "chat_template_kwargs": {"enable_thinking": false}}' >> "$LOG" 2>&1
  "$PY" "$BENCH" http://127.0.0.1:1919 Qwen3.6-35B-A3B 1 '{"_long_prompt": true, "_full_text": true, "chat_template_kwargs": {"enable_thinking": false}}' >> "$LOG" 2>&1
  "$PY" "$BENCH" http://127.0.0.1:1919 Qwen3.6-35B-A3B 1 '{"max_tokens": 400, "_full_text": true, "chat_template_kwargs": {"enable_thinking": false}, "_prompt": "Write a Python function merge_intervals(intervals) that takes a list of [start, end] pairs and returns the merged, sorted list of non-overlapping intervals. Give only the code and one sentence on its time complexity."}' >> "$LOG" 2>&1
  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/c/GIT/Freetoken-colibri-experiment/experiment/stop_server.sh
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
  grep '^{' "$LOG" | sed -E 's/.*"mode": "([a-z_]+)".*"ttft_s": ([0-9.]+).*"decode_tok_s": ([0-9.]+).*/\1 ttft=\2 dec=\3/'
done
