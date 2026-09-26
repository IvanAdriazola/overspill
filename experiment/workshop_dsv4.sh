#!/bin/bash
# (Git Bash, under the GPU lock) Workshop benchmark tasks on DeepSeek-V4-Flash REAP-150B through the disk tier.
# Usage: workshop_dsv4.sh task:flow [task:flow ...]
HERE=$(cd "$(dirname "$0")" && pwd)
LOG="$HERE/workshop_dsv4.log"
echo "=== workshop on dsv4 $(date +%H:%M:%S)" | tee "$LOG"
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u ivan -- bash /mnt/c/GIT/Freetoken-colibri-experiment/experiment/serve_dsv4.sh \
  --num-tokens 8192 --cuda-graph-max-bs 1 --max-running-requests 1 > "$HERE/serve_workshop_dsv4.log" 2>&1 &
SRV=$!
t0=$(date +%s)
until [ "$(curl -s -m 1200 -o /dev/null -w '%{http_code}' http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"dsv4-reap","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')" = 200 ]; do
  kill -0 $SRV 2>/dev/null || { echo "server died" | tee -a "$LOG"; exit 1; }
  grep -q -E "worker exited|worker is gone" "$HERE/serve_workshop_dsv4.log" && { echo "backend died" | tee -a "$LOG"; break; }
  sleep 10
done
echo "ready after $(( $(date +%s) - t0 ))s" | tee -a "$LOG"
cd /c/GIT/chatbot/benchmark
for pair in "$@"; do
  tarea=${pair%%:*}; flow=${pair##*:}
  echo "=== $tarea ($flow) $(date +%H:%M:%S)" | tee -a "$LOG"
  C:/GIT/chatbot/.venv/Scripts/python.exe correr_matriz.py --sistemas chatservice --modelos freetoken-dsv4-reap \
    --tareas "$tarea" --modo-chatservice "$flow" 2>&1 | grep -E "exito=|Resultados guardados|Traceback|Error" | tee -a "$LOG"
done
echo "=== done $(date +%H:%M:%S)" | tee -a "$LOG"
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/c/GIT/Freetoken-colibri-experiment/experiment/stop_server.sh
kill $SRV 2>/dev/null
