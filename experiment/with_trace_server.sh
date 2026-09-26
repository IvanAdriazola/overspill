#!/bin/bash
# (Git Bash) Start the tracing FreeToken server in WSL, wait for a real completion,
# run the given command, always stop the server. Run under the GPU lock:
#   python C:/GIT/chatbot/benchmark/_with_gpu_lock.py trace-exp -- "C:\Program Files\Git\bin\bash.exe" with_trace_server.sh <trace_name> <cmd> [args...]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
TRACE_NAME=$1; shift
PORT=1919
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u ivan -- bash /mnt/c/GIT/Freetoken-colibri-experiment/experiment/serve_trace.sh "$TRACE_NAME" > "$HERE/serve_trace.log" 2>&1 &
SRV_PID=$!
cleanup() {
  # SIGINT lets Python run atexit (final trace flush); then force.
  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/c/GIT/Freetoken-colibri-experiment/experiment/stop_server.sh
  kill "$SRV_PID" 2>/dev/null
}
trap cleanup EXIT
t0=$(date +%s)
until [ "$(curl -s -m 600 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3.6-35B-A3B","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')" = 200 ]; do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "with_trace_server: server exited before ready"; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -ge 1500 ]; then echo "with_trace_server: not ready after 1500s"; exit 1; fi
  sleep 10
done
echo "with_trace_server: ready after $(( $(date +%s) - t0 ))s"
"$@"
sleep 8  # let the tracer's 5 s flush thread write the tail before the kill
