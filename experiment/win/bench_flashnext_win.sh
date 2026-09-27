#!/bin/bash
# (Git Bash, under the GPU lock) Native-Windows Overspill on Qwen3.8-Flash-Next: same prompts as bench_flashnext.sh (WSL)
# and bench_llamacpp_win.sh (native llama.cpp). Logs Windows disk-read MB/s every 5 s.
# Usage: bench_flashnext_win.sh <label> [cpu|mixed] [extra ft serve args]
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=C:/GIT/chatbot/.venv/Scripts/python.exe
BENCH=$(cygpath -m "$HERE/bench_openai.py")
LABEL=$1; shift
LOG="$HERE/flashnext_win_$LABEL.log"
echo "=== Overspill (native Windows) $LABEL $(date +%H:%M:%S) args: $*" | tee "$LOG"
MSYS_NO_PATHCONV=1 cmd.exe /c "$(cygpath -w "$HERE/win/serve_flashnext_win.bat")" "$@" > "$HERE/serve_flashnext_win_$LABEL.log" 2>&1 &
SRV=$!
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$HERE/win/monitor_disk_win.ps1")" > "$HERE/monitor_flashnext_win_$LABEL.csv" 2>&1 &
MON=$!
t0=$(date +%s)
until [ "$(curl -s -m 1200 -o /dev/null -w '%{http_code}' http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"flashnext","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')" = 200 ]; do
  kill -0 $SRV 2>/dev/null || { echo "server died" | tee -a "$LOG"; break; }
  grep -q -E "worker exited|worker is gone|cannot be restarted|Traceback" "$HERE/serve_flashnext_win_$LABEL.log" && { echo "backend died" | tee -a "$LOG"; break; }
  [ $(( $(date +%s) - t0 )) -ge 2400 ] && { echo "not ready" | tee -a "$LOG"; break; }
  sleep 10
done
echo "ready after $(( $(date +%s) - t0 ))s" | tee -a "$LOG"
if kill -0 $SRV 2>/dev/null && ! grep -q -E "worker exited|worker is gone|Traceback" "$HERE/serve_flashnext_win_$LABEL.log"; then
if [ "$QUICK" = long ]; then  # QUICK=long: long-prompt TTFT only (32 output tokens)
  "$PY" "$BENCH" http://127.0.0.1:1919 flashnext 1 '{"_long_prompt": true, "max_tokens": 32, "_full_text": true}' 2>&1 | tee -a "$LOG"
elif [ -n "$QUICK" ]; then  # QUICK=1: decode tok/s only - 64 tokens on each short prompt, no long prompt (~2-4 min)
  "$PY" "$BENCH" http://127.0.0.1:1919 flashnext 1 '{"max_tokens": 64, "_full_text": true}' 2>&1 | tee -a "$LOG"
  "$PY" "$BENCH" http://127.0.0.1:1919 flashnext 1 '{"max_tokens": 64, "_full_text": true, "_prompt": "Write a Python function merge_intervals(intervals) that takes a list of [start, end] pairs and returns the merged, sorted list of non-overlapping intervals. Give only the code and one sentence on its time complexity."}' 2>&1 | tee -a "$LOG"
else
    "$PY" "$BENCH" http://127.0.0.1:1919 flashnext 1 '{"max_tokens": 128, "_full_text": true}' 2>&1 | tee -a "$LOG"
    "$PY" "$BENCH" http://127.0.0.1:1919 flashnext 1 '{"max_tokens": 400, "_full_text": true, "_prompt": "Write a Python function merge_intervals(intervals) that takes a list of [start, end] pairs and returns the merged, sorted list of non-overlapping intervals. Give only the code and one sentence on its time complexity."}' 2>&1 | tee -a "$LOG"
    "$PY" "$BENCH" http://127.0.0.1:1919 flashnext 1 '{"_long_prompt": true, "_full_text": true}' 2>&1 | tee -a "$LOG"
  fi
fi
# stop the server: ft.exe and its whole process tree (its multiprocessing workers have generic command lines,
# so only a tree kill catches them; killing the parent alone orphans them)
for pid in $(MSYS_NO_PATHCONV=1 tasklist.exe /FI "IMAGENAME eq ft.exe" /FO CSV /NH | grep -i ft.exe | cut -d, -f2 | tr -d '"'); do MSYS_NO_PATHCONV=1 taskkill.exe /F /T /PID "$pid" >/dev/null 2>&1; done
kill $SRV $MON 2>/dev/null
powershell.exe -NoProfile -Command 'Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "monitor_disk_win" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }' 2>/dev/null
grep -E '^\{' "$LOG" | sed -E 's/.*"prompt_tokens": ([0-9]+).*"ttft_s": ([0-9.]+).*"decode_tok_s": ([0-9.]+).*/prompt=\1 ttft=\2 dec=\3/'
