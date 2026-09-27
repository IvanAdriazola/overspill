#!/bin/bash
# (Git Bash, under the GPU lock) Find llama.cpp's best --n-cpu-moe for Qwen3.8-Flash-Next on native Windows.
# For each N: start, warm up, one 128-token decode, stop. Stops going lower when the server fails to load (VRAM).
# Usage: sweep_llamacpp_win.sh "48 46 44 42 40 38" [extra args]
HERE=$(cd "$(dirname "$0")" && pwd)
PY=C:/GIT/chatbot/.venv/Scripts/python.exe
BENCH=$(cygpath -m "$HERE/bench_openai.py")
SRV_EXE="D:/tools/llama.cpp-b11205/llama-server.exe"
GGUF="D:/AIModels/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
NS=$1; shift
OUT="$HERE/sweep_llamacpp_win.log"
echo "=== sweep $(date +%H:%M:%S) Ns: $NS extra: $*" | tee -a "$OUT"
for N in $NS; do
  "$SRV_EXE" --model "$GGUF" --alias flashnext --host 127.0.0.1 --port 1920 --ctx-size 8192 --parallel 1 \
    --n-gpu-layers 99 --n-cpu-moe "$N" --flash-attn on --load-mode mmap --threads 11 --threads-batch 11 --jinja --no-mmproj "$@" \
    > "$HERE/sweep_serve_n$N.log" 2>&1 &
  SRV=$!; t0=$(date +%s); ok=1
  until [ "$(curl -s -m 1200 -o /dev/null -w '%{http_code}' http://127.0.0.1:1920/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"flashnext","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')" = 200 ]; do
    kill -0 $SRV 2>/dev/null || { ok=0; break; }
    [ $(( $(date +%s) - t0 )) -ge 1800 ] && { ok=0; break; }
    sleep 5
  done
  if [ $ok = 0 ]; then
    echo "n-cpu-moe=$N: FAILED to load ($(grep -i -m1 -E 'out of memory|failed|error' "$HERE/sweep_serve_n$N.log" | cut -c1-120))" | tee -a "$OUT"
    kill $SRV 2>/dev/null; break
  fi
  "$PY" "$BENCH" http://127.0.0.1:1920 flashnext 1 '{"max_tokens": 64}' >/dev/null 2>&1   # warm-up
  r=$("$PY" "$BENCH" http://127.0.0.1:1920 flashnext 1 '{"max_tokens": 128}' 2>&1 | grep '^{')
  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
  echo "n-cpu-moe=$N ready=$(( $(date +%s) - t0 ))s vram=$vram $(echo "$r" | sed -E 's/.*"decode_tok_s": ([0-9.]+).*/decode=\1 tok\/s/')" | tee -a "$OUT"
  taskkill //F //IM llama-server.exe >/dev/null 2>&1; kill $SRV 2>/dev/null; sleep 3
done
