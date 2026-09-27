#!/bin/bash
# (Git Bash) Native Windows: Overspill vs llama.cpp at simulated RAM sizes. For each cap a RAM hog locks (64 - cap) GiB
# of physical memory (win/ram_hog.py: VirtualLock, not pageable, not usable as cache), so Windows and both engines
# really have ~cap GB - like a smaller PC. Each engine/mode gets a discarded warm-up run, then a measured run.
# Usage: win/matrix_native.sh [caps...]   (default: 32 16 48)
HERE=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$HERE/.." && pwd)
PY=C:/GIT/chatbot/.venv/Scripts/python.exe
LOCK="C:/GIT/chatbot/benchmark/_with_gpu_lock.py"
BASH_EXE="C:\Program Files\Git\bin\bash.exe"
SUMMARY="$HERE/results/matrix_native.log"
CAPS=("$@"); [ ${#CAPS[@]} -eq 0 ] && CAPS=(32 16 48)
export GPU_LOCK_MAX_MINUTES=150
LLAMA_ARGS=(--n-cpu-moe 48 -ub 2048)                       # best native llama.cpp config at 64 GB
OS_CPU_ARGS=(cpu --cache-type naive --memory-ratio 0.97)    # all-CPU + bf16 LM head in VRAM (needs FT_HEAD_HOST=0)

run() {  # run <label> <script> [args] -- under the GPU lock
  local label=$1; shift
  MSYS_NO_PATHCONV=1 "$PY" "$LOCK" "native-$label" -- "$BASH_EXE" "$@" > /dev/null 2>&1
}
summ() {  # summ <cap> <what> <log>
  { echo "cap=${1}GB $2 ($(basename "$3"))"; grep -E "died|not ready|ready after" "$3";
    grep -E '^\{' "$3" | sed -E 's/.*"prompt_tokens": ([0-9]+).*"ttft_s": ([0-9.]+).*"decode_tok_s": ([0-9.]+).*/  prompt=\1 ttft=\2 dec=\3/'; } | tee -a "$SUMMARY"
}
avail() { powershell.exe -NoProfile -Command "[math]::Round((Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes/1024,1)" | tr -d '\r'; }

echo "=== native matrix $(date '+%F %H:%M') caps: ${CAPS[*]} | llama.cpp ${LLAMA_ARGS[*]} | overspill ${OS_CPU_ARGS[*]} (head in VRAM) and mixed (pin budget 40% of cap)" | tee -a "$SUMMARY"
cd "$HERE"
for cap in "${CAPS[@]}"; do
  hog=$(( 64 - cap ))
  HOGLOG=$(mktemp)
  if [ "$hog" -gt 0 ]; then
    "$REPO/.venv-win/Scripts/python.exe" "$HERE/win/ram_hog.py" "$hog" > "$HOGLOG" 2>&1 &
    HOG=$!
    for i in $(seq 1 120); do grep -qE "LOCKED|failed" "$HOGLOG" && break; sleep 2; done
    if ! grep -q LOCKED "$HOGLOG"; then echo "cap=${cap}GB: RAM hog failed: $(cat "$HOGLOG")" | tee -a "$SUMMARY"; kill $HOG 2>/dev/null; continue; fi
  fi
  echo "cap=${cap}GB: hog $(cat "$HOGLOG" | tail -1), available now $(avail) GB" | tee -a "$SUMMARY"

  # Overspill all-CPU, LM head in VRAM
  export FT_HEAD_HOST=0; unset FREETOKEN_PIN_BUDGET_GB
  run "os-cpu-warm-$cap" win/bench_flashnext_win.sh "N${cap}_os_cpu_warmup" "${OS_CPU_ARGS[@]}"
  run "os-cpu-$cap"      win/bench_flashnext_win.sh "N${cap}_os_cpu" "${OS_CPU_ARGS[@]}"
  summ "$cap" "overspill all-CPU, head in VRAM" "flashnext_win_N${cap}_os_cpu.log"
  # Overspill mixed, pin budget scaled to the simulated RAM (a real cap-GB box would compute 40% of cap)
  unset FT_HEAD_HOST; export FREETOKEN_PIN_BUDGET_GB=$(awk -v c="$cap" 'BEGIN{printf "%.1f", c*0.4}')
  run "os-mixed-warm-$cap" win/bench_flashnext_win.sh "N${cap}_os_mixed_warmup" mixed
  run "os-mixed-$cap"      win/bench_flashnext_win.sh "N${cap}_os_mixed" mixed
  summ "$cap" "overspill mixed (pin ${FREETOKEN_PIN_BUDGET_GB} GB)" "flashnext_win_N${cap}_os_mixed.log"
  unset FREETOKEN_PIN_BUDGET_GB
  # llama.cpp native
  run "ll-warm-$cap" bench_llamacpp_win.sh "N${cap}_warmup" "${LLAMA_ARGS[@]}"
  run "ll-$cap"      bench_llamacpp_win.sh "N${cap}" "${LLAMA_ARGS[@]}"
  summ "$cap" "llama.cpp ${LLAMA_ARGS[*]}" "llamacpp_win_N${cap}.log"

  [ -n "${HOG:-}" ] && kill $HOG 2>/dev/null; HOG=; rm -f "$HOGLOG"; sleep 5
done
echo "=== native matrix done $(date '+%F %H:%M')" | tee -a "$SUMMARY"
