#!/bin/bash
# (inside WSL, as ivan) Convert nvidia/MiniMax-M3-NVFP4 (250 GB, on E:\AIModels\m3_src since 2026-09-29) to FTW for
# NATIVE Windows Overspill: written straight to C:\AIModels\m3_ftw (NTFS, same Kingston NVMe as the llama.cpp GGUF on
# D: - like-for-like disk). C: cannot hold all ~245 GB: when it gets below MIN_FREE_GB, the oldest finished shards
# move to G:\AIModels\m3_ftw_spill. No symlinks (Windows cannot follow WSL's): at the end the index gets the spilled
# shards' absolute Windows paths - the reader resolves shards with os.path.join(dir, file), which keeps an absolute
# file as is.
set -e
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda
export TVM_FFI_CUDA_ARCH_LIST=8.6 TORCH_CUDA_ARCH_LIST=8.6 FLASHINFER_CUDA_ARCH_LIST=8.6
export PYTHONPATH=$HOME/src/freetoken-exp/python
export FREETOKEN_SKIP_BANK_PIN=1
SRC=/mnt/e/AIModels/m3_src
OUT=/mnt/c/AIModels/m3_ftw
SPILL=/mnt/g/AIModels/m3_ftw_spill
MIN_FREE_GB=${MIN_FREE_GB:-25}

mkdir -p "$OUT" "$SPILL"
echo "source: $(ls "$SRC"/*.safetensors | wc -l) safetensors"

# spill watcher: keep >= MIN_FREE_GB free on C: by moving the oldest finished shards to G:
(
  while sleep 20; do
    free=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9)
    [ "$free" -ge "$MIN_FREE_GB" ] && continue
    newest=$(ls -t "$OUT"/freetoken-*.ftw 2>/dev/null | head -1)
    for f in $(ls -tr "$OUT"/freetoken-*.ftw 2>/dev/null); do
      [ "$f" = "$newest" ] && break          # never the shard being written
      mv "$f" "$SPILL/" && echo "spilled $(basename "$f") to G: (C: free ${free} GB)"
      break
    done
  done
) &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT

cd ~/ft
.venv/bin/ft checkpoint --model "$SRC" --out "$OUT"
kill $WATCH 2>/dev/null || true

# point the index at the spilled shards (absolute Windows paths)
python3 - "$OUT" "$SPILL" <<'EOF'
import json, os, sys
out, spill = sys.argv[1], sys.argv[2]
p = os.path.join(out, "freetoken_weight.json")
idx = json.load(open(p))
moved = 0
for sh in idx["shards"]:
    if not os.path.exists(os.path.join(out, sh["file"])) and os.path.exists(os.path.join(spill, sh["file"])):
        sh["file"] = "G:\\AIModels\\m3_ftw_spill\\" + sh["file"]
        moved += 1
missing = [sh["file"] for sh in idx["shards"]
           if not sh["file"].startswith("G:") and not os.path.exists(os.path.join(out, sh["file"]))]
json.dump(idx, open(p, "w"))
print(f"index: {len(idx['shards'])} shards, {moved} on G:, missing: {missing}")
EOF
du -sh "$OUT" "$SPILL"
