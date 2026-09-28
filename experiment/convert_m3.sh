#!/bin/bash
# (inside WSL, as ivan) Convert nvidia/MiniMax-M3-NVFP4 (250 GB, split across D: and F:) to FTW on the G: helper disk
# (/mnt/wsl/gmodels). If G: runs low, finished shards are moved to the D: helper disk (/mnt/wsl/dmodels) and
# replaced by symlinks, so the FTW still loads from one directory.
set -e
export PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_HOME=/usr/local/cuda
export TVM_FFI_CUDA_ARCH_LIST=8.6 TORCH_CUDA_ARCH_LIST=8.6 FLASHINFER_CUDA_ARCH_LIST=8.6
export PYTHONPATH=$HOME/src/freetoken-exp/python
SRC=$HOME/models/m3_src_merged
OUT=/mnt/wsl/gmodels/m3_ftw
SPILL=/mnt/wsl/dmodels/m3_ftw_spill
MIN_FREE_GB=${MIN_FREE_GB:-14}

# one source directory: symlinks to the files on D: and F:
mkdir -p "$SRC" "$SPILL"
for part in /mnt/d/AIModels/m3_src /mnt/f/AIModels/m3_src; do
  (cd "$part" && find . -type f -not -path './.cache/*' -printf '%P\n') | while read -r f; do
    mkdir -p "$SRC/$(dirname "$f")"; ln -sf "$part/$f" "$SRC/$f"
  done
done
echo "source files linked: $(find "$SRC" -type l | wc -l)"

# spill watcher: keep >= MIN_FREE_GB free on G: by moving the oldest finished shards to D:
(
  while sleep 20; do
    free=$(df -BG --output=avail /mnt/wsl/gmodels | tail -1 | tr -dc 0-9)
    [ "$free" -ge "$MIN_FREE_GB" ] && continue
    newest=$(ls -t "$OUT"/freetoken-*.ftw 2>/dev/null | head -1)
    for f in $(ls -tr "$OUT"/freetoken-*.ftw 2>/dev/null); do
      [ "$f" = "$newest" ] && break          # never the shard being written
      [ -L "$f" ] && continue
      mv "$f" "$SPILL/" && ln -s "$SPILL/$(basename "$f")" "$f" && echo "spilled $(basename "$f") to D:"
      break
    done
  done
) &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT

cd ~/ft
.venv/bin/ft checkpoint --model "$SRC" --out "$OUT"
du -shL "$OUT"; ls "$OUT" | wc -l; ls -l "$OUT" | grep -c -- '->' | xargs echo "spilled shards:"
