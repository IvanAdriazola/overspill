#!/bin/bash
# (inside WSL) Download nvidia/Qwen3.8-Flash-Next-NVFP4 (132.7 GB) to D: (/mnt/d). Every HF cache lives on D:
# and the xet chunk cache is off: D: has ~136 GB free, so there is no room for a second copy.
set -e
export HF_HOME=/mnt/d/AIModels/.hf_home HF_XET_CHUNK_CACHE_SIZE_BYTES=0
unset HF_HUB_DISABLE_XET
mkdir -p /mnt/d/AIModels/Qwen3.8-Flash-Next-NVFP4
~/ft/.venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
for i in range(5):
    try:
        snapshot_download("nvidia/Qwen3.8-Flash-Next-NVFP4", local_dir="/mnt/d/AIModels/Qwen3.8-Flash-Next-NVFP4", max_workers=8)
        print("DONE", flush=True)
        break
    except Exception as e:
        print("retry", i, e, flush=True)
PY
du -sh /mnt/d/AIModels/Qwen3.8-Flash-Next-NVFP4
