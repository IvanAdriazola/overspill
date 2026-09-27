#!/bin/bash
# (inside WSL) Download unsloth/Qwen3.8-Flash-Next-GGUF UD-IQ4_XS (93.7 GB) to D: for the llama.cpp comparison
# (llama.cpp will run natively on Windows and read it straight from D:). Caches on D:, xet chunk cache off.
set -e
export HF_HOME=/mnt/d/AIModels/.hf_home HF_XET_CHUNK_CACHE_SIZE_BYTES=0
unset HF_HUB_DISABLE_XET
~/ft/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for i in range(5):
    try:
        snapshot_download("unsloth/Qwen3.8-Flash-Next-GGUF", allow_patterns=["UD-IQ4_XS/*"],
                          local_dir="/mnt/d/AIModels/Qwen3.8-Flash-Next-GGUF", max_workers=8)
        print("DONE", flush=True)
        break
    except Exception as e:
        print("retry", i, e, flush=True)
PY
du -sh /mnt/d/AIModels/Qwen3.8-Flash-Next-GGUF
