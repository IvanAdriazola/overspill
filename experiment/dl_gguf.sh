#!/bin/bash
# (inside WSL) Fetch the MXFP4_MOE GGUF of DeepSeek-V4-Flash REAP-150B for the llama.cpp comparison.
# (The original HF download was deleted after the Colibri tests; our FTW copy stays.)
# An 85 GB single file needs the xet backend; FreeToken's venv has hf_xet.
set -e
unset HF_HUB_DISABLE_XET
~/ft/.venv/bin/python - <<'PY'
import os
from huggingface_hub import list_repo_files, snapshot_download
repo = "puwaer/DeepSeek-V4-Flash-0731-reap-150b-gguf"
files = [f for f in list_repo_files(repo) if "MXFP4" in f.upper()]
print("files:", files, flush=True)
for i in range(5):
    try:
        snapshot_download(repo, local_dir=os.path.expanduser("~/models/dsv4_reap150b_gguf"), allow_patterns=files, max_workers=8)
        print("DONE", flush=True)
        break
    except Exception as e:
        print("retry", i, e, flush=True)
PY
du -sh /home/ivan/models/dsv4_reap150b_gguf
