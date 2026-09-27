"""Download DeepSeek-V4-Flash REAP-150B for the native-Windows comparison.

    python dl_dsv4.py gguf   -> G:\\AIModels\\dsv4\\...-MXFP4_MOE.gguf (85 GB, llama.cpp)
    python dl_dsv4.py src    -> F:\\dsv4_src (84.7 GB HF safetensors, only the input of the FTW conversion;
                                F: is the failing HDD, fine for re-downloadable scratch)

The hub cache lives next to each target and xet's chunk cache is off, so nothing lands on C:.
"""

import os
import sys

which = sys.argv[1]
if which == "gguf":
    repo, dest, pats = "puwaer/DeepSeek-V4-Flash-0731-reap-150b-gguf", r"G:\AIModels\dsv4", ["*MXFP4_MOE.gguf"]
elif which == "src":
    repo, dest, pats = "puwaer/DeepSeek-V4-Flash-0731-reap-150b", r"F:\dsv4_src", None
else:
    sys.exit("usage: dl_dsv4.py gguf|src")
os.makedirs(dest, exist_ok=True)
os.environ["HF_HOME"] = os.path.join(os.path.splitdrive(dest)[0] + os.sep, ".hf_home")
os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
os.environ.pop("HF_HUB_DISABLE_XET", None)

from huggingface_hub import snapshot_download  # noqa: E402  (env first)

for attempt in range(8):
    try:
        snapshot_download(repo, local_dir=dest, allow_patterns=pats, max_workers=8)
        print("DONE", repo, flush=True)
        break
    except Exception as e:  # noqa: BLE001
        print("retry", attempt, repr(e)[:300], flush=True)
