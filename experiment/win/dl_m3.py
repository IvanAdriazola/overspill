"""Download MiniMax-M3 for the Overspill vs llama.cpp comparison, split across drives by size.

    python dl_m3.py src    -> nvidia/MiniMax-M3-NVFP4 (250 GB): first ~195 GB of files to D:\\AIModels\\m3_src,
                              the rest to F:\\AIModels\\m3_src (only the input of the FTW conversion; deleted after)
    python dl_m3.py gguf   -> unsloth/MiniMax-M3-GGUF UD-IQ4_XS (208 GB) to C:\\AIModels\\m3_gguf (llama.cpp)

Each drive gets its own hub cache and xet's chunk cache is off, so nothing lands elsewhere.
"""

import os
import sys

from huggingface_hub import HfApi

which = sys.argv[1]
os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
os.environ.pop("HF_HUB_DISABLE_XET", None)


def fetch(repo, dest, files):
    os.environ["HF_HOME"] = os.path.join(os.path.splitdrive(dest)[0] + os.sep, ".hf_home")
    from huggingface_hub import snapshot_download

    os.makedirs(dest, exist_ok=True)
    for attempt in range(8):
        try:
            snapshot_download(repo, local_dir=dest, allow_patterns=files, max_workers=8)
            print("DONE", repo, "->", dest, len(files), "files", flush=True)
            return
        except Exception as e:  # noqa: BLE001
            print("retry", attempt, repr(e)[:300], flush=True)


if which == "src":
    repo = "nvidia/MiniMax-M3-NVFP4"
    info = HfApi().model_info(repo, files_metadata=True)
    files = sorted((s.rfilename, s.size or 0) for s in info.siblings)
    d_files, f_files, acc = [], [], 0
    for name, size in files:
        if acc + size <= 195e9 or not name.endswith(".safetensors"):
            d_files.append(name)
            acc += size
        else:
            f_files.append(name)
    print(f"D: {len(d_files)} files {acc / 1e9:.1f} GB; F: {len(f_files)} files "
          f"{sum(s for n, s in files if n in set(f_files)) / 1e9:.1f} GB", flush=True)
    fetch(repo, r"D:\AIModels\m3_src", d_files)
    fetch(repo, r"F:\AIModels\m3_src", f_files)
elif which == "gguf":
    fetch("unsloth/MiniMax-M3-GGUF", r"C:\AIModels\m3_gguf", ["UD-IQ4_XS/*"])
else:
    sys.exit("usage: dl_m3.py src|gguf")
