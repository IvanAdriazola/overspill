"""Prefetch the M3 source files the FTW converter will read next into the Windows file cache (2026-09-29).

The converter (WSL, convert_m3_ntfs.sh) reads each expert layer's tensors scattered across 2 safetensors files on
E: (an HDD): ~360 small reads/s, seek-bound at ~20 MB/s -> ~250 s per layer, ~4 h for 57 layers. Reading those
files sequentially from Windows (~90 MB/s) puts them in the host file cache, which WSL's drvfs reads are served from.
Follows the converter's progress line ("Loading MiniMax-M3 NVFP4 experts: k/57") and keeps AHEAD layers read ahead.

    python prefetch_m3.py            (runs until the conversion log shows it finished)
"""
import collections
import json
import os
import re
import time

SRC = r"E:\AIModels\m3_src"
LOG = r"C:\GIT\Freetoken-colibri-experiment\experiment\convert_m3_ntfs.log"
AHEAD = 4
CHUNK = 16 << 20

index = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
by_layer = collections.defaultdict(set)
for name, f in index.items():
    m = re.search(r"layers\.(\d+)\.", name)
    if m and "expert" in name:
        by_layer[int(m.group(1))].add(f)
layers = sorted(by_layer)
done = set()
progress_re = re.compile(r"NVFP4 experts:.*?\|\s*(\d+)/(\d+)")


def current():
    with open(LOG, "rb") as fh:
        fh.seek(max(0, os.path.getsize(LOG) - 20000))
        tail = fh.read().decode("utf-8", "replace")
    if "=== exit" in tail:
        return None
    hits = progress_re.findall(tail)
    return int(hits[-1][0]) if hits else 0


def read(f):
    t0, n = time.time(), 0
    with open(os.path.join(SRC, f), "rb", buffering=0) as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            n += len(b)
    dt = time.time() - t0
    print(f"{time.strftime('%H:%M:%S')} read {f} {n / 1e9:.1f} GB in {dt:.0f} s ({n / 1e6 / max(dt, 1e-3):.0f} MB/s)",
          flush=True)


while True:
    k = current()
    if k is None:
        print("conversion finished - stopping", flush=True)
        break
    todo = [f for L in layers[k:k + AHEAD] for f in sorted(by_layer[L]) if f not in done]
    if not todo:
        if k >= len(layers):
            break
        time.sleep(5)
        continue
    f = todo[0]
    read(f)
    done.add(f)
