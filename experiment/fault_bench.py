"""Microbenchmark: how fast can random experts be pulled from a file mapping?

Run inside WSL as root (drops the page cache between modes). Mimics the CPU MoE executor:
12 threads each touching every page of 6 random 13 MB experts (one token-layer), repeated.
Modes:
  fault      - just touch the pages (what the executor does today)
  willneed   - madvise(WILLNEED) on the 6 expert ranges first, then touch
  sequential - MADV_SEQUENTIAL on the whole mapping, then touch
  pread      - explicit parallel pread of each expert into a buffer (upper bound, O_DIRECT off)
Usage: python fault_bench.py <ftw_dir> [layers]
"""

import json
import mmap
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

PAGE = 4096
d = sys.argv[1]
n_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 40
ix = json.load(open(f"{d}/freetoken_weight.json"))
shards = sorted(ix["shards"], key=lambda s: s["global_off"])
banks = [t for t in ix["tensors"] if t["kind"] == "experts_bank" and "#L" in t["name"]]
by_layer = {}
for t in banks:
    base, layer = t["name"].split("#L")
    by_layer.setdefault(int(layer), []).append(t)


def locate(t):
    for sh in shards:
        if sh["global_off"] <= t["global_off"] < sh["global_off"] + sh["nbytes"]:
            return os.path.join(d, sh["file"]), t["global_off"] - sh["global_off"]
    raise KeyError(t["name"])


maps = {}


def mapping(path):
    if path not in maps:
        fd = os.open(path, os.O_RDONLY)
        maps[path] = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        os.close(fd)
    return maps[path]


def expert_ranges(layer, e):
    """(mmap, start, length) for each bank row of expert e in this layer."""
    out = []
    for t in by_layer[layer]:
        path, off = locate(t)
        n_exp = t["shape"][0]
        row = t["nbytes"] // n_exp
        out.append((mapping(path), off + e * row, row))
    return out


def drop_caches():
    os.system("sync; echo 3 > /proc/sys/vm/drop_caches")


def touch(m, start, length):
    view = np.frombuffer(m, dtype=np.uint8, count=length, offset=start)
    return int(view[::PAGE].sum())  # one byte per page -> every page faulted


def run(mode, trials):
    drop_caches()
    for m in maps.values():
        m.madvise(mmap.MADV_SEQUENTIAL if mode == "sequential" else mmap.MADV_NORMAL)
    rng = random.Random(0)
    total = 0
    t0 = time.time()
    with ThreadPoolExecutor(12) as pool:
        for _ in range(trials):
            layer = rng.randrange(n_layers)
            n_exp = by_layer[layer][0]["shape"][0]
            ranges = [r for e in rng.sample(range(n_exp), 6) for r in expert_ranges(layer, e)]
            if mode == "willneed":
                for m, s, n in ranges:
                    a = s - s % PAGE
                    m.madvise(mmap.MADV_WILLNEED, a, n + (s - a))
            if mode == "pread":
                def rd(r):
                    m, s, n = r
                    return len(m[s:s + n])
                list(pool.map(rd, ranges))
            else:
                # split each range into 12 slices, like the executor's row blocks
                work = []
                for m, s, n in ranges:
                    step = max(PAGE, (n // 12) // PAGE * PAGE)
                    work += [(m, o, min(step, s + n - o)) for o in range(s, s + n, step)]
                list(pool.map(lambda w: touch(*w), work))
            total += sum(n for _, _, n in ranges)
    dt = time.time() - t0
    print(f"{mode:10s} {total / 2**30:6.2f} GiB in {dt:6.1f}s = {total / dt / 2**30:5.2f} GiB/s "
          f"({dt / trials * 1000:6.0f} ms per 6-expert layer)", flush=True)


for mode in sys.argv[3].split(",") if len(sys.argv) > 3 else ["fault", "willneed", "sequential", "pread"]:
    run(mode, 30)
