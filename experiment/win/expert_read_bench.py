"""Microbenchmark: how fast can decode's expert reads come off disk on native Windows, by method?

Replays DeepSeek-V4's decode access pattern on its FTW: random (layer, expert) picks, each expert = its 4 bank rows
(gate_up 8 MiB, gate_up_scale 0.5, down 4, down_scale 0.25), 11 threads in parallel. Each method gets its own
fresh random picks (seeded), so none benefits from another's page cache.
  mapped   : what Overspill decode does now - PrefetchVirtualMemory on the mapped rows, then touch every page
  buffered : ReadFile through the cache manager into a scratch buffer (per-thread handles, positional)
  direct   : unbuffered ReadFile (FILE_FLAG_NO_BUFFERING) into pinned-style aligned scratch
Usage: python expert_read_bench.py <ftw_dir> [experts_per_method=240]
"""

import ctypes
import json
import mmap
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from freetoken.moe import win_direct_reader as w
from freetoken.utils.winsys import prefetch_ranges

ftw = sys.argv[1]
n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 240
idx = json.load(open(os.path.join(ftw, "freetoken_weight.json")))
shards = idx["shards"]
banks = [e for e in idx["tensors"] if e["kind"] == "experts_bank"]
by_layer: dict[int, list[dict]] = {}
for e in banks:
    by_layer.setdefault(int(e["name"].split("#L")[1]), []).append(e)
n_exp = banks[0]["shape"][0]


def locate(goff: int, n: int):
    for sh in shards:
        if sh["global_off"] <= goff and goff + n <= sh["global_off"] + sh["nbytes"]:
            return os.path.join(ftw, sh["file"]), goff - sh["global_off"]
    return None


def picks(seed: int):
    rnd = random.Random(seed)
    out = []
    while len(out) < n_per:
        lid, ex = rnd.choice(list(by_layer)), rnd.randrange(n_exp)
        rows = []
        for e in by_layer[lid]:
            rb = e["nbytes"] // n_exp
            loc = locate(e["global_off"] + ex * rb, rb)
            if loc is None:
                break
            rows.append((loc[0], loc[1], rb))
        else:
            out.append(rows)
    return out


maps: dict[str, tuple] = {}


def view(path):
    if path not in maps:
        fd = os.open(path, os.O_RDONLY | os.O_BINARY)
        m = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        os.close(fd)
        maps[path] = (m, ctypes.addressof(ctypes.c_char.from_buffer_copy(b"\0")) if False else None)
    return maps[path][0]


def do_mapped(expert):
    rngs = []
    for path, off, n in expert:
        m = view(path)
        base = ctypes.addressof((ctypes.c_char * 1).from_buffer_copy(b"\0"))  # placeholder, replaced below
        a = np.frombuffer(m, dtype=np.uint8, count=n, offset=off)
        rngs.append((a.ctypes.data, n))
    prefetch_ranges(rngs)
    s = 0
    for addr, n in rngs:
        s += int(np.ctypeslib.as_array((ctypes.c_uint8 * n).from_address(addr))[::4096].sum())
    return s


_local = threading.local()


def do_buffered(expert):
    if not hasattr(_local, "f"):
        _local.f = {}
        _local.buf = bytearray(16 << 20)
    for path, off, n in expert:
        f = _local.f.get(path) or _local.f.setdefault(path, open(path, "rb", buffering=0))
        f.seek(off)
        f.readinto(memoryview(_local.buf)[:n])


handles = w._Handles()


def do_direct(expert):
    if not hasattr(_local, "abuf"):
        _local.abuf = mmap.mmap(-1, 16 << 20)  # page-aligned anonymous scratch
        _local.aaddr = ctypes.addressof(ctypes.c_char.from_buffer(_local.abuf))
    for path, off, n in expert:
        start = off - off % 4096
        span = -(-(off + n - start) // 4096) * 4096
        w._read_at(handles.get(path), _local.aaddr, span, start)


for seed, (name, fn) in enumerate((("mapped", do_mapped), ("buffered", do_buffered), ("direct", do_direct)), start=1):
    ex = picks(1000 + seed)
    total = sum(n for e in ex for _, _, n in e)
    t0 = time.time()
    with ThreadPoolExecutor(11) as pool:
        list(pool.map(fn, ex))
    dt = time.time() - t0
    print(f"{name:9s}: {len(ex)} experts, {total / 2**30:.2f} GiB in {dt:.2f} s = {total / dt / 1e9:.2f} GB/s "
          f"({dt / len(ex) * 1000 * 11:.1f} ms per expert per thread)", flush=True)
