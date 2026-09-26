"""Offline expert-placement simulator (plan step 0, see EXPERIMENT.md).

Replays routing traces from ``experiment_trace.py`` against VRAM/RAM/disk
placement policies, with budgets shrunk so the model "doesn't fit", and
turns the resulting hits/misses into bytes and estimated time.

Tiers: VRAM slot cache (V experts), RAM cache (R experts), disk (the rest).
A VRAM miss served from RAM costs a PCIe copy; a RAM miss costs a disk read
(+ the PCIe copy). Prefetched copies overlap the layer's compute; demand
copies stall it.

Usage: python simulate.py <trace_dir> [--out results.json]
"""

from __future__ import annotations

import argparse
import glob
import heapq
import json
import pickle
import re
import time
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import product
from multiprocessing import Pool

import numpy as np

# Qwen3.6-35B-A3B, NVFP4 (FreeToken bank sizes)
N_LAYERS = 40
N_EXPERTS = 256
EXPERT_BYTES = 1_775_616
TOTAL = N_LAYERS * N_EXPERTS

# Machine (RTX 3060, PCIe 4.0 x16, NVMe through WSL's virtual disk)
PCIE_BPS = 24e9
DISK_BPS = {"wsl_vhdx": 1.5e9, "native_nvme": 5.0e9}
PREFILL_COMPUTE_TPS = 1020.0      # measured FreeToken prefill, compute-bound regime
MEASURED_DECODE_TPS = 62.0         # FreeToken, all experts pinned in RAM, 3347 VRAM slots
REAL_VRAM_SLOTS = 3347


# ---------------------------------------------------------------- trace loading

@dataclass
class Trace:
    decode: list[np.ndarray] = field(default_factory=list)       # per step: [L, k] expert ids
    pred: list[np.ndarray | None] = field(default_factory=list)  # per step: [L, 16] lookahead for layer+1
    prefill: list[list[tuple[np.ndarray, np.ndarray]]] = field(default_factory=list)  # per chunk: per layer (uniq, counts)
    prefill_tokens: list[int] = field(default_factory=list)
    order: list[tuple[str, int]] = field(default_factory=list)   # ("d", i) / ("p", i) in time order


def load(trace_dir: str) -> Trace:
    files = sorted(glob.glob(f"{trace_dir}/*.npz"), key=lambda f: int(re.search(r"_(\d+)\.npz$", f).group(1)))
    recs = []
    for f in files:
        z = np.load(f)
        chunk = [dict() for _ in range(int(z["n"]))]
        for key in z.files:
            if key == "n":
                continue
            i, name = key.split(".", 1)
            chunk[int(i)][name] = z[key]
        recs.extend(chunk)
    tr = Trace()
    step: list = []
    for r in recs:
        layer = int(r["layer"])
        if layer == 0 and step:
            _close(tr, step)
            step = []
        step.append(r)
    if step:
        _close(tr, step)
    return tr


def _close(tr: Trace, step: list) -> None:
    if len(step) != N_LAYERS:
        return  # partial step at a flush/kill boundary
    if bool(step[0]["prefill"]):
        tr.order.append(("p", len(tr.prefill)))
        tr.prefill.append([(r["uniq"].astype(np.int64), r["counts"]) for r in step])
        tr.prefill_tokens.append(int(step[0]["n_tokens"]))
    else:
        # batch size 1 in Workshop runs; take the first request's row
        ids = np.stack([r["ids"][0] for r in step]).astype(np.int64)
        pred = [r["pred_next"][0] if "pred_next" in r else None for r in step]
        tr.order.append(("d", len(tr.decode)))
        tr.decode.append(ids)
        tr.pred.append(np.stack(pred[:-1]).astype(np.int64) if all(p is not None for p in pred[:-1]) else None)


# ---------------------------------------------------------------- caches

class LRU:
    def __init__(self, cap: int):
        self.cap, self.d = cap, OrderedDict()

    def __contains__(self, e):
        return e in self.d

    def touch(self, e):
        self.d.move_to_end(e)

    def insert(self, e, protect=()):
        if self.cap <= 0:
            return
        if e in self.d:
            self.d.move_to_end(e)
            return
        if len(self.d) >= self.cap:
            for victim in self.d:
                if victim not in protect:
                    del self.d[victim]
                    break
            else:
                return
        self.d[e] = True


class LFU:
    """Heat counters with periodic halving; membership rebalanced every
    ``period`` decode steps to the hottest ``cap`` experts, with Colibri's
    hysteresis (``hot > cold*1.25 + 4``) on each swap. Between rebalances a
    miss is inserted only if there is free room."""

    def __init__(self, cap: int, period: int = 16, half_life: int = 1024):
        self.cap, self.period, self.half_life = cap, period, half_life
        self.heat = np.zeros(TOTAL, dtype=np.float64)
        self.members: set[int] = set()
        self.ticks = 0

    def __contains__(self, e):
        return e in self.members

    def touch(self, e):
        pass

    def note(self, e, w=1.0):
        self.heat[e] += w

    def insert(self, e, protect=()):
        if len(self.members) < self.cap:
            self.members.add(e)

    def tick(self):
        self.ticks += 1
        if self.ticks % self.half_life == 0:
            self.heat *= 0.5
        if self.ticks % self.period or self.cap <= 0:
            return
        want = np.argpartition(-self.heat, min(self.cap, TOTAL - 1))[: self.cap]
        outsiders = sorted((e for e in want.tolist() if e not in self.members), key=lambda e: -self.heat[e])
        if not outsiders:
            return
        residents = sorted(self.members, key=lambda e: self.heat[e])
        for hot, cold in zip(outsiders, residents):
            if self.heat[hot] > self.heat[cold] * 1.25 + 4:
                self.members.discard(cold)
                self.members.add(hot)
            else:
                break


class Belady:
    """Offline optimum for the VRAM tier: evict the resident used farthest in
    the future. Needs ``next_use`` for every decode access (precomputed)."""

    def __init__(self, cap: int):
        self.cap, self.members, self.heap = cap, {}, []

    def __contains__(self, e):
        return e in self.members

    def touch(self, e, nxt=None):
        if nxt is not None:
            self.members[e] = nxt
            heapq.heappush(self.heap, (-nxt, e))

    def insert(self, e, protect=(), nxt=None):
        if self.cap <= 0 or nxt is None:
            return
        skipped = []
        while len(self.members) >= self.cap:
            neg, victim = heapq.heappop(self.heap)
            if self.members.get(victim) != -neg:
                continue  # stale entry
            if victim in protect:
                skipped.append((neg, victim))  # in use this layer: keep, retry after
                continue
            del self.members[victim]
        for item in skipped:
            heapq.heappush(self.heap, item)
        self.members[e] = nxt
        heapq.heappush(self.heap, (-nxt, e))


def _preload(ram, slots: int) -> None:
    """RAM starts as after a fresh load: an even share of every layer's
    experts (all of them when it fits), not empty."""
    per_layer = min(N_EXPERTS, slots // N_LAYERS)
    for l in range(N_LAYERS):
        for e in range(per_layer):
            ram.insert(l * N_EXPERTS + e)


def next_uses(tr: Trace) -> list[np.ndarray]:
    """For each decode step and layer slot, the decode-step index of that
    expert's next use (inf if never)."""
    last: dict[int, int] = {}
    out = [np.full((N_LAYERS, s.shape[1]), np.inf) for s in tr.decode]
    for i in range(len(tr.decode) - 1, -1, -1):
        g = tr.decode[i] + np.arange(N_LAYERS)[:, None] * N_EXPERTS
        for l in range(N_LAYERS):
            for j, e in enumerate(g[l].tolist()):
                out[i][l, j] = last.get(e, np.inf)
                last[e] = i
    return out


# ---------------------------------------------------------------- simulation

def simulate(tr: Trace, cfg: dict, nxt: list | None = None) -> dict:
    V, R = cfg["vram_slots"], cfg["ram_slots"]
    vram = {"lru": lambda: LRU(V), "lfu": lambda: LFU(V), "belady": lambda: Belady(V)}[cfg["vram"]]()
    ram = {"lru": lambda: LRU(R), "lfu": lambda: LFU(R)}[cfg["ram"]]()
    _preload(ram, R)
    k_pf, prefill_mode = cfg["prefetch"], cfg["prefill"]
    disk_bps = DISK_BPS[cfg["disk"]]
    t_pcie, t_disk = EXPERT_BYTES / PCIE_BPS, EXPERT_BYTES / disk_bps
    c = dict(vram_hit=0, ram_hit=0, disk=0, pf_pcie=0, pf_disk=0, pf_used=0,
             exposed_s=0.0, demand_s=0.0, prefill_s=0.0, prefill_disk=0, prefill_tokens=0)
    lfu_tiers = [t for t in (vram, ram) if isinstance(t, LFU)]
    prefetched: set[int] = set()

    for kind, idx in tr.order:
        if kind == "p":
            tokens = tr.prefill_tokens[idx]
            c["prefill_tokens"] += tokens
            layer_compute = tokens / PREFILL_COMPUTE_TPS / N_LAYERS
            for l, (uniq, counts) in enumerate(tr.prefill[idx]):
                needed = (np.arange(N_EXPERTS) if prefill_mode == "whole" else uniq) + l * N_EXPERTS
                n_disk = 0
                for e in needed.tolist():
                    if e not in ram:
                        n_disk += 1
                        if prefill_mode == "union":
                            ram.insert(e)
                    for t in lfu_tiers:
                        t.note(e, 1.0)
                transfer = len(needed) * t_pcie + n_disk * t_disk
                c["prefill_s"] += max(layer_compute, transfer)
                c["prefill_disk"] += n_disk
            continue

        ids = tr.decode[idx]
        pred = tr.pred[idx]
        uses = nxt[idx] if nxt is not None else None
        for l in range(N_LAYERS):
            g = (ids[l] + l * N_EXPERTS).tolist()
            protect = set(g)
            demand = 0.0
            for j, e in enumerate(g):
                nu = uses[l, j] if uses is not None else None
                for t in lfu_tiers:
                    t.note(e)
                if e in vram:
                    c["vram_hit"] += 1
                    if e in prefetched:
                        c["pf_used"] += 1
                        prefetched.discard(e)
                    vram.touch(e, nu) if isinstance(vram, Belady) else vram.touch(e)
                    continue
                if e in ram:
                    c["ram_hit"] += 1
                    ram.touch(e)
                    demand += t_pcie
                else:
                    c["disk"] += 1
                    demand += t_disk + t_pcie
                    ram.insert(e)
                if isinstance(vram, Belady):
                    vram.insert(e, protect, nu)
                else:
                    vram.insert(e, protect)
            # prefetch layer l+1's predicted experts behind this layer's compute
            overlap = 0.0
            if k_pf and pred is not None and l + 1 < N_LAYERS:
                for e in (pred[l, :k_pf] + (l + 1) * N_EXPERTS).tolist():
                    if e in vram:
                        continue
                    if e in ram:
                        overlap += t_pcie
                        c["pf_pcie"] += 1
                    else:
                        overlap += t_disk + t_pcie
                        c["pf_disk"] += 1
                        ram.insert(e)
                    if not isinstance(vram, Belady):
                        vram.insert(e, protect)
                        prefetched.add(e)
            c["demand_s"] += demand
            c["exposed_s"] += demand + overlap  # compute share subtracted below
        for t in lfu_tiers:
            t.tick()

    steps = len(tr.decode)
    picks = c["vram_hit"] + c["ram_hit"] + c["disk"]
    out = dict(cfg)
    out.update(
        steps=steps,
        vram_hit_rate=c["vram_hit"] / max(picks, 1),
        ram_hit_rate=c["ram_hit"] / max(picks, 1),
        disk_rate=c["disk"] / max(picks, 1),
        disk_mb_per_token=(c["disk"] + c["pf_disk"]) * EXPERT_BYTES / 1e6 / max(steps, 1),
        pcie_mb_per_token=(c["ram_hit"] + c["disk"] + c["pf_pcie"] + c["pf_disk"]) * EXPERT_BYTES / 1e6 / max(steps, 1),
        prefetch_precision=c["pf_used"] / max(c["pf_pcie"] + c["pf_disk"], 1),
        demand_ms_per_token=1000 * c["demand_s"] / max(steps, 1),
        prefetch_ms_per_token=1000 * (c["exposed_s"] - c["demand_s"]) / max(steps, 1),
        prefill_tokens=c["prefill_tokens"],
        prefill_tps=c["prefill_tokens"] / max(c["prefill_s"], 1e-9),
        prefill_disk_gb=c["prefill_disk"] * EXPERT_BYTES / 1e9,
    )
    return out


def with_time(r: dict, compute_ms: float) -> dict:
    """Per token: compute + demand stalls + prefetch traffic not hidden by compute."""
    layer_compute = compute_ms / N_LAYERS
    hidden = min(r["prefetch_ms_per_token"], compute_ms)  # upper bound on overlap
    exposed_pf = r["prefetch_ms_per_token"] - hidden
    r["est_ms_per_token"] = compute_ms + r["demand_ms_per_token"] + exposed_pf
    r["est_decode_tps"] = 1000 / r["est_ms_per_token"]
    r["layer_compute_ms"] = layer_compute
    return r


# ---------------------------------------------------------------- sweep

_TR: Trace | None = None
_NXT = None


def _init_globals(tr):
    # Set before the Pool is created: workers are forked and inherit them.
    global _TR, _NXT
    _TR = tr
    _NXT = next_uses(tr)


def _run(cfg):
    return simulate(_TR, cfg, _NXT if cfg["vram"] == "belady" else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir")
    ap.add_argument("--out", default="sim_results.json")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    t0 = time.time()
    cache = Path(args.trace_dir) / "_parsed.pkl"
    if cache.exists():
        tr = pickle.loads(cache.read_bytes())
    else:
        tr = load(args.trace_dir)
        cache.write_bytes(pickle.dumps(tr))
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    print(f"decode steps {len(tr.decode)}, prefill chunks {len(tr.prefill)} ({sum(tr.prefill_tokens)} tokens)")

    # Calibrate compute: the real FreeToken config (all experts in RAM, LRU VRAM, whole-layer prefill)
    # must reproduce the measured decode speed; compute = measured - simulated PCIe stalls.
    base_cfg = dict(vram="lru", ram="lru", vram_slots=REAL_VRAM_SLOTS, ram_slots=TOTAL,
                    prefetch=0, prefill="whole", disk="wsl_vhdx")
    base = simulate(tr, base_cfg)
    compute_ms = 1000 / MEASURED_DECODE_TPS - base["demand_ms_per_token"]
    print(f"calibration: LRU VRAM hit {base['vram_hit_rate']:.3f}, PCIe stall {base['demand_ms_per_token']:.2f} ms/token "
          f"-> compute {compute_ms:.2f} ms/token")

    grid = []
    for ram_frac, vram, ram, pf, prefill, disk in product(
        [1.0, 0.75, 0.5, 0.35], ["lru", "lfu", "belady"], ["lru", "lfu"], [0, 8, 16],
        ["whole", "union"], ["wsl_vhdx", "native_nvme"],
    ):
        if vram == "belady" and pf:
            continue  # Belady is the no-prefetch optimum reference
        if ram_frac == 1.0 and (ram != "lru" or disk != "wsl_vhdx"):
            continue  # everything in RAM: RAM policy and disk don't matter
        grid.append(dict(vram=vram, ram=ram, vram_slots=REAL_VRAM_SLOTS, ram_slots=int(TOTAL * ram_frac),
                         ram_frac=ram_frac, prefetch=pf, prefill=prefill, disk=disk))
    print(f"{len(grid)} configs", flush=True)
    _init_globals(tr)
    with Pool(args.workers) as pool:
        results = [with_time(r, compute_ms) for r in pool.map(_run, grid)]
    json.dump({"compute_ms": compute_ms, "base": base, "results": results}, open(args.out, "w"), indent=1)

    print(f"\n{'RAM%':>5} {'vram':>6} {'ram':>4} {'pf':>3} {'prefill':>7} {'disk':>11} | {'VRAMhit':>7} {'disk%':>6} "
          f"{'diskMB/t':>8} {'dec tok/s':>9} {'prefill tok/s':>13}")
    for r in sorted(results, key=lambda r: (-r["ram_frac"], r["disk"], -r["est_decode_tps"])):
        print(f"{r['ram_frac']*100:5.0f} {r['vram']:>6} {r['ram']:>4} {r['prefetch']:3d} {r['prefill']:>7} {r['disk']:>11} | "
              f"{r['vram_hit_rate']:7.3f} {r['disk_rate']*100:6.2f} {r['disk_mb_per_token']:8.1f} "
              f"{r['est_decode_tps']:9.1f} {r['prefill_tps']:13.0f}")


if __name__ == "__main__":
    main()
