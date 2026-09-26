"""Routing-trace capture for the tiering experiment (see EXPERIMENT.md).

Enabled by ``FT_ROUTE_TRACE=<dir>``; needs ``--graph 0`` so decode runs eagerly
(CUDA-graph replays skip Python). For every MoE forward it records, per token:
the real top-k expert ids of this layer, and a lookahead prediction for the
next layer (next layer's router applied to this layer's MoE input - the
cheapest router-lookahead signal a prefetcher could use).

Records are buffered and written by a daemon thread every ``_FLUSH_SECONDS``
(the server is usually killed, not exited, so atexit alone loses the tail).
"""

from __future__ import annotations

import atexit
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch

_DIR = os.environ.get("FT_ROUTE_TRACE")
ENABLED = bool(_DIR)
_LOOKAHEAD_K = 16

_gates: dict[int, torch.nn.Module] = {}
_buf: list[dict] = []
_FLUSH_SECONDS = 5
_chunk = 0
_lock = threading.Lock()


def register_gate(layer_id: int, gate) -> None:
    if ENABLED:
        _gates[layer_id] = gate


def record(layer_id: int, hidden_states: torch.Tensor, router_logits: torch.Tensor, top_k: int) -> None:
    """Called from the MoE block before the routed experts run (the expert
    kernel may overwrite ``hidden_states`` in place)."""
    from freetoken.core import get_global_ctx

    batch = get_global_ctx().batch
    ids = torch.topk(router_logits, top_k, dim=-1).indices
    rec = {
        "layer": layer_id,
        "prefill": batch.is_prefill,
        "t": time.time(),
        "uids": np.array([r.uid for r in batch.reqs], dtype=np.int64),
        "n_tokens": ids.shape[0],
    }
    if batch.is_prefill:
        # Prefill only needs which experts the chunk touches (and how often),
        # not the per-token order: keeps traces small.
        uniq, counts = torch.unique(ids, return_counts=True)
        rec["uniq"] = uniq.to(torch.int16).cpu().numpy()
        rec["counts"] = counts.to(torch.int32).cpu().numpy()
    else:
        rec["ids"] = ids.to(torch.int16).cpu().numpy()
    nxt = _gates.get(layer_id + 1)
    if nxt is not None and not batch.is_prefill:
        pred = torch.topk(nxt.forward(hidden_states), _LOOKAHEAD_K, dim=-1).indices
        rec["pred_next"] = pred.to(torch.int16).cpu().numpy()
    with _lock:
        _buf.append(rec)


def flush() -> None:
    global _chunk
    with _lock:
        records = _buf[:]
        _buf.clear()
    if not records:
        return
    out = Path(_DIR)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, rec in enumerate(records):
        for key, value in rec.items():
            arrays[f"{i}.{key}"] = np.asarray(value)
    np.savez_compressed(out / f"trace_{int(time.time())}_{_chunk:05d}.npz", n=len(records), **arrays)
    _chunk += 1


def _flusher() -> None:
    while True:
        time.sleep(_FLUSH_SECONDS)
        flush()


if ENABLED:
    atexit.register(flush)
    threading.Thread(target=_flusher, name="route-trace-flush", daemon=True).start()
