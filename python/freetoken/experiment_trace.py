"""Routing-trace capture for the tiering experiment (see EXPERIMENT.md).

Enabled by ``FT_ROUTE_TRACE=<dir>``; needs ``--graph 0`` so decode runs eagerly
(CUDA-graph replays skip Python). For every MoE forward it records, per token:
the real top-k expert ids of this layer, and a lookahead prediction for the
next layer (next layer's router applied to this layer's MoE input - the
cheapest router-lookahead signal a prefetcher could use).

One ``.npz`` per forward pass (all layers of one batch step) is too many
files, so records are buffered and flushed every ``_FLUSH_EVERY`` steps.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

_DIR = os.environ.get("FT_ROUTE_TRACE")
ENABLED = bool(_DIR)
_LOOKAHEAD_K = 16
_FLUSH_EVERY = 256

_gates: dict[int, torch.nn.Module] = {}
_buf: list[dict] = []
_step = 0
_chunk = 0


def register_gate(layer_id: int, gate) -> None:
    if ENABLED:
        _gates[layer_id] = gate


def record(layer_id: int, hidden_states: torch.Tensor, router_logits: torch.Tensor, top_k: int) -> None:
    """Called from the MoE block before the routed experts run (the expert
    kernel may overwrite ``hidden_states`` in place)."""
    global _step
    from freetoken.core import get_global_ctx

    batch = get_global_ctx().batch
    ids = torch.topk(router_logits, top_k, dim=-1).indices
    rec = {
        "layer": layer_id,
        "prefill": batch.is_prefill,
        "uids": np.array([r.uid for r in batch.reqs], dtype=np.int64),
        "ids": ids.to(torch.int16).cpu().numpy(),
    }
    nxt = _gates.get(layer_id + 1)
    if nxt is not None:
        pred = torch.topk(nxt.forward(hidden_states), _LOOKAHEAD_K, dim=-1).indices
        rec["pred_next"] = pred.to(torch.int16).cpu().numpy()
    _buf.append(rec)
    if layer_id == 0:
        _step += 1
        if _step % _FLUSH_EVERY == 0:
            flush()


def flush() -> None:
    global _chunk
    if not _buf:
        return
    out = Path(_DIR)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, rec in enumerate(_buf):
        for key, value in rec.items():
            arrays[f"{i}.{key}"] = np.asarray(value)
    np.savez_compressed(out / f"trace_{int(time.time())}_{_chunk:05d}.npz", n=len(_buf), **arrays)
    _chunk += 1
    _buf.clear()
