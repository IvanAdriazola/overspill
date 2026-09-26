"""Router-lookahead expert prefetch (tiering experiment, see EXPERIMENT.md).

``FT_LOOKAHEAD_K=<k>`` (default 0 = off). During decode, at MoE layer L the
next layer's router is applied to L's MoE input; its top-k experts are made
resident in the VRAM slot cache right after L's own lookup, and copied from
pinned host RAM on a side stream while L's experts compute. Layer L+1 waits
for that copy just before its GEMM, so correct predictions become hits and
wrong ones only cost PCIe bandwidth (unpredicted experts are fetched on
demand as before). Everything stays device-side: CUDA-graph capturable.

Simulated on real Workshop traces: VRAM hit 86.7% -> 94.1% (k=8) / 96.5% (k=16).
"""

from __future__ import annotations

import os

import torch

K = int(os.environ.get("FT_LOOKAHEAD_K", "0"))
ENABLED = K > 0

_gates: dict[int, object] = {}


def register_gate(layer_id: int, gate) -> None:
    if ENABLED:
        _gates[layer_id] = gate


def predict_next(layer_id: int, hidden_states: torch.Tensor) -> torch.Tensor | None:
    """Top-K expert ids of layer ``layer_id + 1``, predicted from this layer's MoE input."""
    gate = _gates.get(layer_id + 1)
    if gate is None:
        return None
    return torch.topk(gate.forward(hidden_states), K, dim=-1).indices.to(torch.int32)
