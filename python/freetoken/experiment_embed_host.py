"""Move the token-embedding table to pinned host memory (tiering experiment).

``FT_EMBED_HOST=1``. The embedding is a pure row gather: the GPU reads the few rows a
batch needs straight from pinned, device-mapped host RAM over UVA (the same Triton gather
qwen4_exp uses for its PLE table), so the table's VRAM (1 GiB for DeepSeek-V4) goes to
the expert cache instead. Graph-safe: no host code runs per token.
"""

from __future__ import annotations

import os
import types

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("FT_EMBED_HOST", "").lower() in ("1", "true", "yes", "on")


def _iter_ops(root):
    from freetoken.layers.base import BaseOP

    seen, stack = set(), [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        yield obj
        for value in vars(obj).values():
            if isinstance(value, BaseOP):
                stack.append(value)
            elif isinstance(value, (list, tuple)):
                stack.extend(v for v in value if isinstance(v, BaseOP))


def _host_forward(self, x: torch.Tensor) -> torch.Tensor:
    from freetoken.kernel.triton.ple import ple_gather_rows

    ids = x.reshape(-1)
    out = torch.empty((ids.numel(), self._host_dim), dtype=torch.bfloat16, device=x.device)
    return ple_gather_rows(self._host_ptr, self._host_rows, self._host_dim, ids, out, 1.0, is_fp8=False)


def move_embeddings_to_host(model) -> int:
    """Returns the bytes of VRAM released."""
    from freetoken.kernel.pinned import device_ptr
    from freetoken.layers.embedding import ParallelLMHead, VocabParallelEmbedding
    from freetoken.moe.host_banks import HostBank

    freed = 0
    for op in _iter_ops(model):
        if type(op) is not VocabParallelEmbedding or isinstance(op, ParallelLMHead):
            continue
        w = op.weight
        if not w.is_cuda or op.tp_size != 1 or w.dtype != torch.bfloat16 or op._embed_scale is not None:
            continue
        bank = HostBank(tuple(w.shape), w.dtype)
        bank.tensor.copy_(w)
        bank.pin()
        op._host_bank = bank
        op._host_rows, op._host_dim = w.shape
        op._host_ptr = device_ptr(bank.tensor)
        freed += w.numel() * w.element_size()
        op.weight = torch.empty(0, dtype=w.dtype, device=w.device)
        op.forward = types.MethodType(_host_forward, op)
        del w
    if freed:
        torch.cuda.empty_cache()
        logger.info_rank0(f"FT_EMBED_HOST: {freed / 2**30:.2f} GiB of embedding moved to pinned host RAM")
    return freed
