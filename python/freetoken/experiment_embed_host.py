"""Move the token embedding and the LM head to pinned host memory (tiering experiment).

``FT_EMBED_HOST=1``: the embedding is a pure row gather -- the GPU reads the few rows a
batch needs straight from pinned, device-mapped host RAM over UVA (the Triton gather
qwen4_exp uses for its PLE table).
``FT_HEAD_HOST=1``: the (untied, bf16) LM head becomes a GEMV that streams its weight from
pinned host RAM over PCIe -- ~1 GiB per generated token for DeepSeek-V4 (~40-50 ms on
PCIe 4.0 x16), which buys 1 GiB of VRAM for activations on a 12 GB card.
Both are graph-safe: no host code runs per token.
"""

from __future__ import annotations

import os
import types

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

def _flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


EMBED = _flag("FT_EMBED_HOST")
HEAD = _flag("FT_HEAD_HOST")
ENABLED = EMBED or HEAD


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


import triton
import triton.language as tl


@triton.jit
def _host_gemv_kernel(w_addr, x_ptr, y_ptr, V, D: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr):
    """y[m, v] = sum_d W[v, d] * x[m, d] with bf16 W at a raw (host, UVA-mapped) address."""
    pid = tl.program_id(0)
    m = tl.program_id(1)
    rows = pid * BLOCK_V + tl.arange(0, BLOCK_V)
    rmask = rows < V
    w = w_addr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
    acc = tl.zeros([BLOCK_V], dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        cols = d0 + tl.arange(0, BLOCK_D)
        x = tl.load(x_ptr + m * D + cols).to(tl.float32)
        wt = tl.load(w + rows[:, None].to(tl.int64) * D + cols[None, :], mask=rmask[:, None], other=0.0)
        acc += tl.sum(wt.to(tl.float32) * x[None, :], axis=1)
    tl.store(y_ptr + m * V + rows, acc.to(y_ptr.dtype.element_ty), mask=rmask)


def _head_forward(self, x: torch.Tensor) -> torch.Tensor:
    from freetoken.core import get_global_ctx

    batch = get_global_ctx().batch
    if batch.is_prefill:
        x = x[batch.attn_metadata.get_last_indices(batch.size)]
    x = x.contiguous()
    rows, dim = self._host_rows, self._host_dim
    out = torch.empty((x.shape[0], rows), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(rows, 16), x.shape[0])
    _host_gemv_kernel[grid](self._host_ptr, x, out, rows, D=dim, BLOCK_V=16, BLOCK_D=256, num_warps=4)
    return out


def move_embeddings_to_host(model) -> int:
    """Returns the bytes of VRAM released."""
    from freetoken.kernel.pinned import device_ptr
    from freetoken.layers.embedding import ParallelLMHead, VocabParallelEmbedding
    from freetoken.moe.host_banks import HostBank

    freed = 0
    for op in _iter_ops(model):
        is_head = isinstance(op, ParallelLMHead)
        if is_head:
            if not HEAD or op.tied_embedding is not None or op.bias is not None:
                continue
        elif type(op) is not VocabParallelEmbedding or not EMBED:
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
        op.forward = types.MethodType(_head_forward if is_head else _host_forward, op)
        del w
    if freed:
        torch.cuda.empty_cache()
        logger.info_rank0(f"[exp] {freed / 2**30:.2f} GiB of embedding/LM-head weights moved to pinned host RAM")
    return freed
