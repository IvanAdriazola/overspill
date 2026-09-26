"""Grammar-constrained decoding (OpenAI ``response_format`` json_schema / json_object).

One xgrammar ``GrammarMatcher`` per constrained request, keyed by uid. The scheduler:

* ``add`` -- compiles the request's grammar (cached by schema string) on admission;
* ``fill`` -- builds the next-token bitmask for the constrained rows of a batch that is
  about to be forwarded (the sampler applies it to the logits before argmax/sampling);
* ``accept`` -- advances the matcher with the sampled token when the batch drains;
* ``remove`` -- drops the matcher when the request finishes or is aborted.

Because the mask for step N+1 depends on the token sampled at step N, a batch that carries
a constrained request must be drained before the next one is scheduled (see
``Scheduler.overlap_loop``). Unconstrained traffic keeps full overlap scheduling.

xgrammar is imported lazily, on the first constrained request, so it stays an optional
dependency: without it a constrained request fails with a clear error and nothing else
changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch

from freetoken.constrained import GrammarError, has_constraint  # noqa: F401 (re-export)
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.core import Req, SamplingParams  # noqa: F401

logger = init_logger(__name__)


class GrammarManager:
    def __init__(self, tokenizer: Any, vocab_size: int, stop_token_ids: List[int], device: torch.device):
        self._tokenizer = tokenizer
        self._vocab_size = vocab_size
        self._stop_token_ids = sorted(int(t) for t in stop_token_ids)
        self._device = device
        self._xgr = None
        self._compiler = None
        self._matchers: Dict[int, Any] = {}
        self._bitmask_cpu: torch.Tensor | None = None

    # -- lifecycle ------------------------------------------------------------------------
    def _ensure_compiler(self) -> None:
        if self._compiler is not None:
            return
        try:
            import xgrammar as xgr
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise GrammarError(
                "response_format json_object/json_schema needs the 'xgrammar' package "
                "(pip install xgrammar)"
            ) from exc
        info = xgr.TokenizerInfo.from_huggingface(
            self._tokenizer, vocab_size=self._vocab_size, stop_token_ids=self._stop_token_ids
        )
        self._xgr = xgr
        # xgrammar caches compiled grammars by (schema, options) internally.
        self._compiler = xgr.GrammarCompiler(info, max_threads=8, cache_enabled=True)
        logger.info_rank0(
            "grammar: xgrammar ready (vocab=%d, stop_tokens=%s)", self._vocab_size, self._stop_token_ids
        )

    def add(self, uid: int, params: "SamplingParams") -> None:
        """Compile and attach a matcher for ``uid``. Raises GrammarError on a bad schema."""
        self._ensure_compiler()
        try:
            if params.json_schema is not None:
                # Compact, deterministic layout: no free whitespace (a model can otherwise
                # loop on newlines/spaces inside a structure), ", " / ": " separators
                # (what the model writes naturally).
                compiled = self._compiler.compile_json_schema(
                    params.json_schema, any_whitespace=False
                )
            else:
                compiled = self._compiler.compile_builtin_json_grammar()
        except Exception as exc:  # noqa: BLE001 -- surface any compile failure to the client
            raise GrammarError(f"invalid response_format schema: {exc}") from exc
        self._matchers[uid] = self._xgr.GrammarMatcher(compiled)

    def remove(self, uid: int) -> None:
        self._matchers.pop(uid, None)

    # -- per step -------------------------------------------------------------------------
    def is_constrained(self, uid: int) -> bool:
        return uid in self._matchers

    def batch_has_constrained(self, reqs: List["Req"]) -> bool:
        if not self._matchers:
            return False
        from .prefill import ChunkedReq

        return any(r.uid in self._matchers and not isinstance(r, ChunkedReq) for r in reqs)

    def fill(self, reqs: List["Req"]) -> Tuple[torch.Tensor, List[int]] | None:
        """Bitmask (on device) + the batch rows it covers, or None if no row is constrained."""
        if not self._matchers:
            return None
        from .prefill import ChunkedReq

        rows = [
            i for i, r in enumerate(reqs)
            if r.uid in self._matchers and not isinstance(r, ChunkedReq)
        ]
        if not rows:
            return None
        xgr = self._xgr
        n = len(reqs)
        if self._bitmask_cpu is None or self._bitmask_cpu.shape[0] < n:
            self._bitmask_cpu = xgr.allocate_token_bitmask(max(n, 8), self._vocab_size).pin_memory()
        bitmask = self._bitmask_cpu[:n]
        for i in rows:
            self._matchers[reqs[i].uid].fill_next_token_bitmask(bitmask, i)
        return bitmask.to(self._device, non_blocking=True), rows

    def accept(self, uid: int, token: int) -> Tuple[bool, bool]:
        """Advance the matcher. Returns (accepted, terminated)."""
        m = self._matchers.get(uid)
        if m is None:
            return True, False
        ok = m.accept_token(token)
        if not ok:
            logger.warning_rank0("grammar: request %d sampled a token the grammar rejects (%d)", uid, token)
            return False, True
        return True, m.is_terminated()
