"""Wire-level helpers for grammar-constrained decoding (OpenAI ``response_format``).

Kept free of scheduler/engine imports so the API server, the sampler and the scheduler can
all use it. The per-request matcher state lives in ``freetoken.scheduler.grammar``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, List, Tuple

if TYPE_CHECKING:
    import torch

    from freetoken.core import SamplingParams


class GrammarError(ValueError):
    pass


def has_constraint(params: "SamplingParams | None") -> bool:
    return params is not None and (params.json_schema is not None or params.json_object)


def apply_bitmask(logits: "torch.Tensor", bitmask: "torch.Tensor", rows: List[int]) -> None:
    import xgrammar as xgr

    indices = None if len(rows) == logits.shape[0] else rows
    xgr.apply_token_bitmask_inplace(logits, bitmask, indices=indices)


def response_format_to_constraint(response_format: dict | None) -> Tuple[str | None, bool]:
    """Map an OpenAI / llama.cpp-server ``response_format`` to (json_schema string, json_object).

    Accepted shapes:
      {"type": "text"} / None                                -> (None, False)   unconstrained
      {"type": "json_schema", "json_schema": {"schema": {...}}} (OpenAI)
      {"type": "json_schema", "schema": {...}}                  (lenient)
      {"type": "json_object", "schema": {...}}                  (llama.cpp server) -> schema
      {"type": "json_object"}                                   -> any JSON value/object
    Raises GrammarError for an unknown type or a malformed schema field.
    """
    if response_format is None:
        return None, False
    if not isinstance(response_format, dict):
        raise GrammarError("response_format must be an object")
    rtype = response_format.get("type")
    if rtype in (None, "text"):
        return None, False
    if rtype not in ("json_schema", "json_object"):
        raise GrammarError(f"unsupported response_format type {rtype!r}")
    schema: Any = None
    if rtype == "json_schema":
        js = response_format.get("json_schema")
        if isinstance(js, dict):
            schema = js.get("schema", js.get("schema_"))
        if schema is None:
            schema = response_format.get("schema")
        if schema is None:
            raise GrammarError("response_format json_schema requires json_schema.schema")
    else:
        schema = response_format.get("schema")
    if schema is None:
        return None, True
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError as exc:
            raise GrammarError(f"response_format schema is not valid JSON: {exc}") from exc
    if not isinstance(schema, (dict, bool)):
        raise GrammarError("response_format schema must be a JSON object")
    if schema is True or schema == {}:
        return None, True
    # Keep the client's property order: xgrammar emits properties in schema order, and clients
    # rely on it (e.g. a "reasoning" field placed before the decision fields).
    return json.dumps(schema, separators=(",", ":")), False


def validate_schema_string(schema: str) -> None:
    """Cheap tokenizer-free check so a bad schema is a 400 at the API, not a scheduler error.
    No-op when xgrammar is not installed (the scheduler then reports the missing package)."""
    try:
        import xgrammar as xgr
    except ImportError:
        return
    try:
        xgr.Grammar.from_json_schema(schema, any_whitespace=False)
    except Exception as exc:  # noqa: BLE001
        raise GrammarError(f"invalid response_format schema: {exc}") from exc
