"""EvalScope BFCL hook using hma_sft compose."""

from __future__ import annotations

import copy
import json
from contextvars import ContextVar
from typing import Any, Optional

from hma_sft.adapters.bfcl import BFCLAdapter
from hma_sft.core.apply import REJECT_PREFIX, back_map_call
from hma_sft.core.spec import MutationSpec, function_block

_CURRENT: ContextVar[Optional[MutationSpec]] = ContextVar("hma_sft_spec", default=None)
_IN_SCORE: ContextVar[bool] = ContextVar("hma_sft_score", default=False)
_INSTALLED = False
_ADAPTER = BFCLAdapter()


def _has_reject(obj: Any) -> bool:
    if isinstance(obj, dict):
        if str(obj.get("name") or "").startswith(REJECT_PREFIX):
            return True
        return any(_has_reject(v) or str(k).startswith(REJECT_PREFIX) for k, v in obj.items())
    if isinstance(obj, list):
        return any(_has_reject(x) for x in obj)
    return isinstance(obj, str) and obj.startswith(REJECT_PREFIX)


def _back_map_result(model_result: Any, spec: MutationSpec, *, underscore: bool) -> Any:
    if spec is None or spec.is_empty():
        return model_result
    if isinstance(model_result, list):
        if model_result and isinstance(model_result[0], list):
            return [_back_map_result(t, spec, underscore=underscore) for t in model_result]
        return [_back_map_one(x, spec, underscore) for x in model_result]
    return _back_map_one(model_result, spec, underscore)


def _back_map_one(item: Any, spec: MutationSpec, underscore: bool) -> Any:
    if isinstance(item, dict) and len(item) == 1 and "name" not in item:
        sname, args = next(iter(item.items()))
        mapped = back_map_call({"name": sname, "arguments": args}, spec, emit_fc_underscore=underscore)
        return {mapped["name"]: mapped["arguments"]}
    if isinstance(item, dict) and "name" in item:
        return back_map_call(item, spec, emit_fc_underscore=underscore)
    return item


def _align_function_shape(surface: list, orig: list) -> list:
    """Keep EvalScope `function` list in the same wrap as the official entry."""
    want_wrapped = bool(orig) and isinstance(orig[0], dict) and isinstance(orig[0].get("function"), dict)
    out = []
    for t in surface:
        if not isinstance(t, dict):
            continue
        if want_wrapped:
            if isinstance(t.get("function"), dict):
                out.append(t)
            else:
                out.append({"type": "function", "function": t})
        else:
            out.append(function_block(t))
    return out


def _prepare(meta: dict, split: str) -> MutationSpec:
    orig = copy.deepcopy(meta.get("function") or [])
    spec, surface = _ADAPTER.spec_from_evalscope(meta, split)
    if spec.skip:
        meta["_hma_spec"] = spec.to_dict()
        meta["_hma_skip"] = True
        return spec
    meta["_hma_function_orig"] = orig
    meta["function"] = _align_function_shape(surface, orig)
    meta["_hma_spec"] = spec.to_dict()
    return spec


def _restore_catalog(meta: dict) -> dict:
    orig = meta.get("_hma_function_orig")
    if isinstance(orig, list):
        out = copy.deepcopy(meta)
        out["function"] = copy.deepcopy(orig)
        return out
    return meta


def install_bfcl_hooks(*, split: str = "eval") -> None:
    """Patch EvalScope BFCL-v4 for H_mix_full."""
    global _INSTALLED
    if _INSTALLED:
        return
    from evalscope.benchmarks.bfcl.v4.bfcl_v4_adapter import BFCLV4Adapter

    orig_on_inference = BFCLV4Adapter._on_inference
    orig_init_handler = BFCLV4Adapter._init_handler
    orig_match_score = BFCLV4Adapter.match_score

    def _init_handler(self):
        orig_init_handler(self)
        handler = getattr(self, "handler", None)
        if handler is None or getattr(handler, "_hma_sft_wrapped", False):
            return
        orig_exec = handler.decode_execute
        orig_ast = handler.decode_ast

        def decode_execute(result, has_tool_call_tag=False):
            spec = _CURRENT.get()
            if spec is not None and not spec.skip and not _IN_SCORE.get():
                result = _back_map_result(result, spec, underscore=False)
            return orig_exec(result, has_tool_call_tag)

        def decode_ast(result, language, has_tool_call_tag=False):
            spec = _CURRENT.get()
            if spec is not None and not spec.skip:
                result = _back_map_result(result, spec, underscore=True)
            return orig_ast(result, language, has_tool_call_tag)

        handler.decode_execute = decode_execute
        handler.decode_ast = decode_ast
        handler._hma_sft_wrapped = True

    def _on_inference(self, model, sample):
        meta = sample.metadata if isinstance(sample.metadata, dict) else {}
        spec = _prepare(meta, split)
        token = _CURRENT.set(spec)
        try:
            return orig_on_inference(self, model, sample)
        finally:
            _CURRENT.reset(token)

    def match_score(self, original_prediction, filtered_prediction, reference, task_state):
        meta = task_state.metadata if isinstance(getattr(task_state, "metadata", None), dict) else {}
        spec = MutationSpec.from_dict(meta.get("_hma_spec") or {})
        if spec.is_empty() and meta.get("function"):
            spec, _ = _ADAPTER.spec_from_evalscope(
                {"id": meta.get("id"), "function": meta.get("_hma_function_orig") or meta.get("function"),
                 "ground_truth": meta.get("ground_truth")},
                split,
            )
        token = _CURRENT.set(spec)
        score_token = _IN_SCORE.set(True)
        try:
            try:
                model_result = json.loads(filtered_prediction)
            except Exception:
                model_result = filtered_prediction
            mapped = _back_map_result(model_result, spec, underscore=True)
            filtered_prediction = json.dumps(mapped)
            if _has_reject(mapped):
                from evalscope.api.metric import Score

                score = Score(extracted_prediction=filtered_prediction, prediction=original_prediction)
                score.value = {"acc": 0}
                score.metadata = {"hma_sft": True, "reject": True, "valid": False}
                return score
            task_state.metadata = _restore_catalog(meta)
            return orig_match_score(self, original_prediction, filtered_prediction, reference, task_state)
        finally:
            _IN_SCORE.reset(score_token)
            _CURRENT.reset(token)

    BFCLV4Adapter._init_handler = _init_handler
    BFCLV4Adapter._on_inference = _on_inference
    BFCLV4Adapter.match_score = match_score
    _INSTALLED = True
    print("[hma_sft] BFCL EvalScope hooks installed env=H_mix_full", flush=True)
