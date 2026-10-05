"""Apply MutationSpec to catalogs / calls and back-map predictions."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Tuple

from hma_sft.core.spec import MutationSpec, function_block, lookup_tool_keyed, tool_name_of
from hma_sft.core.types import Call

REJECT_PREFIX = "_hma_reject_"


def parse_arguments(args: Any) -> Tuple[Dict[str, Any], bool]:
    if isinstance(args, str):
        if not args.strip():
            return {}, True
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {}, True
        return parsed if isinstance(parsed, dict) else {}, True
    if isinstance(args, dict):
        return dict(args), False
    return {}, False


def serialize_arguments(args: Dict[str, Any], as_string: bool) -> Any:
    if as_string:
        return json.dumps(args, ensure_ascii=False)
    return args


def apply_widen_to_prop(prop: Dict[str, Any], arg: str, desc: str) -> None:
    """Widen stays string-typed (BFCL). Description text is LLM-generated."""
    if not str(desc or "").strip():
        raise ValueError(
            f"widen description for {arg!r} must be LLM-generated "
            "(empty/missing); run scripts/generate_mix.py"
        )
    raw_t = prop.get("type")
    if isinstance(raw_t, list):
        prop["type"] = next((str(x) for x in raw_t if x and x != "array"), "string")
    elif not isinstance(raw_t, str) or not raw_t:
        prop["type"] = "string"
    prop["description"] = desc


def apply_to_schema(tools: List[Dict[str, Any]], spec: MutationSpec) -> List[Dict[str, Any]]:
    dropped = spec.dropped_name_set()
    kept: List[Dict[str, Any]] = []
    for raw in tools:
        t = copy.deepcopy(raw)
        fn = function_block(t)
        cname = str(fn.get("name") or "")
        if cname in dropped or cname.replace(".", "_") in dropped:
            continue
        amap = lookup_tool_keyed(spec.arg_name_map, cname) or {}
        params = fn.get("parameters")
        if amap and isinstance(params, dict) and isinstance(params.get("properties"), dict):
            params["properties"] = {amap.get(k, k): v for k, v in params["properties"].items()}
            required = params.get("required")
            if isinstance(required, list):
                params["required"] = [amap.get(r, r) for r in required]
        wargs = lookup_tool_keyed(spec.widen_arg_map, cname) or []
        if wargs:
            wdescs = lookup_tool_keyed(spec.widen_desc_map, cname) or {}
            params = fn.get("parameters")
            props = params.get("properties") if isinstance(params, dict) else None
            if isinstance(props, dict):
                for arg in wargs:
                    surf = amap.get(arg, arg) if amap else arg
                    prop = props.get(surf)
                    if isinstance(prop, dict):
                        apply_widen_to_prop(prop, arg, str(wdescs.get(arg) or ""))
        opts = lookup_tool_keyed(spec.optional_props, cname) or {}
        if opts:
            params = fn.setdefault("parameters", {})
            if not isinstance(params, dict):
                fn["parameters"] = params = {}
            props = params.setdefault("properties", {})
            if isinstance(props, dict):
                for oname, oschema in opts.items():
                    if oname not in props:
                        props[oname] = copy.deepcopy(oschema)
        desc = lookup_tool_keyed(spec.desc_map, cname)
        if desc is not None:
            fn["description"] = desc
        mapped = spec.tool_name_map.get(cname) or lookup_tool_keyed(spec.tool_name_map, cname)
        if mapped:
            fn["name"] = mapped
        kept.append(t)
    extras = [copy.deepcopy(d) for d in spec.near_tools + spec.far_tools]
    added = [copy.deepcopy(d) for d in spec.added_tools]
    return kept + added + extras


def apply_to_tool_call(call: Dict[str, Any], spec: MutationSpec) -> Dict[str, Any]:
    call = copy.deepcopy(call)
    cname = str(call.get("name") or "")
    args, was_str = parse_arguments(call.get("arguments"))
    if cname in spec.suite_map:
        call["name"] = spec.suite_map[cname]
        call["arguments"] = serialize_arguments(args, was_str)
        return call
    amap = lookup_tool_keyed(spec.arg_name_map, cname) or {}
    if amap:
        args = {amap.get(k, k): v for k, v in args.items()}
    if cname in spec.tool_name_map:
        call["name"] = spec.tool_name_map[cname]
    else:
        mapped = lookup_tool_keyed(spec.tool_name_map, cname)
        if mapped:
            call["name"] = mapped
    call["arguments"] = serialize_arguments(args, was_str)
    return call


def apply_to_calls(calls: List[Call], spec: MutationSpec) -> List[Call]:
    out: List[Call] = []
    for c in calls:
        mapped = apply_to_tool_call({"name": c.name, "arguments": c.arguments}, spec)
        args, _ = parse_arguments(mapped.get("arguments"))
        out.append(Call(name=str(mapped.get("name") or ""), arguments=args))
    return out


def back_map_call(
    call: Dict[str, Any],
    spec: MutationSpec,
    *,
    emit_fc_underscore: bool = False,
) -> Dict[str, Any]:
    """Surface → canonical. Distractors and dropped official names are rejected."""
    call = copy.deepcopy(call)
    sname = str(call.get("name") or "")
    if sname in spec.distractor_name_set() or sname in spec.dropped_name_set():
        call["name"] = f"{REJECT_PREFIX}{sname.replace('.', '_')}"
        return call
    cname = spec.resolve_canonical_tool(sname)
    args, was_str = parse_arguments(call.get("arguments"))
    s2c = spec.surface_to_canonical_args(cname)
    if s2c:
        args = {s2c.get(k, k): v for k, v in args.items()}
    opts = lookup_tool_keyed(spec.optional_props, cname) or {}
    if opts:
        args = {k: v for k, v in args.items() if k not in opts}
    wargs = lookup_tool_keyed(spec.widen_arg_map, cname) or []
    for arg in wargs:
        val = args.get(arg)
        if isinstance(val, list):
            args[arg] = val[0] if val else val
    if emit_fc_underscore and "." in cname:
        cname = cname.replace(".", "_")
    call["name"] = cname
    call["arguments"] = serialize_arguments(args, was_str)
    return call


def catalog_names(tools: List[Dict[str, Any]]) -> List[str]:
    return [tool_name_of(t) for t in tools if tool_name_of(t)]
