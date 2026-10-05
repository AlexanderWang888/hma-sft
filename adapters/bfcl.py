"""BFCL live-scene adapter: H_mix_full."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from hma_sft.adapters.base import DatasetAdapter
from hma_sft.core.lexicon import load_split
from hma_sft.core.spec import function_block, tool_name_of
from hma_sft.core.types import Call, Tool, Trajectory

_SCENE_RULES = (
    ("movies", re.compile(r"^(Movies_|Media_.*Movie|Media_\d+_FindMovies)", re.I)),
    ("events", re.compile(r"^Events_|events_api", re.I)),
    ("music", re.compile(r"^Music_", re.I)),
    ("travel", re.compile(r"^(Travel_|Hotels_|Flights_|Buses_|Trains_|RentalCars_|Airbnb)", re.I)),
    ("weather", re.compile(r"^(Weather_|get_current_weather|weather_|api_weather|weather\.)", re.I)),
    ("services", re.compile(r"^(Services_|get_service_providers|view_service_provider|search_service|open_service)", re.I)),
)
_BANNED_DESC = ("replacement for", "interface note", "no longer performs", "original arguments still work")

DEFAULT_TRAIN = Path("toolmind_bfcl/data/bfcl_live_scene_holdout/train_sharegpt.jsonl")
DEFAULT_EVAL = Path("toolmind_bfcl/data/bfcl_live_scene_holdout/test_sharegpt.jsonl")


def _first(v: Any) -> str:
    if isinstance(v, list):
        return str(v[0]) if v else ""
    return str(v or "")


def _lookup_family(name: str, families: Dict[str, Any]) -> Dict[str, Any] | None:
    tools = families.get("tools") or {}
    if name in tools:
        return tools[name]
    want = name.replace(".", "_")
    for k, v in tools.items():
        if k.replace(".", "_") == want:
            return v
    return None


def _scene_of_name(name: str) -> str | None:
    key = (name or "").replace(".", "_")
    for scene, pat in _SCENE_RULES:
        if pat.search(key):
            return scene
    return None


def _arg_maps_from_families(
    names: List[str], families: Dict[str, Any]
) -> Dict[str, Dict[str, str]]:
    tools = families.get("tools") or {}
    out: Dict[str, Dict[str, str]] = {}
    for n in names:
        fam = _lookup_family(n, families) or tools.get(n) or {}
        amap = fam.get("arg_map") or {}
        if isinstance(amap, dict) and amap:
            out[n] = {str(k): str(v) for k, v in amap.items() if v}
    return out


def _clean_desc(text: str) -> str:
    low = (text or "").lower()
    for bad in _BANNED_DESC:
        if bad in low:
            raise ValueError(f"leaky description: {bad!r}")
    return text


def _build_cover_tool(fn: Dict[str, Any], fam: Dict[str, Any], arg_map: Dict[str, str]) -> Dict[str, Any]:
    t = copy.deepcopy(fn)
    block = function_block(t)
    new_name = str(fam.get("cover_name") or "")
    if not new_name:
        raise ValueError("family missing LLM cover_name; run generate_mix.py")
    block["name"] = new_name
    params = block.setdefault("parameters", {})
    if not isinstance(params, dict):
        block["parameters"] = params = {}
    props = params.setdefault("properties", {})
    if not isinstance(props, dict):
        params["properties"] = props = {}
    opt = fam.get("optional") or {}
    oname = str(opt.get("name") or "")
    if oname and oname not in props:
        oschema = opt.get("schema")
        if not isinstance(oschema, dict) or not oschema:
            raise ValueError("family missing LLM optional schema; run generate_mix.py")
        props[oname] = copy.deepcopy(oschema)
    if arg_map:
        params["properties"] = {arg_map.get(k, k): v for k, v in props.items()}
        required = params.get("required")
        if isinstance(required, list):
            params["required"] = [arg_map.get(r, r) for r in required]
    descs = [str(x) for x in (fam.get("cover_descriptions") or []) if x]
    if descs:
        block["description"] = _clean_desc(descs[0])
        orig = str(fn.get("name") or function_block(fn).get("name") or "")
        if orig and orig in str(block.get("description") or ""):
            block["description"] = str(block["description"]).replace(orig, "this endpoint")
    if t.get("type") != "function" and "function" not in t:
        t = {"type": "function", "function": block}
    return t


def _openai_tool(raw: Dict[str, Any]) -> Tool:
    fn = function_block(raw)
    params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
    return Tool(
        name=str(fn.get("name") or ""),
        description=str(fn.get("description") or ""),
        parameters=params,
        raw=raw,
    )


def _parse_tools(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        return []
    return [t for t in raw if isinstance(t, dict)]


def _parse_fc(value: Any) -> List[Dict[str, Any]]:
    obj = json.loads(value) if isinstance(value, str) else value
    items = obj if isinstance(obj, list) else [obj]
    return [c for c in items if isinstance(c, dict) and c.get("name")]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _row_to_traj(row: Dict[str, Any]) -> Trajectory:
    tools = _parse_tools(row.get("tools"))
    calls: List[Call] = []
    for m in row.get("conversations") or []:
        if m.get("from") != "function_call":
            continue
        for c in _parse_fc(m.get("value")):
            args = c.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append(Call(name=str(c["name"]), arguments=args))
    return Trajectory(
        id=str(row.get("id") or ""),
        query="",
        catalog=[_openai_tool(t) for t in tools],
        calls=calls,
        extra={"row": row, "scene": row.get("scene")},
    )


def gt_tool_names(entry: Dict[str, Any], catalog: List[str]) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()

    def _add(n: str) -> None:
        n = str(n or "").strip()
        if n and n not in seen:
            seen.add(n)
            names.append(n)

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k not in ("ground_truth",) and not str(k).startswith("_"):
                    if isinstance(v, (dict, list)) or "." in k or "_" in k:
                        _add(k)
                _walk(v)
        elif isinstance(obj, list):
            for x in obj:
                if isinstance(x, str):
                    m = re.match(r"^([A-Za-z_][\w.]*)\s*\(", x)
                    if m:
                        _add(m.group(1))
                else:
                    _walk(x)

    _walk(entry.get("ground_truth"))
    cat = set(catalog)
    in_cat = [n for n in names if n in cat]
    return in_cat or names


class BFCLAdapter(DatasetAdapter):
    name = "bfcl"

    def __init__(
        self,
        train_path: Path | None = None,
        eval_path: Path | None = None,
        *,
        root: Path | None = None,
    ):
        self.root = Path(root) if root else Path.cwd()
        self.train_path = Path(train_path) if train_path else self.root / DEFAULT_TRAIN
        self.eval_path = Path(eval_path) if eval_path else self.root / DEFAULT_EVAL

    def load_train(self) -> List[Trajectory]:
        return [_row_to_traj(r) for r in _load_jsonl(self.train_path)]

    def load_eval(self) -> List[Trajectory]:
        return [_row_to_traj(r) for r in _load_jsonl(self.eval_path)]

    def spec_from_evalscope(self, entry: Dict[str, Any], split: str = "eval"):
        fns = entry.get("function") if isinstance(entry.get("function"), list) else []
        catalog = [_openai_tool(t) for t in fns if isinstance(t, dict)]
        names = [t.name for t in catalog]
        gt = gt_tool_names(entry, names)
        calls = [Call(name=n) for n in gt]
        traj = Trajectory(
            id=str(entry.get("id") or ""),
            catalog=catalog,
            calls=calls,
            extra={"scene": entry.get("scene")},
        )
        spec, surface, _ = self.mutate(traj, split)
        return spec, surface

    def plan_context(self, traj: Trajectory, split: str) -> Dict[str, Any]:
        families = load_split("bfcl", "families", split)
        fns = [t.to_openai() for t in traj.catalog]
        names = [t.name for t in traj.catalog]
        gt = list(dict.fromkeys(c.name for c in traj.calls if c.name in set(names)))
        if not gt:
            gt = names[:1]
        arg_map = _arg_maps_from_families(names, families)

        covers: Dict[str, Dict[str, Any]] = {}
        dropped: List[str] = []
        for g in gt:
            fam = _lookup_family(g, families)
            if not fam:
                continue
            raw = next((x for x in fns if tool_name_of(x) == g), None)
            if raw is None:
                continue
            tool = _build_cover_tool(raw, fam, arg_map.get(g) or {})
            covers[g] = {"surface": tool_name_of(tool), "tool": tool}
            dropped.append(g)

        siblings = [n for n in names if n not in set(dropped)]
        used = set(names) | {c["surface"] for c in covers.values()}
        rename_map: Dict[str, str] = {}
        for n in siblings:
            fam = _lookup_family(n, families) or {}
            pick = _first(fam.get("rename_name"))
            if pick and pick not in used:
                rename_map[n] = pick
                used.add(pick)

        widen: Dict[str, List[str]] = {}
        widen_desc: Dict[str, Dict[str, str]] = {}
        optional: Dict[str, Dict[str, Any]] = {}
        for n in names:
            fam = _lookup_family(n, families) or {}
            w = fam.get("widen")
            warg = str(w.get("arg") or "") if isinstance(w, dict) else str(w or "")
            if warg:
                wdesc = str(w.get("description") or "") if isinstance(w, dict) else ""
                if not wdesc.strip():
                    raise ValueError("family missing LLM widen description; run generate_mix.py")
                widen[n] = [warg]
                widen_desc[n] = {warg: wdesc}
            opt = fam.get("optional") or {}
            oname = str(opt.get("name") or "")
            if oname:
                oschema = opt.get("schema")
                if not isinstance(oschema, dict) or not oschema:
                    raise ValueError("family missing LLM optional schema; run generate_mix.py")
                optional[n] = {oname: copy.deepcopy(oschema)}

        far = []
        blocked = set(used)
        for raw in list(families.get("far") or []):
            n = tool_name_of(raw)
            if n and n not in blocked:
                far.append(copy.deepcopy(raw))
                blocked.add(n)
            if len(far) >= 2:
                break

        scene = traj.extra.get("scene") or _scene_of_name(gt[0] if gt else "")
        near = []
        for raw in list((families.get("scene_near") or {}).get(scene or "") or []):
            n = tool_name_of(raw)
            if n and n not in blocked:
                near.append(copy.deepcopy(raw))
                blocked.add(n)

        return {
            "covers": covers,
            "dropped": dropped,
            "rename_map": rename_map,
            "arg_rename_map": arg_map,
            "widen_arg_map": widen,
            "widen_desc_map": widen_desc,
            "optional_props": optional,
            "near_tools": near,
            "far_tools": far,
        }
