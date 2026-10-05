"""τ² telecom adapter: extra near; env-level mix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from hma_sft.adapters.base import DatasetAdapter
from hma_sft.core.lexicon import load_split
from hma_sft.core.types import Call, Tool, Trajectory

DEFAULT_TRAIN = Path("tau2_hma/data/original/telecom_sharegpt.jsonl")
OFFICIAL_13 = [
    "get_customer_by_phone",
    "get_customer_by_id",
    "get_customer_by_name",
    "get_details_by_id",
    "get_bills_for_customer",
    "suspend_line",
    "resume_line",
    "get_data_usage",
    "enable_roaming",
    "disable_roaming",
    "transfer_to_human_agents",
    "refuel_data",
    "send_payment_request",
]


def _openai(name: str, desc: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": desc, "parameters": params},
    }


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
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


class Tau2Adapter(DatasetAdapter):
    name = "tau2"

    def __init__(self, train_path: Path | None = None, *, root: Path | None = None):
        self.root = Path(root) if root else Path.cwd()
        self.train_path = Path(train_path) if train_path else self.root / DEFAULT_TRAIN

    def load_train(self) -> List[Trajectory]:
        out: List[Trajectory] = []
        for i, row in enumerate(_load_jsonl(self.train_path)):
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
            catalog = []
            for t in tools:
                fn = t.get("function") if isinstance(t.get("function"), dict) else t
                catalog.append(
                    Tool(
                        name=str(fn.get("name") or ""),
                        description=str(fn.get("description") or ""),
                        parameters=fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {},
                        raw=t,
                    )
                )
            out.append(
                Trajectory(
                    id=str(row.get("id") or f"tau2_{i}"),
                    catalog=catalog,
                    calls=calls,
                    extra={"row": row},
                )
            )
        return out

    def load_eval(self) -> List[Trajectory]:
        """Eval tasks live in tau2-bench; catalog is the official 13-tool set."""
        lex = load_split("tau2", "impl", "eval")
        catalog = [
            Tool(name=n, description="", parameters={})
            for n in lex.get("official_tools") or OFFICIAL_13
        ]
        return [Trajectory(id="telecom_catalog", catalog=catalog, extra={"lex": lex})]

    def plan_context(self, traj: Trajectory, split: str) -> Dict[str, Any]:
        lex = load_split("tau2", "impl", split)

        def _llm_params(meta: Dict[str, Any], kind: str, surface: str) -> Dict[str, Any]:
            params = meta.get("parameters")
            if not isinstance(params, dict) or not params:
                raise ValueError(
                    f"{kind} {surface!r} missing LLM parameters; run generate_mix.py"
                )
            return params

        covers: Dict[str, Dict[str, Any]] = {}
        for cname, meta in (lex.get("cover") or {}).items():
            surface = str(meta["surface"])
            covers[cname] = {
                "surface": surface,
                "tool": _openai(
                    surface,
                    str(meta.get("description") or ""),
                    _llm_params(meta, "cover", surface),
                ),
            }
        folds = []
        for surface, meta in (lex.get("suite") or {}).items():
            folds.append(
                {
                    "members": list(meta.get("members") or []),
                    "surface": surface,
                    "tool": _openai(
                        surface,
                        str(meta.get("description") or ""),
                        _llm_params(meta, "suite", surface),
                    ),
                }
            )
        reshape = lex.get("reshape") or {}
        widen: Dict[str, List[str]] = {}
        widen_desc: Dict[str, Dict[str, str]] = {}
        optional: Dict[str, Dict[str, Any]] = {}
        for cname, meta in reshape.items():
            w = meta.get("widen") or {}
            if w:
                widen[cname] = list(w.keys())
                widen_desc[cname] = {str(a): str(d) for a, d in w.items()}
            opt = meta.get("optional") or {}
            if opt:
                optional[cname] = dict(opt)
        return {
            "covers": covers,
            "dropped": list(covers) + [m for f in folds for m in f["members"]],
            "rename_map": dict(lex.get("rename") or {}),
            "arg_rename_map": {k: dict(v) for k, v in (lex.get("arg_rename") or {}).items()},
            "widen_arg_map": widen,
            "widen_desc_map": widen_desc,
            "optional_props": optional,
            "folds": folds,
            "near_tools": list(lex.get("near") or []),
            "far_tools": list(lex.get("far") or []),
        }
