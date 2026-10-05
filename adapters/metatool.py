"""MetaTool occupation adapter: eval splits covers into cover/rename; train is cover-only."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from hma_sft.adapters.base import DatasetAdapter
from hma_sft.core.lexicon import load_split
from hma_sft.core.types import Call, Tool, Trajectory

HMA_SCENES = (
    "Finance Staff",
    "Housewife",
    "artists&designers",
    "elders",
)
SLUG = {
    "Finance Staff": "finance_staff",
    "Housewife": "housewife",
    "artists&designers": "artists_designers",
    "elders": "elders",
}

HEAD = (
    "You are a helpful AI assistant. Your current task is to choose the appropriate "
    "tool to solve the user's query based on their question. I will provide you with "
    "the user's question and information about the tools.\n"
    "If there is a tool in the list that is applicable to this query, please return "
    "the name of the tool (you can only choose one tool). If there isn't, please "
    "return 'None.' Additionally, you will need to support your answer with a brief "
    "explanation.\n"
    "This is the current production catalog after two products were folded "
    "into one suite."
)

TOOL_RE = re.compile(
    r"\d+\.\s*tool name:\s*([^,\n]+),\s*tool description:\s*(.*?)(?=\n\d+\.\s*tool name:|\n\s*$)",
    re.S,
)
FS_RE = re.compile(r'query:\s*"((?:\\.|[^"\\])*)"\s*tool:\s*([A-Za-z0-9_&+-]+)', re.S)

DEFAULT_OFFICIAL = Path("toolmind_metatool/data/official/Task2_Subtask2_occupation.json")

# EVAL split only: first 4 official tools stay cover; last 4 become standard
# rename. The train split keeps all 8 covers in the cover family (paper
# Table 1, tab:protocol-map: MetaTool trains {cover, fold}, no tool rename).
COVER_KEEP = 4


def _clean_desc(desc: str) -> str:
    d = " ".join((desc or "").split()).strip()
    if len(d) >= 2 and d[0] in "[\"'" and d[-1] in "]\"'":
        inner = d[1:-1].strip()
        if len(inner) >= 2 and inner[0] in "\"'" and inner[-1] in "\"'":
            inner = inner[1:-1]
        d = inner
    return d.strip()


def scene_catalog(prompt: str) -> list[tuple[str, str]]:
    start = prompt.find("[List of Tools with Names and Descriptions Start]")
    end = prompt.find("[List of Tools with Names and Descriptions End]")
    block = prompt[start:end] if start >= 0 and end > start else prompt
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in TOOL_RE.finditer(block):
        name = m.group(1).strip()
        desc = _clean_desc(m.group(2))
        if name and name not in seen:
            seen.add(name)
            out.append((name, desc))
    return out


def fewshots(prompt: str) -> list[tuple[str, str]]:
    start, end = prompt.find("[Examples Start]"), prompt.find("[Examples End]")
    if start < 0 or end <= start:
        return []
    out: list[tuple[str, str]] = []
    for m in FS_RE.finditer(prompt[start:end]):
        q = m.group(1).replace('\\"', '"')
        out.append((q, m.group(2).strip()))
    return out


def render_prompt(query: str, catalog: list[tuple[str, str]], shots: list[tuple[str, str]]) -> str:
    lines = [f"{i}. tool name: {n}, tool description: {d}" for i, (n, d) in enumerate(catalog, 1)]
    tool_block = "\n".join(lines)
    ex = "\n".join(f'query: "{q}" tool: {t}' for q, t in shots)
    return (
        f"{HEAD}\n"
        f"User's Query:\n[User's Query Start]\n{query}\n[User's Query End].\n"
        f"List of Tools with Names and Descriptions:\n"
        f"[List of Tools with Names and Descriptions Start]\n{tool_block}\n\n"
        f"[List of Tools with Names and Descriptions End]\n"
        f"Here are some examples:\n[Examples Start]\n{ex}\n[Examples End]\n\n"
        f'User query: "{query}" tool:'
    )


def _name_desc_tool(name: str, desc: str) -> Dict[str, Any]:
    return {"name": name, "description": desc}


class MetaToolAdapter(DatasetAdapter):
    name = "metatool"

    def __init__(self, official_path: Path | None = None, *, root: Path | None = None):
        self.root = Path(root) if root else Path.cwd()
        self.official_path = Path(official_path) if official_path else self.root / DEFAULT_OFFICIAL

    def _load_items(self) -> List[Dict[str, Any]]:
        data = json.loads(self.official_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            items = data.get("data") or data.get("items") or []
        else:
            items = data
        return [x for x in items if isinstance(x, dict) and x.get("scenario") in HMA_SCENES]

    def _to_traj(self, item: Dict[str, Any]) -> Trajectory:
        cat = scene_catalog(str(item.get("action_prompt") or ""))
        gold = str(item.get("gold") or "")
        return Trajectory(
            id=str(item.get("id") or ""),
            query=str(item.get("query") or ""),
            catalog=[Tool(name=n, description=d) for n, d in cat],
            calls=[Call(name=gold)] if gold else [],
            extra={"item": item, "scene": item.get("scenario"), "fewshots": fewshots(str(item.get("action_prompt") or ""))},
        )

    def load_eval(self) -> List[Trajectory]:
        return [self._to_traj(x) for x in self._load_items()]

    def load_train(self) -> List[Trajectory]:
        # Official test items are eval-only; train jsonl is built elsewhere.
        # Preview / mutate still works on the same occupation catalog.
        return self.load_eval()

    def plan_context(self, traj: Trajectory, split: str) -> Dict[str, Any]:
        spec = load_split("metatool", "harness", split)
        scene = str(traj.extra.get("scene") or "")
        sc = spec[scene]
        official_desc = {t.name: t.description for t in traj.catalog}
        cover_items = list((sc.get("covers") or {}).items())

        # Paper Table 1 (tab:protocol-map), MetaTool columns. Training uses
        # {cover, fold} only — every official tool stays in the cover family
        # (no tool rename, no near/far lures), mirroring the training
        # pipeline, which routes all 8 covers through cover_map. Evaluation
        # uses the full operator set: the first COVER_KEEP covers stay cover
        # and the rest surface as standard tool renames plus near/far lures.
        if split == "train":
            cover_keep, rename_items = cover_items, []
            near: List[Dict[str, Any]] = []
            far: List[Dict[str, Any]] = []
        else:
            cover_keep = cover_items[:COVER_KEEP]
            rename_items = cover_items[COVER_KEEP:]
            near = [_name_desc_tool(str(x["name"]), str(x["description"])) for x in sc.get("near") or []]
            far = [_name_desc_tool(str(x["name"]), str(x["description"])) for x in sc.get("far") or []]

        covers: Dict[str, Dict[str, Any]] = {}
        for orig, meta in cover_keep:
            surface = str(meta["name"])
            desc = str(meta["description"])
            covers[orig] = {
                "surface": surface,
                "tool": _name_desc_tool(surface, desc),
            }

        rename_map = {orig: str(meta["name"]) for orig, meta in rename_items}
        desc_map = {orig: official_desc.get(orig, "") for orig, _ in rename_items}

        merge = sc["merge"]
        members = [str(x) for x in merge["from"]]
        suite_name = str(merge["name"])
        folds = [
            {
                "members": members,
                "surface": suite_name,
                "tool": _name_desc_tool(suite_name, str(merge["description"])),
            }
        ]
        # near / far are set in the split branch above (eval only; the train
        # split uses no distractors, per paper Table 1).

        # Schema-level operators: MetaTool catalogs carry no parameters, so these are
        # usually empty, but the plumbing exists to match the full 7-operator protocol.
        arg_rename_map = {
            str(k): {str(a): str(b) for a, b in (v or {}).items()}
            for k, v in (sc.get("arg_rename") or {}).items()
        }
        widen: Dict[str, List[str]] = {}
        widen_desc: Dict[str, Dict[str, str]] = {}
        optional: Dict[str, Dict[str, Any]] = {}
        for cname, meta in (sc.get("reshape") or {}).items():
            w = meta.get("widen") or {}
            if isinstance(w, dict) and w:
                widen[str(cname)] = [str(x) for x in w]
                widen_desc[str(cname)] = {str(a): str(d) for a, d in w.items()}
            opt = meta.get("optional") or {}
            if opt:
                optional[str(cname)] = dict(opt)

        return {
            "covers": covers,
            "dropped": members + [k for k in covers],
            "rename_map": rename_map,
            "desc_map": desc_map,
            "arg_rename_map": arg_rename_map,
            "widen_arg_map": widen,
            "widen_desc_map": widen_desc,
            "optional_props": optional,
            "folds": folds,
            "near_tools": near,
            "far_tools": far,
        }

    def render_mutated(self, traj: Trajectory, split: str) -> Dict[str, Any]:
        spec, catalog, calls = self.mutate(traj, split)
        pairs = [(str(t.get("name") or ""), str(t.get("description") or "")) for t in catalog]
        gold_official = traj.calls[0].name if traj.calls else ""
        if gold_official in spec.suite_map:
            gold = spec.suite_map[gold_official]
            kind = "fold"
        elif gold_official in spec.tool_name_map:
            gold = spec.tool_name_map[gold_official]
            kind = "cover" if gold_official in spec.dropped_name_set() else "rename"
        else:
            gold = gold_official
            kind = "plain"
        shots = []
        for q, t in traj.extra.get("fewshots") or []:
            if t in spec.suite_map:
                shots.append((q, spec.suite_map[t]))
            elif t in spec.tool_name_map:
                shots.append((q, spec.tool_name_map[t]))
            else:
                raise KeyError(f"{traj.id} few-shot {t} has no mix route")
        prompt = render_prompt(traj.query, pairs, shots)
        official_names = {t.name for t in traj.catalog}
        leak = official_names & {n for n, _ in pairs}
        if leak:
            raise AssertionError(f"{traj.id} official names still in catalog: {sorted(leak)}")
        return {
            "id": traj.id,
            "scenario": traj.extra.get("scene"),
            "query": traj.query,
            "gold": gold,
            "gold_canonical": gold_official,
            "env": spec.env_id,
            "route_kind": kind,
            "candidates": [n for n, _ in pairs],
            "action_prompt": prompt,
            "spec": {
                "tool_name_map": spec.tool_name_map,
                "suite_map": spec.suite_map,
                "dropped_tools": spec.dropped_tools,
            },
        }
