#!/usr/bin/env python3
"""Print one mutated catalog per bench (does not write old experiment dirs)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hma_sft.adapters.bfcl import BFCLAdapter
from hma_sft.adapters.metatool import MetaToolAdapter
from hma_sft.adapters.tau2 import Tau2Adapter
from hma_sft.core.spec import tool_name_of


def _preview_bfcl(root: Path, split: str) -> None:
    ad = BFCLAdapter(root=root)
    rows = ad.load_eval() if split == "eval" else ad.load_train()
    if not rows:
        print("BFCL: no rows (missing sharegpt jsonl)")
        return
    traj = rows[0]
    spec, cat, calls = ad.mutate(traj, split)
    print(f"== BFCL {traj.id} {spec.env_id} ==")
    print("catalog:", [tool_name_of(t) for t in cat])
    print("dropped:", spec.dropped_tools)
    print("rename:", spec.tool_name_map)
    print("calls:", [{"name": c.name, "arguments": c.arguments} for c in calls])


def _preview_metatool(root: Path, split: str) -> None:
    ad = MetaToolAdapter(root=root)
    rows = ad.load_eval()
    if not rows:
        print("MetaTool: no official occupation json")
        return
    item = ad.render_mutated(rows[0], split)
    print(f"== MetaTool {item['id']} {item['env']} ==")
    print("candidates:", item["candidates"])
    print("gold:", item["gold"], "kind:", item["route_kind"])


def _preview_tau2(root: Path, split: str) -> None:
    ad = Tau2Adapter(root=root)
    rows = ad.load_eval()
    spec, cat, _ = ad.mutate(rows[0], split)
    names = [tool_name_of(t) for t in cat]
    print(f"== τ² {spec.env_id} {split} ==")
    print("catalog:", names)
    official = set(rows[0].extra.get("lex", {}).get("official_tools") or [])
    leak = official & set(names)
    print("official names still in catalog:", sorted(leak) or "none")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=["bfcl", "metatool", "tau2", "all"], default="all")
    p.add_argument("--split", choices=["eval"], default="eval")
    p.add_argument("--root", default=str(ROOT))
    args = p.parse_args()
    root = Path(args.root)
    if args.bench in ("bfcl", "all"):
        _preview_bfcl(root, args.split)
    if args.bench in ("metatool", "all"):
        _preview_metatool(root, args.split)
    if args.bench in ("tau2", "all"):
        _preview_tau2(root, args.split)


if __name__ == "__main__":
    main()
