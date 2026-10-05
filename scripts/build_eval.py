#!/usr/bin/env python3
"""Write mutated eval artifacts under hma_sft/data/ only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hma_sft.adapters.bfcl import BFCLAdapter
from hma_sft.adapters.metatool import HMA_SCENES, MetaToolAdapter
from hma_sft.adapters.tau2 import Tau2Adapter
from hma_sft.core.lexicon import (
    assert_split_surfaces_disjoint,
    bench_dir,
    collect_names,
    load_json,
)
from hma_sft.core.spec import tool_name_of

# eval lexicon basename per bench; also the key used for the train∩eval held-out check.
EVAL_LEXICON = {"bfcl": "families", "metatool": "harness", "tau2": "impl"}


def eval_surfaces(bench: str) -> list[str]:
    """Sorted eval-surface names declared by the bench's eval lexicon."""
    name = EVAL_LEXICON[bench]
    path = bench_dir(bench) / "eval" / f"{name}.json"
    if not path.is_file():
        return []
    return sorted(collect_names(load_json(path)))


def build_metatool(root: Path, out: Path) -> None:
    ad = MetaToolAdapter(root=root)
    items = [ad.render_mutated(t, "eval") for t in ad.load_eval()]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metatool_H_mix.json"
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    by_scene = {s: [x for x in items if x["scenario"] == s] for s in HMA_SCENES}
    for s, rows in by_scene.items():
        slug = s.replace(" ", "_").replace("&", "and")
        (out / f"metatool_H_mix_{slug}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"MetaTool eval {len(items)} → {path}")
    assert_split_surfaces_disjoint("metatool", EVAL_LEXICON["metatool"], label="metatool")


def build_bfcl(root: Path, out: Path) -> None:
    proto_path = root / "toolmind_bfcl" / "configs" / "bfcl_live_scene_holdout_v1.json"
    proto = json.loads(proto_path.read_text(encoding="utf-8")) if proto_path.is_file() else {}
    ad = BFCLAdapter(root=root)
    samples = []
    for traj in ad.load_eval()[:5]:
        spec, cat, _ = ad.mutate(traj, "eval")
        official = {t.name for t in traj.catalog}
        surface = {tool_name_of(t) for t in cat}
        leak = official & surface & set(spec.dropped_tools)
        if leak:
            raise AssertionError(f"{traj.id} official names still in catalog: {sorted(leak)}")
        samples.append(
            {
                "id": traj.id,
                "names": [tool_name_of(t) for t in cat],
                "dropped": spec.dropped_tools,
                "rename": spec.tool_name_map,
            }
        )
    payload = {
        "env": "H_mix_full",
        "n_test": proto.get("n_test") or len(proto.get("test_ids") or []),
        "test_ids": proto.get("test_ids") or [],
        "note": "score live-scene 123 only",
        "samples": samples,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "bfcl_H_mix_full_123.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"BFCL eval 123 → {path} samples={len(samples)}")
    assert_split_surfaces_disjoint("bfcl", EVAL_LEXICON["bfcl"], label="bfcl")


def build_tau2(root: Path, out: Path) -> None:
    ad = Tau2Adapter(root=root)
    spec, cat, _ = ad.mutate(ad.load_eval()[0], "eval")
    payload = {
        "env": spec.env_id,
        "catalog": cat,
        "names": [tool_name_of(t) for t in cat],
        "dropped": spec.dropped_tools,
        "rename": spec.tool_name_map,
        "suite": spec.suite_map,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "tau2_H_mix_impl_catalog.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"τ² catalog {len(payload['names'])} → {path}")
    assert_split_surfaces_disjoint("tau2", EVAL_LEXICON["tau2"], label="tau2")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=["bfcl", "metatool", "tau2", "all"], default="all")
    p.add_argument("--root", default=str(ROOT))
    args = p.parse_args()
    root = Path(args.root)
    out = root / "hma_sft" / "data" / "eval"
    if args.bench in ("bfcl", "all"):
        build_bfcl(root, out)
    if args.bench in ("metatool", "all"):
        build_metatool(root, out)
    if args.bench in ("tau2", "all"):
        build_tau2(root, out)

    benches = ["bfcl", "metatool", "tau2"] if args.bench == "all" else [args.bench]
    surfaces = {b: eval_surfaces(b) for b in benches}
    out.mkdir(parents=True, exist_ok=True)
    spath = out / "_surfaces.json"
    spath.write_text(
        json.dumps(surfaces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    total = sum(len(v) for v in surfaces.values())
    print(f"eval surfaces {total} → {spath}")


if __name__ == "__main__":
    main()
