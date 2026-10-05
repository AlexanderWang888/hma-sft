"""Eval mix surfaces must not reuse official gold names."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hma_sft.adapters.tau2 import OFFICIAL_13
from hma_sft.core.lexicon import (
    LEX,
    assert_split_surfaces_disjoint,
    collect_names,
    load_split,
)


def _has(bench: str, name: str) -> bool:
    return (LEX / bench / "eval" / f"{name}.json").is_file()


def _has_split(bench: str, name: str, split: str) -> bool:
    return (LEX / bench / split / f"{name}.json").is_file()


def _check_train_eval(bench: str, name: str) -> None:
    """Held-out principle (paper §3): train and eval surfaces must be disjoint.

    Inert until a train family lexicon is dropped in from the training pipeline.
    """
    if not (_has_split(bench, name, "train") and _has_split(bench, name, "eval")):
        print(f"skip {bench}: train∩eval needs both train and eval lexicons")
        return
    assert_split_surfaces_disjoint(bench, name, label=bench)


def test_bfcl_eval_vs_official() -> None:
    if not _has("bfcl", "families"):
        print("skip bfcl: lexicon not generated")
        return
    seed = json.loads((ROOT / "hma_sft" / "seeds" / "bfcl_official.json").read_text())
    official = set(seed.get("official_tools") or [])
    mix = collect_names(load_split("bfcl", "families", "eval"))
    leak = official & mix
    assert not leak, leak


def test_tau2_eval_vs_official() -> None:
    if not _has("tau2", "impl"):
        print("skip tau2: lexicon not generated")
        return
    mix = collect_names(load_split("tau2", "impl", "eval"))
    leak = set(OFFICIAL_13) & mix
    assert not leak, leak


def test_metatool_eval_vs_official() -> None:
    if not _has("metatool", "harness"):
        print("skip metatool: lexicon not generated")
        return
    seed = json.loads((ROOT / "hma_sft" / "seeds" / "metatool_official.json").read_text())
    official = set()
    scenes = seed.get("scenes") or {}
    if isinstance(scenes, dict):
        for scene in scenes.values():
            official.update(scene.get("fold_from") or [])
            official.update(scene.get("official_cover_keys") or [])
    elif isinstance(scenes, list):
        for scene in scenes:
            official.update((scene.get("cover") or {}).keys())
            official.update((scene.get("fold") or {}).keys())
    mix = collect_names(load_split("metatool", "harness", "eval"))
    leak = official & mix
    assert not leak, leak


def test_bfcl_train_eval_disjoint() -> None:
    _check_train_eval("bfcl", "families")


def test_tau2_train_eval_disjoint() -> None:
    _check_train_eval("tau2", "impl")


def test_metatool_train_eval_disjoint() -> None:
    _check_train_eval("metatool", "harness")


if __name__ == "__main__":
    test_bfcl_eval_vs_official()
    test_tau2_eval_vs_official()
    test_metatool_eval_vs_official()
    test_bfcl_train_eval_disjoint()
    test_tau2_train_eval_disjoint()
    test_metatool_train_eval_disjoint()
    print("ok")
