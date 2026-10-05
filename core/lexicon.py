"""Load LLM-generated eval lexicons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Set

ROOT = Path(__file__).resolve().parents[1]
LEX = ROOT / "lexicons"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def bench_dir(bench: str) -> Path:
    return LEX / bench


def load_split(bench: str, name: str, split: str) -> Any:
    path = bench_dir(bench) / split / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing. Generate with OPENAI_API_KEY and "
            "python3 hma_sft/scripts/generate_mix.py"
        )
    return load_json(path)


def collect_names(obj: Any, acc: Set[str] | None = None) -> Set[str]:
    acc = acc if acc is not None else set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in {"name", "cover_name", "rename_name", "surface"} and isinstance(v, str) and v:
                acc.add(v)
            collect_names(v, acc)
    elif isinstance(obj, list):
        for x in obj:
            collect_names(x, acc)
    return acc


def assert_disjoint(train_names: Iterable[str], eval_names: Iterable[str], *, label: str) -> None:
    leak = set(train_names) & set(eval_names)
    if leak:
        raise AssertionError(f"{label} train∩eval names: {sorted(leak)[:20]}")


def assert_split_surfaces_disjoint(bench: str, name: str, *, label: str | None = None) -> None:
    """Enforce held-out surfaces: train and eval mix surfaces must not overlap.

    Only runs when both `<bench>/train/<name>.json` and `<bench>/eval/<name>.json`
    exist. This package generates eval lexicons; a train family may be dropped in
    from the training pipeline, at which point this check becomes active.
    """
    train_path = bench_dir(bench) / "train" / f"{name}.json"
    eval_path = bench_dir(bench) / "eval" / f"{name}.json"
    if not (train_path.is_file() and eval_path.is_file()):
        return
    assert_disjoint(
        collect_names(load_json(train_path)),
        collect_names(load_json(eval_path)),
        label=label or bench,
    )
