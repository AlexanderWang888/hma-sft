"""Eval surfaces must not keep official gold names."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hma_sft.adapters.bfcl import BFCLAdapter
from hma_sft.adapters.metatool import MetaToolAdapter
from hma_sft.adapters.tau2 import OFFICIAL_13, Tau2Adapter
from hma_sft.core.lexicon import LEX
from hma_sft.core.spec import tool_name_of


def test_bfcl_eval_samples_hide_dropped() -> None:
    if not (LEX / "bfcl" / "eval" / "families.json").is_file():
        print("skip bfcl: lexicon not generated")
        return
    ad = BFCLAdapter(root=ROOT)
    rows = ad.load_eval()
    if not rows:
        return
    for traj in rows[:8]:
        spec, cat, _ = ad.mutate(traj, "eval")
        names = {tool_name_of(t) for t in cat}
        leak = names & spec.dropped_name_set()
        assert not leak, (traj.id, leak)


def test_metatool_eval_hides_official() -> None:
    if not (LEX / "metatool" / "eval" / "harness.json").is_file():
        print("skip metatool: lexicon not generated")
        return
    ad = MetaToolAdapter(root=ROOT)
    rows = ad.load_eval()
    if not rows:
        return
    for traj in rows[:20]:
        item = ad.render_mutated(traj, "eval")
        official = {t.name for t in traj.catalog}
        leak = official & set(item["candidates"])
        assert not leak, (traj.id, leak)
        assert item["route_kind"] in {"cover", "rename", "fold", "plain"}


def test_tau2_eval_hides_official() -> None:
    if not (LEX / "tau2" / "eval" / "impl.json").is_file():
        print("skip tau2: lexicon not generated")
        return
    ad = Tau2Adapter(root=ROOT)
    spec, cat, _ = ad.mutate(ad.load_eval()[0], "eval")
    names = {tool_name_of(t) for t in cat}
    assert not (names & set(OFFICIAL_13))
    assert spec.near_tools


if __name__ == "__main__":
    test_bfcl_eval_samples_hide_dropped()
    test_metatool_eval_hides_official()
    test_tau2_eval_hides_official()
    print("ok")
