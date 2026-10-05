#!/usr/bin/env python3
"""LLM-generate eval mix from official seeds only.

Requires OPENAI_API_KEY. Does not ship names or implementations.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hma_sft.gen.openai_client import chat_completions, chat_json

PKG = ROOT / "hma_sft"
SEED_TAU2 = PKG / "seeds" / "tau2_official.json"
SEED_MT = PKG / "seeds" / "metatool_official.json"
SEED_BFCL = PKG / "seeds" / "bfcl_official.json"
GEN_IMPL = PKG / "runtime" / "generated" / "tau2_impl.py"

SYSTEM = (
    "You generate HMA-SFT mix artifacts. Operators: "
    "cover, fold/suite, rename, arg_rename, reshape, near, far. "
    "Never emit official gold names as working surfaces. "
    "Pick exactly one name per mapping — no candidate lists for later random choice. "
    "Return only the requested format."
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)


def generate_tau2() -> None:
    seed = _load_json(SEED_TAU2)
    eval_obj = chat_json(
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    "Create the EVAL lexicon JSON for τ² telecom H_mix_impl.\n"
                    f"Official seeds:\n{json.dumps(seed, indent=2)}\n"
                    "Schema: official_tools (same list), cover {official:{surface,arg_map,description,parameters}}, "
                    "suite {surface:{members,action_param,action_map,description,parameters}}, "
                    "rename {official:surface}, arg_rename {official:{old:new}}, "
                    "reshape {official:{widen:{param:description},optional:{name:schema}}}, "
                    "near [openai tools], far [openai tools].\n"
                    "Every cover/suite surface MUST carry a full LLM-written parameters JSON schema "
                    "(type/properties/required) and an LLM-written description. "
                    "Every reshape widen value is an LLM-written parameter description.\n"
                    "Exactly one surface per official tool. 3 near, 2 far.\n"
                    "JSON object only."
                ),
            },
        ],
        max_tokens=8192,
    )
    eval_obj["_doc"] = "LLM-generated eval family."
    eval_obj["official_tools"] = list(seed["official_tools"])
    _dump(PKG / "lexicons" / "tau2" / "eval" / "impl.json", eval_obj)

    official_src = (
        ROOT / "tau2_hma" / "vendor" / "tau2-bench-offline" / "src"
        / "tau2" / "domains" / "telecom" / "tools.py"
    )
    src = official_src.read_text(encoding="utf-8") if official_src.is_file() else ""
    if len(src) > 24000:
        src = src[:24000]
    raw = chat_completions(
        [
            {"role": "system", "content": SYSTEM + " Output Python only."},
            {
                "role": "user",
                "content": (
                    "Write Python module with class GeneratedTau2Mix. "
                    "Each mix surface is @is_tool. Use self.db; do not call official @is_tool methods. "
                    "Near/far raise ValueError. No official method names.\n"
                    f"Lexicon:\n{json.dumps(eval_obj, indent=2)}\n"
                    f"Official seeds:\n{json.dumps(seed, indent=2)}\n"
                    f"Official source:\n{src}\n"
                ),
            },
        ],
        max_tokens=8192,
        temperature=0.1,
    )
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:python)?\n", "", text)
        text = re.sub(r"\n```$", "", text)
    if "class GeneratedTau2Mix" not in text:
        raise SystemExit("OpenAI did not return GeneratedTau2Mix")
    GEN_IMPL.parent.mkdir(parents=True, exist_ok=True)
    GEN_IMPL.write_text(
        "# LLM-generated. Do not hand-edit.\n\n" + text + "\n",
        encoding="utf-8",
    )
    print(f"wrote {GEN_IMPL}", flush=True)


def generate_metatool() -> None:
    seed = _load_json(SEED_MT)
    eval_obj = chat_json(
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    "Create MetaTool EVAL harness JSON. Scenes: Finance Staff, Housewife, "
                    "artists&designers, elders.\n"
                    "Each scene: merge{from,name,description}, covers{official:{name,description}} "
                    "(8 official tools, first 4 cover / last 4 rename), near[2], far[2], "
                    "arg_rename{official:{old_param:new_param}}, reshape{official:{widen:{param:description},optional:{name:schema}}}.\n"
                    "MetaTool tools carry no parameters, so arg_rename/reshape may be empty objects {}; "
                    "keep the keys present for protocol completeness.\n"
                    f"Official seeds:\n{json.dumps(seed, indent=2)}\n"
                    "Exactly one new name per official tool. JSON object only."
                ),
            },
        ],
        max_tokens=8192,
    )
    eval_obj["_doc"] = "LLM-generated MetaTool eval harness."
    _dump(PKG / "lexicons" / "metatool" / "eval" / "harness.json", eval_obj)


def generate_bfcl() -> None:
    seed = _load_json(SEED_BFCL)
    eval_fam = chat_json(
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": (
                    "Create BFCL EVAL families JSON:\n"
                    "{tools:{official:{cover_name, rename_name, arg_map, widen{arg,description}, "
                    "optional{name,schema}, cover_descriptions}}, scene_near:{scene:[openai tools]}, far:[openai tools]}.\n"
                    "The widen description and the optional schema are LLM-written artifacts; "
                    "every string must be authored by you, not copied from a canned example.\n"
                    "Exactly one cover_name and one rename_name per official tool. No alias lists.\n"
                    f"Official tool names: {seed.get('official_tools')}\n"
                    f"Scenes: {seed.get('scenes')}\n"
                    "JSON object only."
                ),
            },
        ],
        max_tokens=8192,
    )
    eval_fam["_comment"] = "LLM-generated EVAL families. One name per slot."
    _dump(PKG / "lexicons" / "bfcl" / "eval" / "families.json", eval_fam)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench", choices=["tau2", "metatool", "bfcl", "all"], default="all")
    args = p.parse_args()
    if args.bench in ("tau2", "all"):
        generate_tau2()
    if args.bench in ("metatool", "all"):
        generate_metatool()
    if args.bench in ("bfcl", "all"):
        generate_bfcl()
    print("done. next: PYTHONPATH=. python3 hma_sft/scripts/build_eval.py", flush=True)


if __name__ == "__main__":
    main()
