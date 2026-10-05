#!/usr/bin/env python3
"""τ² telecom eval: H_mix_impl. User-sim uses official OpenAI.

Set OPENAI_API_KEY yourself. This package does not ship a key.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAU2 = ROOT / "tau2_hma"
VENDOR = TAU2 / "vendor" / "tau2-bench-offline"
for p in (str(VENDOR / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("TAU2_DATA_DIR", str(VENDOR / "data"))

def _require_openai_key() -> str:
    from hma_sft.gen.openai_client import resolve_api_key

    return resolve_api_key()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "qwen35_4b_base"))
    ap.add_argument("--model-endpoint", default=os.environ.get("ENDPOINT", "http://127.0.0.1:8200/v1"))
    ap.add_argument("--served-model-name", default=os.environ.get("SERVED_MODEL_NAME", "Qwen3.5-4B"))
    ap.add_argument("--harness", default="H_mix_impl", choices=["H_mix_impl", "original"])
    ap.add_argument("--mode", default=os.environ.get("MODE", "full"), choices=["smoke", "small", "full"])
    ap.add_argument("--user-model", default=os.environ.get("OPENAI_MODEL") or os.environ.get("USER_MODEL", "gpt-4.1"))
    ap.add_argument(
        "--user-api-base",
        default=os.environ.get("OPENAI_BASE_URL") or os.environ.get("USER_API_BASE", "https://api.openai.com/v1"),
    )
    ap.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "12")))
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    user_key = _require_openai_key()

    from hma_sft.eval.tau2_patch import patch_tau2_eval

    patch_tau2_eval()

    from tau2_hma.data.eval_official_splits import load_split_ids
    from tau2_hma.eval.run_eval import run_eval

    task_limit = 0
    task_ids = None
    workers = args.num_workers
    if args.mode == "smoke":
        task_limit = int(os.environ.get("TASK_LIMIT_OVERRIDE", "1"))
        workers = 1
    elif args.mode == "small":
        task_ids = load_split_ids("telecom", "small")
    else:
        task_ids = load_split_ids("telecom", "base")

    out = Path(args.out) if args.out else (
        ROOT / "hma_sft" / "outputs" / "tau2" / f"{args.model_name}_telecom_{args.harness}_{args.mode}.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[hma_sft tau2] harness={args.harness} "
        f"user-sim={args.user_model} api={args.user_api_base} "
        f"mode={args.mode} n={len(task_ids) if task_ids else task_limit} → {out}",
        flush=True,
    )
    run_eval(
        model_name=args.model_name,
        model_endpoint=args.model_endpoint,
        domain="telecom",
        harness=args.harness,
        k=args.k,
        user_model=args.user_model,
        out_path=str(out),
        task_limit=task_limit,
        served_model_name=args.served_model_name,
        task_ids=task_ids,
        seed=42,
        user_api_base=args.user_api_base,
        user_api_key=user_key,
        num_workers=workers,
        resume=True,
        max_retries=int(os.environ.get("EVAL_MAX_RETRIES", "3")),
        agent_backend=os.environ.get("AGENT_BACKEND", "chat_tools"),
    )


if __name__ == "__main__":
    main()
