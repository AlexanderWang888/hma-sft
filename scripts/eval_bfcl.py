#!/usr/bin/env python3
"""BFCL live-scene 123 × H_mix_full."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BFCL = ROOT / "toolmind_bfcl"
for p in (ROOT, BFCL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

LIVE_SUBSET = ["live_multiple"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "live_scene"], default="live_scene")
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "Qwen3.5-4B"))
    ap.add_argument("--api-url", default=os.environ.get("VLLM_BASE", "http://127.0.0.1:8001/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument(
        "--work-dir",
        default=str(ROOT / "hma_sft" / "outputs" / "bfcl_v4"),
    )
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("EVAL_BATCH", "1")))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--is-fc-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--underscore-to-dot",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = ap.parse_args()

    os.environ["BFCL_HOLDOUT_PROTOCOL"] = "bfcl_live_scene_holdout_v1"
    from hma_sft.eval.bfcl_hook import install_bfcl_hooks
    from perturb.live_holdout import install_holdout_filter, load_test_ids

    work_dir = Path(args.work_dir) / f"{args.model}_{args.mode}_H_mix_full"
    work_dir.mkdir(parents=True, exist_ok=True)
    install_bfcl_hooks(split="eval")
    install_holdout_filter(load_test_ids())

    from evalscope import TaskConfig, run_task

    limit = 5 if args.mode == "smoke" and args.limit is None else args.limit
    task_cfg = TaskConfig(
        model=args.model,
        api_url=args.api_url,
        api_key=args.api_key,
        eval_type="openai_api",
        datasets=["bfcl_v4"],
        eval_batch_size=args.batch_size,
        dataset_args={
            "bfcl_v4": {
                "subset_list": LIVE_SUBSET,
                "extra_params": {
                    "underscore_to_dot": args.underscore_to_dot,
                    "is_fc_model": args.is_fc_model,
                    "SERPAPI_API_KEY": os.environ.get("SERPAPI_API_KEY"),
                },
            }
        },
        generation_config={"temperature": 0},
        work_dir=str(work_dir),
        no_timestamp=True,
    )
    if limit is not None:
        task_cfg.limit = limit
    print(
        f"[hma_sft bfcl] env=H_mix_full mode={args.mode} n_test=123 "
        f"model={args.model} api={args.api_url} work_dir={work_dir}",
        flush=True,
    )
    run_task(task_cfg=task_cfg)
    reviews = work_dir / "reviews"
    if reviews.exists():
        from perturb.live_holdout import print_holdout_summary

        print_holdout_summary(reviews)


if __name__ == "__main__":
    main()
