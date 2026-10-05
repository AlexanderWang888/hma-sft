#!/usr/bin/env python3
"""MetaTool occupation CSR on H_mix eval (4 scenes).

Template: qwen3_6, enable_thinking=false.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hma_sft.adapters.metatool import HMA_SCENES
from hma_sft.eval.parse import extract_tool

DEFAULT_DATA = ROOT / "hma_sft" / "data" / "eval" / "metatool_H_mix.json"


def _chat_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def chat_api(api_base: str, user: str, model: str, max_tokens: int) -> str:
    url = _chat_url(api_base)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "extra_body": {
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    if not content and msg.get("reasoning_content"):
        content = str(msg.get("reasoning_content"))
    return content


def _csr(ok: int, n: int) -> str:
    if not n:
        return "n/a"
    return f"{ok}/{n}={ok / n * 100:.1f}%"


def summarize(rows: list[dict]) -> dict:
    by_sc: dict[str, list] = defaultdict(list)
    for r in rows:
        by_sc[r["scenario"]].append(r)
    scenes = {}
    for sc in HMA_SCENES:
        xs = by_sc.get(sc, [])
        ok = sum(1 for x in xs if x["hit"])
        scenes[sc] = {
            "n": len(xs),
            "hit": ok,
            "csr": (ok / len(xs) if xs else None),
            "status": dict(Counter(x["status"] for x in xs)),
        }
    n = len(rows)
    hit = sum(1 for r in rows if r["hit"])
    return {
        "n": n,
        "hit": hit,
        "csr": hit / n if n else None,
        "scenes": scenes,
        "status": dict(Counter(r["status"] for r in rows)),
    }


def _print_summary(rep: dict) -> None:
    print()
    print("=== MetaTool H_mix CSR ===")
    for sc in HMA_SCENES:
        s = rep["scenes"].get(sc, {})
        if not s.get("n"):
            continue
        print(f"  {sc:22s}  {_csr(s.get('hit', 0), s.get('n', 0))}")
    print(f"  {'ALL':22s}  {_csr(rep['hit'], rep['n'])}")
    print(f"  status {rep['status']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument(
        "--api-base",
        default="http://127.0.0.1:8001,http://127.0.0.1:8002",
    )
    ap.add_argument("--model", default="Qwen3.5-4B")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scenario", default="")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "hma_sft" / "outputs" / "metatool_H_mix",
    )
    args = ap.parse_args()

    if not args.data.is_file():
        print(f"[eval] missing {args.data}; run hma_sft/scripts/build_eval.py", file=sys.stderr)
        sys.exit(1)

    items = json.loads(args.data.read_text(encoding="utf-8"))
    if args.scenario:
        items = [x for x in items if x["scenario"] == args.scenario]
    if args.limit:
        items = items[: args.limit]
    bases = [b.strip() for b in args.api_base.split(",") if b.strip()]
    if not bases:
        print("[eval] empty --api-base", file=sys.stderr)
        sys.exit(1)
    rr = itertools.cycle(bases)
    rr_lock = threading.Lock()

    def next_base() -> str:
        with rr_lock:
            return next(rr)

    print(
        f"[eval] n={len(items)} env=H_mix model={args.model} "
        f"backends={bases} workers={args.workers} thinking=off template=qwen3_6",
        flush=True,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.out_dir / "predictions.jsonl"
    done: dict[str, dict] = {}
    if pred_path.is_file():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("id") and r.get("raw") is not None:
                done[r["id"]] = r
        print(f"[eval] resume {len(done)} from {pred_path}", flush=True)

    todo = [x for x in items if x["id"] not in done]
    t0 = time.time()

    def run_one(x: dict) -> dict:
        last_err = ""
        raw = ""
        for attempt in range(3):
            try:
                raw = chat_api(next_base(), x["action_prompt"], args.model, args.max_tokens)
                last_err = ""
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_err = str(e)
                time.sleep(1.5 * (attempt + 1))
        pred, status = extract_tool(raw, x["candidates"])
        gold = x["gold"]
        return {
            "id": x["id"],
            "scenario": x["scenario"],
            "query": x["query"],
            "gold": gold,
            "gold_canonical": x.get("gold_canonical"),
            "pred": pred,
            "status": status if not last_err else "error",
            "hit": bool(pred) and pred == gold,
            "raw": raw,
            "error": last_err,
        }

    with pred_path.open("a", encoding="utf-8") as fout:
        if todo:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
                futs = [ex.submit(run_one, x) for x in todo]
                for i, fut in enumerate(as_completed(futs), 1):
                    row = fut.result()
                    done[row["id"]] = row
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()
                    if i % 20 == 0 or i == len(todo):
                        print(f"[eval] {i}/{len(todo)} elapsed={time.time() - t0:.0f}s", flush=True)

    rows = [done[x["id"]] for x in items if x["id"] in done]
    rep = summarize(rows)
    rep.update(
        {
            "model": args.model,
            "api_base": args.api_base,
            "protocol": "hma_sft_metatool_H_mix",
            "env": "H_mix",
            "template": "qwen3_6",
            "enable_thinking": False,
        }
    )
    _print_summary(rep)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] wrote {pred_path}")
    print(f"[eval] wrote {summary_path}")


if __name__ == "__main__":
    main()
