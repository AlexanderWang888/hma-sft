#!/usr/bin/env bash
# MetaTool H_mix CSR (4 scenes).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$ROOT/hma_sft/scripts/eval_metatool.py" "$@"
