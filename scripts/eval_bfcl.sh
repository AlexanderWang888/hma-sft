#!/usr/bin/env bash
# BFCL live-scene 123 × H_mix_full.
# MODE=smoke|live_scene
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
MODE="${MODE:-live_scene}"
exec python3 "$ROOT/hma_sft/scripts/eval_bfcl.py" --mode "$MODE" "$@"
