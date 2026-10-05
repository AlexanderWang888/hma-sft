#!/usr/bin/env bash
# τ² telecom mix. HARNESS=H_mix_impl|original. MODE=smoke|small|full
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/tau2_hma/vendor/tau2-bench-offline/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-$ROOT/tau2_hma/vendor/tau2-bench-offline/data}"
if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "EMPTY" ]]; then
  echo "[hma_sft] set OPENAI_API_KEY to your official OpenAI key" >&2
  exit 1
fi
HARNESS="${HARNESS:-H_mix_impl}"
case "$HARNESS" in
  H_mix_impl|original) ;;
  *)
    echo "[hma_sft] unsupported harness: $HARNESS (use H_mix_impl)" >&2
    exit 1
    ;;
esac
MODE="${MODE:-full}"
exec python3 "$ROOT/hma_sft/scripts/eval_tau2.py" --harness "$HARNESS" --mode "$MODE" "$@"
