# Seeds

Official catalogs only. Mix surfaces and τ² Python bodies are **not** stored here.

```bash
export OPENAI_API_KEY=sk-...          # your key
export OPENAI_MODEL=gpt-4.1           # optional
PYTHONPATH=. python3 hma_sft/scripts/generate_mix.py --bench all
PYTHONPATH=. python3 hma_sft/scripts/build_eval.py
```
