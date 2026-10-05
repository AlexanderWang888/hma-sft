# lexicons/

Eval-only output of `generate_mix.py`. Empty in git. Do not hand-write mix names.

```bash
export OPENAI_API_KEY=sk-...
PYTHONPATH=. python3 hma_sft/scripts/generate_mix.py --bench all
PYTHONPATH=. python3 hma_sft/scripts/build_eval.py
```
