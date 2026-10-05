Eval catalogs are built after LLM generation:

```bash
PYTHONPATH=. python3 hma_sft/scripts/generate_mix.py --bench all
PYTHONPATH=. python3 hma_sft/scripts/build_eval.py
```
