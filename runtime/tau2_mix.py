"""τ² mix toolkit. Bodies come from OpenAI-generated code."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from tau2.domains.telecom.tools import TelecomTools
from tau2_hma.telecom_impl.config import OFFICIAL_TOOL_NAMES

GENERATED = Path(__file__).resolve().parent / "generated" / "tau2_impl.py"


def _load_generated_class():
    if not GENERATED.is_file():
        raise FileNotFoundError(
            f"missing {GENERATED}. Generate with your own key:\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "  PYTHONPATH=. python3 hma_sft/scripts/generate_mix.py --bench tau2"
        )
    spec = importlib.util.spec_from_file_location("hma_sft_tau2_impl", GENERATED)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {GENERATED}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, "GeneratedTau2Mix", None)
    if cls is None:
        raise ImportError(f"{GENERATED} has no GeneratedTau2Mix")
    return cls


class TelecomToolsHmaSft(TelecomTools):
    """Official TelecomTools + LLM-generated mix methods. Official 13 names hidden."""

    _generated_cls = None
    _generated_names: set[str] = set()

    def __init__(self, db: Any, cfg: Optional[Any] = None) -> None:
        type(self)._attach_generated()
        super().__init__(db)
        self._hidden = set(OFFICIAL_TOOL_NAMES)

    @classmethod
    def _attach_generated(cls) -> None:
        if cls._generated_cls is not None:
            return
        gen = _load_generated_class()
        cls._generated_cls = gen
        names = set()
        for name, val in vars(gen).items():
            if name.startswith("_") and name not in ("_customer_by_id_index",):
                if not callable(val):
                    continue
            if callable(val) and (getattr(val, "__tool__", False) or name.startswith("_")):
                setattr(cls, name, val)
                if getattr(val, "__tool__", False):
                    names.add(name)
        cls._generated_names = names

    @property
    def tools(self) -> Dict[str, Callable]:
        names = set(self._func_tools.keys()) | set(self._generated_names)
        out = {}
        for name in names:
            if name in self._hidden:
                continue
            if hasattr(self, name):
                out[name] = getattr(self, name)
        return out


def install_hma_sft_mix(env: Any) -> Any:
    db = getattr(env.tools, "db", None)
    if db is None:
        db = getattr(env, "db", None)
    env.tools = TelecomToolsHmaSft(db)
    return env


def ensure_generated_or_hint() -> Path:
    if not GENERATED.is_file():
        raise FileNotFoundError(
            "τ² mix impl is LLM-generated. Set OPENAI_API_KEY and run "
            "hma_sft/scripts/generate_mix.py --bench tau2"
        )
    return GENERATED
