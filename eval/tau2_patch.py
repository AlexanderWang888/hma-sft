"""Wire H_mix_impl into tau2_hma eval without editing tau2_hma files."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, List

from hma_sft.core.lexicon import load_split


def install_hma_sft_mix_eval(env: Any, canonical_tools: List[dict]) -> tuple[Any, Any]:
    from hma_sft.runtime.tau2_mix import TelecomToolsHmaSft
    from tau2_hma.telecom_impl.apply import (
        ImplMixEnvironment,
        ImplMixInterceptor,
        surface_schemas_from_toolkit,
    )
    from tau2_hma.telecom_impl.config import ImplMixConfig

    lex = load_split("tau2", "impl", "eval")
    cfg = ImplMixConfig(
        kind="H_mix_impl",
        include_distract=True,
        cover=deepcopy(lex.get("cover") or {}),
        suite=deepcopy(lex.get("suite") or {}),
        rename=dict(lex.get("rename") or {}),
        arg_rename={k: dict(v) for k, v in (lex.get("arg_rename") or {}).items()},
        reshape=deepcopy(lex.get("reshape") or {}),
        near=deepcopy(lex.get("near") or []),
        far=deepcopy(lex.get("far") or []),
    )
    for field in ImplMixConfig.__dataclass_fields__:
        if field.startswith("include_") and field != "include_distract":
            setattr(cfg, field, False)
    mix = TelecomToolsHmaSft(env.tools.db)
    env.tools = mix
    schemas = surface_schemas_from_toolkit(mix)
    names = []
    for s in schemas:
        fn = s.get("function") if isinstance(s.get("function"), dict) else s
        names.append(str(fn.get("name") or ""))
    official = set(lex.get("official_tools") or [])
    leak = official & set(names)
    if leak:
        raise RuntimeError(f"τ² mix catalog still has official names: {sorted(leak)}")
    interceptor = ImplMixInterceptor(canonical_tools, cfg, schemas)
    interceptor.spec.strategy = "H_mix_impl"
    return ImplMixEnvironment(env, interceptor), interceptor


def patch_tau2_eval() -> None:
    from tau2_hma.core.schema import MutationSpec
    import tau2_hma.eval.harness_mutator as hm
    import tau2_hma.telecom_impl.apply as apply_mod
    import tau2_hma.telecom_impl.config as cfg_mod

    if "H_mix_impl" not in hm.HARNESSES:
        hm.HARNESSES.append("H_mix_impl")

    orig_build = hm.build_harness_spec

    def build_harness_spec(domain: str, harness: str):
        if harness == "H_mix_impl":
            if domain != "telecom":
                raise ValueError("H_mix_impl is telecom-only")
            return MutationSpec(strategy="H_mix_impl", description="hma_sft impl mix")
        if harness == "original":
            return orig_build(domain, harness)
        raise RuntimeError(f"unsupported harness {harness!r}")

    hm.build_harness_spec = build_harness_spec
    apply_mod.IMPL_MIX_HARNESSES = ("H_mix_impl",)
    apply_mod.is_impl_mix_harness = lambda harness: harness == "H_mix_impl"

    def install_impl_mix(env, harness, canonical_tools):
        if harness != "H_mix_impl":
            raise RuntimeError(f"unsupported harness {harness!r}")
        return install_hma_sft_mix_eval(env, canonical_tools)

    apply_mod.install_impl_mix = install_impl_mix
    cfg_mod.task_uses_clean_catalog = lambda *a, **k: False
    print("[hma_sft] τ² eval patched: H_mix_impl", flush=True)
