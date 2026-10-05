"""Compose enabled operators into one MutationSpec."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from hma_sft.core.operators import OP_BY_NAME, OP_ORDER
from hma_sft.core.spec import MutationSpec
from hma_sft.core.types import Catalog


def compose(
    catalog: Catalog,
    ctx: Dict[str, Any],
    enabled: Iterable[str],
    *,
    env_id: str = "",
    mutation_id: str = "",
) -> MutationSpec:
    want = {str(x) for x in enabled}
    spec = MutationSpec(env_id=env_id, mutation_id=mutation_id, strategy=env_id)
    bits: List[str] = []
    for op in OP_ORDER:
        if op.name not in want:
            continue
        plan = op.plan(catalog, ctx)
        op.merge_into(spec, plan)
        bits.append(op.name)
    spec.description = "+".join(bits)
    spec.assert_official_hidden()
    return spec
