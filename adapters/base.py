"""Dataset adapter ABC. Each bench only fills IO + lexicon + scoring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal

from hma_sft.core.apply import apply_to_calls, apply_to_schema
from hma_sft.core.compose import compose
from hma_sft.core.protocol import get_protocol
from hma_sft.core.spec import MutationSpec, name_variants, tool_name_of
from hma_sft.core.types import Trajectory

Split = Literal["train", "eval"]


class DatasetAdapter(ABC):
    name: str

    def protocol(self):
        return get_protocol(self.name)

    @abstractmethod
    def load_train(self) -> List[Trajectory]:
        raise NotImplementedError

    @abstractmethod
    def load_eval(self) -> List[Trajectory]:
        raise NotImplementedError

    @abstractmethod
    def plan_context(self, traj: Trajectory, split: Split) -> Dict[str, Any]:
        """Lexicon-backed operator inputs for this trajectory."""
        raise NotImplementedError

    def mutate(self, traj: Trajectory, split: Split) -> tuple[MutationSpec, List[Dict[str, Any]], List]:
        proto = self.protocol()
        ctx = self.plan_context(traj, split)
        spec = compose(
            traj.catalog,
            ctx,
            proto.ops,
            env_id=proto.env_id,
            mutation_id=f"{traj.id}:{split}",
        )
        official = [t.to_openai() for t in traj.catalog]
        surface_catalog = apply_to_schema(official, spec)
        surface_calls = apply_to_calls(traj.calls, spec)
        official_gold = {t.name for t in traj.catalog}
        spec.assert_official_hidden(official_gold)
        hidden = spec.hidden_official_names()
        for raw in surface_catalog:
            n = tool_name_of(raw)
            if any(v in hidden for v in name_variants(n)):
                raise AssertionError(f"{traj.id} hidden official remains in catalog: {n}")
        return spec, surface_catalog, surface_calls
