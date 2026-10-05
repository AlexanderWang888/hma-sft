"""Seven mix operators. Dataset lexicons fill plans."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from hma_sft.core.spec import MutationSpec
from hma_sft.core.types import Catalog, Tool


@dataclass
class OperatorPlan:
    name: str
    tool_name_map: Dict[str, str] = field(default_factory=dict)
    arg_name_map: Dict[str, Dict[str, str]] = field(default_factory=dict)
    dropped_tools: List[str] = field(default_factory=list)
    added_tools: List[Dict[str, Any]] = field(default_factory=list)
    widen_arg_map: Dict[str, List[str]] = field(default_factory=dict)
    widen_desc_map: Dict[str, Dict[str, str]] = field(default_factory=dict)
    optional_props: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    near_tools: List[Dict[str, Any]] = field(default_factory=list)
    far_tools: List[Dict[str, Any]] = field(default_factory=list)
    desc_map: Dict[str, str] = field(default_factory=dict)
    suite_map: Dict[str, str] = field(default_factory=dict)


class Operator(ABC):
    name: str = ""

    @abstractmethod
    def plan(self, catalog: Catalog, ctx: Dict[str, Any]) -> OperatorPlan:
        raise NotImplementedError

    def merge_into(self, spec: MutationSpec, plan: OperatorPlan) -> None:
        spec.tool_name_map.update(plan.tool_name_map)
        for k, v in plan.arg_name_map.items():
            spec.arg_name_map.setdefault(k, {}).update(v)
        spec.dropped_tools.extend(plan.dropped_tools)
        spec.added_tools.extend(plan.added_tools)
        for k, v in plan.widen_arg_map.items():
            spec.widen_arg_map.setdefault(k, [])
            spec.widen_arg_map[k].extend(x for x in v if x not in spec.widen_arg_map[k])
        for k, v in plan.widen_desc_map.items():
            spec.widen_desc_map.setdefault(k, {}).update(v)
        for k, v in plan.optional_props.items():
            spec.optional_props.setdefault(k, {}).update(v)
        spec.near_tools.extend(plan.near_tools)
        spec.far_tools.extend(plan.far_tools)
        spec.desc_map.update(plan.desc_map)
        spec.suite_map.update(plan.suite_map)


class ToolRename(Operator):
    name = "rename"

    def plan(self, catalog: Catalog, ctx: Dict[str, Any]) -> OperatorPlan:
        mapping: Dict[str, str] = dict(ctx.get("rename_map") or {})
        dropped = set(ctx.get("dropped") or [])
        mapping = {k: v for k, v in mapping.items() if k not in dropped}
        desc_map = {
            k: v
            for k, v in (ctx.get("desc_map") or {}).items()
            if k in mapping
        }
        return OperatorPlan(name=self.name, tool_name_map=mapping, desc_map=desc_map)


class ArgRename(Operator):
    name = "arg_rename"

    def plan(self, catalog: Catalog, ctx: Dict[str, Any]) -> OperatorPlan:
        return OperatorPlan(name=self.name, arg_name_map=dict(ctx.get("arg_rename_map") or {}))


class SchemaReshape(Operator):
    name = "reshape"

    def plan(self, catalog: Catalog, ctx: Dict[str, Any]) -> OperatorPlan:
        plan = OperatorPlan(
            name=self.name,
            widen_arg_map={k: list(v) for k, v in (ctx.get("widen_arg_map") or {}).items()},
            widen_desc_map={k: dict(v) for k, v in (ctx.get("widen_desc_map") or {}).items()},
            optional_props={k: dict(v) for k, v in (ctx.get("optional_props") or {}).items()},
        )
        missing = []
        for tname, args in plan.widen_arg_map.items():
            descs = plan.widen_desc_map.get(tname) or {}
            missing.extend(a for a in args if not str(descs.get(a) or "").strip())
        if missing:
            raise ValueError(
                "reshape widen descriptions must be LLM-generated (missing for: "
                f"{missing}); run scripts/generate_mix.py to (re)generate the lexicon"
            )
        return plan


class CoverReplace(Operator):
    name = "cover"

    def plan(self, catalog: Catalog, ctx: Dict[str, Any]) -> OperatorPlan:
        covers: Dict[str, Dict[str, Any]] = dict(ctx.get("covers") or {})
        dropped = list(covers)
        added = [c["tool"] for c in covers.values() if c.get("tool")]
        name_map = {k: str(c["surface"]) for k, c in covers.items() if c.get("surface")}
        return OperatorPlan(
            name=self.name,
            dropped_tools=dropped,
            added_tools=added,
            tool_name_map=name_map,
        )


class SuiteFold(Operator):
    name = "fold"

    def plan(self, catalog: Catalog, ctx: Dict[str, Any]) -> OperatorPlan:
        folds: List[Dict[str, Any]] = list(ctx.get("folds") or [])
        dropped: List[str] = []
        added: List[Dict[str, Any]] = []
        suite_map: Dict[str, str] = {}
        for fold in folds:
            members = [str(x) for x in (fold.get("members") or [])]
            surface = str(fold.get("surface") or "")
            tool = fold.get("tool")
            dropped.extend(members)
            if tool:
                added.append(tool)
            for m in members:
                suite_map[m] = surface
        return OperatorPlan(
            name=self.name,
            dropped_tools=dropped,
            added_tools=added,
            suite_map=suite_map,
        )


class NearDistractor(Operator):
    name = "near"

    def plan(self, catalog: Catalog, ctx: Dict[str, Any]) -> OperatorPlan:
        return OperatorPlan(name=self.name, near_tools=list(ctx.get("near_tools") or []))


class FarDistractor(Operator):
    name = "far"

    def plan(self, catalog: Catalog, ctx: Dict[str, Any]) -> OperatorPlan:
        return OperatorPlan(name=self.name, far_tools=list(ctx.get("far_tools") or []))


OP_ORDER = (
    CoverReplace(),
    SuiteFold(),
    ToolRename(),
    ArgRename(),
    SchemaReshape(),
    NearDistractor(),
    FarDistractor(),
)

OP_BY_NAME = {op.name: op for op in OP_ORDER}


def tools_to_openai(catalog: Catalog) -> List[Dict[str, Any]]:
    return [t.to_openai() if isinstance(t, Tool) else t for t in catalog]  # type: ignore[arg-type]
