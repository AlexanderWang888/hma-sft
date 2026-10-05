"""MutationSpec for the seven mix operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


def name_variants(name: str) -> List[str]:
    if not name:
        return []
    return list({name, name.replace(".", "_")})


def lookup_tool_keyed(d: Dict[str, Any], name: str) -> Any:
    if name in d:
        return d[name]
    want = name.replace(".", "_")
    for k, v in d.items():
        if str(k).replace(".", "_") == want:
            return v
    return None


@dataclass
class MutationSpec:
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
    mutation_id: str = ""
    env_id: str = ""
    strategy: str = ""
    description: str = ""
    skip: bool = False
    skip_reason: str = ""

    def dropped_name_set(self) -> set[str]:
        names: set[str] = set()
        for n in self.dropped_tools:
            names.update(name_variants(n))
        return names

    def surface_to_canonical_tool(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for canon, surface in {**self.tool_name_map, **self.suite_map}.items():
            for key in name_variants(surface):
                out[key] = canon
        return out

    def resolve_canonical_tool(self, name: str) -> str:
        s2c = self.surface_to_canonical_tool()
        if name in s2c:
            return s2c[name]
        want = name.replace(".", "_")
        for canon in list(self.tool_name_map) + list(self.arg_name_map):
            if canon == name or canon.replace(".", "_") == want:
                return canon
        return name

    def surface_to_canonical_args(self, tool_name: str) -> Dict[str, str]:
        amap = lookup_tool_keyed(self.arg_name_map, tool_name) or {}
        return {v: k for k, v in amap.items()}

    def distractor_name_set(self) -> set[str]:
        names: set[str] = set()
        for raw in self.near_tools + self.far_tools:
            n = tool_name_of(raw)
            names.update(name_variants(n))
        return names

    def is_empty(self) -> bool:
        return not (
            self.tool_name_map
            or self.arg_name_map
            or self.dropped_tools
            or self.added_tools
            or self.widen_arg_map
            or self.widen_desc_map
            or self.optional_props
            or self.near_tools
            or self.far_tools
            or self.desc_map
            or self.suite_map
        )

    def mix_surface_names(self) -> set[str]:
        names: set[str] = set()
        for raw in self.added_tools + self.near_tools + self.far_tools:
            names.update(name_variants(tool_name_of(raw)))
        for v in self.tool_name_map.values():
            names.update(name_variants(v))
        for v in self.suite_map.values():
            names.update(name_variants(v))
        return names

    def hidden_official_names(self) -> set[str]:
        """Official names that must not remain callable (cover/fold/rename sources)."""
        names = set(self.dropped_name_set())
        for k in self.tool_name_map:
            names.update(name_variants(k))
        for k in self.suite_map:
            names.update(name_variants(k))
        return names

    def assert_official_hidden(self, official_gold: set[str] | None = None) -> None:
        surfaces = self.mix_surface_names()
        hidden = self.hidden_official_names()
        leak = surfaces & hidden
        if leak:
            raise AssertionError(f"hidden official names leaked onto mix surfaces: {sorted(leak)}")
        if official_gold:
            gold = {n for g in official_gold for n in name_variants(g)}
            leak2 = surfaces & gold
            if leak2:
                raise AssertionError(f"official names used as mix surfaces: {sorted(leak2)}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name_map": self.tool_name_map,
            "arg_name_map": self.arg_name_map,
            "dropped_tools": self.dropped_tools,
            "added_tools": self.added_tools,
            "widen_arg_map": self.widen_arg_map,
            "widen_desc_map": self.widen_desc_map,
            "optional_props": self.optional_props,
            "near_tools": self.near_tools,
            "far_tools": self.far_tools,
            "desc_map": self.desc_map,
            "suite_map": self.suite_map,
            "mutation_id": self.mutation_id,
            "env_id": self.env_id,
            "strategy": self.strategy,
            "description": self.description,
            "skip": self.skip,
            "skip_reason": self.skip_reason,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MutationSpec":
        d = d or {}
        return cls(
            tool_name_map=dict(d.get("tool_name_map") or {}),
            arg_name_map={k: dict(v) for k, v in (d.get("arg_name_map") or {}).items()},
            dropped_tools=list(d.get("dropped_tools") or []),
            added_tools=list(d.get("added_tools") or []),
            widen_arg_map={k: list(v) for k, v in (d.get("widen_arg_map") or {}).items()},
            widen_desc_map={k: dict(v) for k, v in (d.get("widen_desc_map") or {}).items()},
            optional_props={k: dict(v) for k, v in (d.get("optional_props") or {}).items()},
            near_tools=list(d.get("near_tools") or []),
            far_tools=list(d.get("far_tools") or []),
            desc_map=dict(d.get("desc_map") or {}),
            suite_map=dict(d.get("suite_map") or {}),
            mutation_id=str(d.get("mutation_id") or ""),
            env_id=str(d.get("env_id") or ""),
            strategy=str(d.get("strategy") or ""),
            description=str(d.get("description") or ""),
            skip=bool(d.get("skip")),
            skip_reason=str(d.get("skip_reason") or ""),
        )


def tool_name_of(spec: Dict[str, Any]) -> str:
    if isinstance(spec.get("function"), dict):
        return str(spec["function"].get("name") or "")
    return str(spec.get("name") or "")


def function_block(spec: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(spec.get("function"), dict):
        return spec["function"]
    return spec
