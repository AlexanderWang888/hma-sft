"""Canonical intermediate types shared by all three benches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class Tool:
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_openai(self) -> Dict[str, Any]:
        if self.raw.get("type") == "function" or isinstance(self.raw.get("function"), dict):
            out = dict(self.raw)
            fn = dict(out.get("function") or {})
            fn["name"] = self.name
            fn["description"] = self.description
            if self.parameters:
                fn["parameters"] = self.parameters
            out["function"] = fn
            out["type"] = "function"
            return out
        if self.raw.get("name") or self.parameters:
            return {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
                or self.raw.get("parameters")
                or {"type": "object", "properties": {}},
            }
        return {"name": self.name, "description": self.description}


@dataclass
class Call:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


Catalog = List[Tool]


@dataclass
class Trajectory:
    id: str
    query: str = ""
    catalog: List[Tool] = field(default_factory=list)
    calls: List[Call] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


Split = Literal["train", "eval"]
