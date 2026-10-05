"""MetaTool CSR parse (name match only)."""

from __future__ import annotations

import re

_THINK = re.compile(r"<think>.*?</think>", re.S)
_THINK_OPEN = re.compile(r"<think>.*", re.S)
_NONE = re.compile(r"\b(none|no tool|n/a|not applicable|no suitable tool)\b", re.I)


def strip_think(text: str) -> str:
    text = _THINK.sub("", text or "")
    text = _THINK_OPEN.sub("", text)
    return text.strip()


def extract_tool(text: str, candidates: list[str]) -> tuple[str | None, str]:
    raw = strip_think(text)
    if not raw:
        return None, "empty"
    ordered = sorted(candidates, key=len, reverse=True)
    hits: list[str] = []
    lower = raw.lower()
    for name in ordered:
        if name and (name in raw or name.lower() in lower) and name not in hits:
            hits.append(name)
    if len(hits) == 1:
        return hits[0], "ok"
    if len(hits) > 1:
        m = re.search(r"(?:tool\s*[:=]\s*)([A-Za-z0-9_&+-]+)", raw, re.I)
        if m:
            token = m.group(1).strip()
            for name in ordered:
                if token == name or token.lower() == name.lower():
                    return name, "ok"
        return None, "ambiguous"
    if _NONE.search(raw):
        return None, "none"
    return None, "empty"
