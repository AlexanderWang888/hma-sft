"""Official OpenAI Chat Completions. Key comes from the environment only."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

OFFICIAL_BASE = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1"


class MissingOpenAIKey(RuntimeError):
    pass


def resolve_api_key() -> str:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key or key in {"EMPTY", "your-key-here", "sk-xxx"}:
        raise MissingOpenAIKey(
            "Set OPENAI_API_KEY to your own official OpenAI key. "
            "hma_sft does not ship a key."
        )
    return key


def resolve_base_url() -> str:
    raw = (os.environ.get("OPENAI_BASE_URL") or OFFICIAL_BASE).strip().rstrip("/")
    if raw.endswith("/chat/completions"):
        raw = raw[: -len("/chat/completions")]
    return raw or OFFICIAL_BASE


def resolve_model() -> str:
    return (os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def chat_completions(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    response_format: dict[str, str] | None = None,
) -> str:
    """POST /v1/chat/completions on the official OpenAI API."""
    key = resolve_api_key()
    url = resolve_base_url() + "/chat/completions"
    body: dict[str, Any] = {
        "model": resolve_model(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        body["response_format"] = response_format
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"OpenAI HTTP {e.code}: {detail}") from e
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    if not content:
        raise RuntimeError(f"empty OpenAI completion: {data!r}"[:400])
    return str(content)


def chat_json(messages: list[dict[str, str]], **kwargs: Any) -> Any:
    raw = chat_completions(
        messages,
        response_format={"type": "json_object"},
        **kwargs,
    )
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return json.loads(text)
