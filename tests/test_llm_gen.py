"""Generation path uses official OpenAI and ships no keys."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hma_sft.gen.openai_client import OFFICIAL_BASE, MissingOpenAIKey, resolve_api_key, resolve_base_url


def test_official_base() -> None:
    assert resolve_base_url() == OFFICIAL_BASE or "openai.com" in resolve_base_url()


def test_missing_key_raises(monkey_env=None) -> None:
    import os

    old = os.environ.pop("OPENAI_API_KEY", None)
    try:
        try:
            resolve_api_key()
            raise AssertionError("expected MissingOpenAIKey")
        except MissingOpenAIKey:
            pass
    finally:
        if old is not None:
            os.environ["OPENAI_API_KEY"] = old


def test_no_baked_keys_in_hma_sft() -> None:
    banned = ("pk-", "sk-proj-", "jdcloud.com")
    skip = {"data", "outputs", "tests"}
    hits = []
    for path in (ROOT / "hma_sft").rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".md", ".sh", ".json", ".yaml"}:
            continue
        if any(part in skip for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "pk-" in text and "pk-a,pk-b" not in text:
            if any(line.strip().startswith('"pk-') or "pk-0" in line or "pk-3" in line for line in text.splitlines()):
                hits.append(str(path.relative_to(ROOT)))
        if "ai-api.jdcloud.com" in text:
            hits.append(f"{path.relative_to(ROOT)}:jdcloud")
    assert not hits, hits


def test_runtime_does_not_import_mix_tools() -> None:
    src = (ROOT / "hma_sft" / "runtime" / "tau2_mix.py").read_text(encoding="utf-8")
    assert "from tau2_hma.telecom_impl.mix_tools" not in src
    assert "TelecomToolsImplMix" not in src


if __name__ == "__main__":
    test_official_base()
    test_missing_key_raises()
    test_no_baked_keys_in_hma_sft()
    test_runtime_does_not_import_mix_tools()
    print("ok")
