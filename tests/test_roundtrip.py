"""BFCL-style apply then back_map restores canonical calls."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hma_sft.core.apply import apply_to_schema, apply_to_tool_call, back_map_call
from hma_sft.core.spec import MutationSpec


def test_rename_arg_roundtrip() -> None:
    spec = MutationSpec(
        tool_name_map={"get_foo": "pull_foo"},
        arg_name_map={"get_foo": {"customer_id": "party_ref"}},
    )
    surface = apply_to_tool_call(
        {"name": "get_foo", "arguments": {"customer_id": "C1"}}, spec
    )
    assert surface["name"] == "pull_foo"
    assert surface["arguments"]["party_ref"] == "C1"
    back = back_map_call(surface, spec)
    assert back["name"] == "get_foo"
    assert back["arguments"]["customer_id"] == "C1"


def test_widen_unwraps_singleton_list() -> None:
    spec = MutationSpec(widen_arg_map={"refuel_data": ["line_id"]})
    back = back_map_call(
        {"name": "refuel_data", "arguments": {"line_id": ["L1"]}}, spec
    )
    assert back["arguments"]["line_id"] == "L1"


def test_assert_official_hidden_catches_official_near() -> None:
    spec = MutationSpec(
        tool_name_map={"get_customer_by_phone": "fetch_party_line"},
        near_tools=[{"name": "get_customer_by_phone", "description": "x"}],
    )
    try:
        spec.assert_official_hidden({"get_customer_by_phone"})
    except AssertionError:
        return
    raise AssertionError("official rename source reused as near must fail")


def test_assert_official_hidden_allows_arg_only_official() -> None:
    spec = MutationSpec(arg_name_map={"get_foo": {"a": "b"}})
    spec.assert_official_hidden({"get_foo"})


def test_distractor_rejected() -> None:
    spec = MutationSpec(
        near_tools=[{"name": "lookup_sim_profile", "description": "x"}],
    )
    back = back_map_call({"name": "lookup_sim_profile", "arguments": {}}, spec)
    assert str(back["name"]).startswith("_hma_reject_")


def test_widen_requires_llm_desc() -> None:
    tools = [
        {
            "name": "refuel_data",
            "parameters": {
                "type": "object",
                "properties": {"line_id": {"type": "string", "description": "id"}},
                "required": ["line_id"],
            },
        }
    ]
    spec = MutationSpec(widen_arg_map={"refuel_data": ["line_id"]})
    try:
        apply_to_schema(tools, spec)
    except ValueError:
        pass
    else:
        raise AssertionError("widen without LLM description must fail")
    spec2 = MutationSpec(
        widen_arg_map={"refuel_data": ["line_id"]},
        widen_desc_map={"refuel_data": {"line_id": "Line identifier; one value or a batch."}},
    )
    out = apply_to_schema(tools, spec2)
    got = out[0]["parameters"]["properties"]["line_id"]["description"]
    assert got == "Line identifier; one value or a batch."


if __name__ == "__main__":
    test_rename_arg_roundtrip()
    test_widen_unwraps_singleton_list()
    test_assert_official_hidden_catches_official_near()
    test_assert_official_hidden_allows_arg_only_official()
    test_distractor_rejected()
    test_widen_requires_llm_desc()
    print("ok")
