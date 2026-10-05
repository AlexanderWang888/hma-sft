"""Locked mix protocols per bench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# slot name → operator class name in operators.py
# These are the EVAL operator sets (paper Table 1, tab:protocol-map, Eval
# column; BFCL has no suite fold by design). MetaTool catalogs are
# name+description only (no parameter schemas), so arg_rename / reshape are
# declared to match the paper's full evaluation operator set but are inert
# for CSR scoring; the remaining five carry all the selection-relevant
# mutation.
# Training operator sets (paper Table 1, Train column) are enforced per split
# in the adapters — e.g. MetaTool training keeps every official tool in the
# cover family ({cover, fold} only; no tool rename / near / far), see
# adapters/metatool.py plan_context.
BFCL_OPS: Tuple[str, ...] = ("cover", "rename", "arg_rename", "reshape", "near", "far")
METATOOL_OPS: Tuple[str, ...] = ("cover", "rename", "arg_rename", "reshape", "fold", "near", "far")
TAU2_OPS: Tuple[str, ...] = ("cover", "fold", "rename", "arg_rename", "reshape", "near", "far")


@dataclass(frozen=True)
class Protocol:
    bench: str
    env_id: str
    ops: Tuple[str, ...]


PROTOCOLS: Dict[str, Protocol] = {
    "bfcl": Protocol(bench="bfcl", env_id="H_mix_full", ops=BFCL_OPS),
    "metatool": Protocol(bench="metatool", env_id="H_mix", ops=METATOOL_OPS),
    "tau2": Protocol(bench="tau2", env_id="H_mix_impl", ops=TAU2_OPS),
}


def get_protocol(bench: str) -> Protocol:
    if bench not in PROTOCOLS:
        raise KeyError(f"unknown bench {bench!r}; want {sorted(PROTOCOLS)}")
    return PROTOCOLS[bench]


def enabled_ops(bench: str) -> List[str]:
    return list(get_protocol(bench).ops)
