"""Append-only gate evidence and fail-closed prerequisite checks."""

import json
import os
from pathlib import Path

from oar.records import GateRecord, from_payload, to_json


class GateBlocked(RuntimeError):
    """Required evidence is absent, stopped, inconclusive or invalid."""


# Later owners may add explicit, evidenced branch permissions. A stopped gate
# never implicitly grants a different experimental branch. Synthesis is always OK.
_PREREQUISITES = {
    "WP00": (),
    "WP01": ("WP00",),
    "WP02": ("WP00", "WP01"),
    "WP03": ("WP00", "WP01", "WP02"),
    "WP04": ("WP00", "WP01", "WP02", "WP03"),
    "WP05": ("WP04",),
    "WP06": ("WP03", "WP04", "WP05"),
    "WP07": ("WP06",),
    "WP08": ("WP07",),
    "WP09": (),
}


def record_gate(record: GateRecord, out: Path) -> None:
    """Append one complete JSON line and flush it to durable local storage."""
    if not isinstance(record, GateRecord):
        raise TypeError("expected GateRecord")
    line = to_json(record) + "\n"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def require_gate(package: str, ledger: Path) -> None:
    """Require the latest explicit pass for each prerequisite, never infer one."""
    if package not in _PREREQUISITES:
        raise GateBlocked(f"unknown package: {package}")
    required = set(_PREREQUISITES[package])
    pending = list(required)
    while pending:
        for prerequisite in _PREREQUISITES[pending.pop()]:
            if prerequisite not in required:
                required.add(prerequisite)
                pending.append(prerequisite)
    if not required:
        return
    latest = {}
    try:
        with Path(ledger).open(encoding="utf-8") as stream:
            for line in stream:
                record = from_payload(json.loads(line))
                if not isinstance(record, GateRecord):
                    raise ValueError("ledger contains a non-gate record")
                latest[record.package] = record
    except (OSError, ValueError, TypeError) as exc:
        raise GateBlocked(f"invalid or unavailable prerequisite ledger for {package}: {exc}") from exc
    for prerequisite in sorted(required):
        if prerequisite not in latest or latest[prerequisite].status != "pass":
            raise GateBlocked(f"{package} requires passed {prerequisite}")
