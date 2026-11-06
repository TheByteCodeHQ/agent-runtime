"""A tamper-evident audit trail. Each event carries the hash of the one before it, so editing, deleting, inserting or reordering
any line breaks the chain and ``verify`` says where.

This detects changes; it does not prevent them, and someone who can rewrite the whole file can rebuild a valid chain. For real
assurance, ship the final hash somewhere the agent's host cannot write (a ticket, a log service, a signed commit).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .guard import redact_value

GENESIS = "0" * 64


def _digest(prev: str, body: dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True)
    return hashlib.sha256((prev + canonical).encode("utf-8")).hexdigest()


class Trace:
    """Collects events in memory and, if a path is given, appends them to a JSONL file as they happen."""

    def __init__(self, path: Path | None = None, clock: Callable[[], float] = time.time) -> None:
        self.path, self.clock = path, clock
        self.events: list[dict[str, Any]] = []
        self.last_hash = GENESIS
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def add(self, kind: str, **data: Any) -> dict[str, Any]:
        body = {"seq": len(self.events), "ts": round(self.clock(), 3), "kind": kind, "data": redact_value(data)}
        event = {**body, "prev": self.last_hash, "hash": _digest(self.last_hash, body)}
        self.events.append(event)
        self.last_hash = event["hash"]
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, default=str) + "\n")
        return event


def read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            raise ValueError(f"{path.name} line {number}: not valid JSON") from None
    return events


def verify(events: list[dict[str, Any]]) -> list[str]:
    """Problems found in the chain; an empty list means every link holds."""
    problems = []
    prev = GENESIS
    for index, event in enumerate(events):
        try:
            body = {"seq": event["seq"], "ts": event["ts"], "kind": event["kind"], "data": event["data"]}
            if event["seq"] != index:
                problems.append(f"event {index}: sequence number is {event['seq']} (an event was removed, added or moved)")
            if event["prev"] != prev:
                problems.append(f"event {index}: does not follow the previous event")
            if _digest(event["prev"], body) != event["hash"]:
                problems.append(f"event {index}: contents do not match their hash (edited)")
            prev = event["hash"]
        except (KeyError, TypeError):
            problems.append(f"event {index}: malformed")
            prev = GENESIS
    return problems
