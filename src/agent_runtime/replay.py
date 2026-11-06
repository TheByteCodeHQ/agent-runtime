"""Replay a recorded run against a different policy, with no model and no real tools, and see which decisions would change.

The model's recorded decisions are fed back in order and each tool returns what it returned the first time. So the only thing that
differs between the original run and the replay is the policy. That makes a safety rule testable: tighten it, replay the incident,
and read the list of verdicts that flipped.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from .agent import run_agent
from .budget import Budget
from .policy import Policy
from .tools import Tool, ToolRegistry
from .trace import Trace
from .types import Decision, Message, RunReport, ToolCall

NO_RECORD = "[replay: no recorded result for this call]"


def _key(tool: str, args: Any) -> str:
    return f"{tool}:{json.dumps(args, sort_keys=True, default=str)}"


class ReplayModel:
    name = "replay"

    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self._queue = deque(decisions)

    def decide(self, history: list[Message], tools: list[dict[str, Any]]) -> Decision:
        if not self._queue:
            return Decision(final="[replay: the recording ends here]")
        data = self._queue.popleft()
        if "final" in data:
            return Decision(final=data["final"], usage=data.get("usage", {}))
        return Decision(call=ToolCall(data["tool"], data.get("args", {})), usage=data.get("usage", {}))


@dataclass(frozen=True)
class Change:
    index: int
    tool: str
    args: dict[str, Any]
    was: str
    now: str
    reason: str

    def __str__(self) -> str:
        return f"#{self.index} {self.tool}({json.dumps(self.args, default=str)[:80]}): {self.was} -> {self.now}  ({self.reason})"


@dataclass
class ReplayResult:
    changes: list[Change]
    original: RunReport
    replayed: RunReport
    compared: int


def _verdicts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e["data"] for e in events if e["kind"] == "verdict"]


def replay(events: list[dict[str, Any]], policy: Policy, *, budget: Budget | None = None, approve: bool | None = None) -> ReplayResult:
    """``approve`` decides every ask in the replay; None uses what the human originally answered (and refuses if it never asked)."""
    start = next((e["data"] for e in events if e["kind"] == "run_start"), None)
    if start is None:
        raise ValueError("The trace has no run_start event, so it cannot be replayed")
    decisions = [e["data"] for e in events if e["kind"] == "decision"]
    results: dict[str, deque[str]] = defaultdict(deque)
    answers: dict[str, deque[bool]] = defaultdict(deque)
    last_call: dict[str, Any] = {}
    for e in events:
        data = e["data"]
        if e["kind"] == "verdict":
            last_call = data
        elif e["kind"] == "tool_result":
            results[_key(last_call.get("tool"), last_call.get("args"))].append(data["output"])
        elif e["kind"] == "approval":
            answers[_key(data["tool"], data["args"])].append(bool(data["approved"]))

    registry = ToolRegistry()
    for meta in start["tools"]:
        def make(name: str):
            def func(**args: Any) -> str:
                queue = results[_key(name, args)]
                return queue.popleft() if queue else NO_RECORD
            return func
        registry.register(Tool(meta["name"], "replayed", make(meta["name"]), {"type": "object"}, meta["risk"],
                               meta.get("untrusted_output", False), meta.get("egress", False)))

    def approver(call: ToolCall, reason: str) -> bool:
        if approve is not None:
            return approve
        queue = answers[_key(call.tool, call.args)]
        return queue.popleft() if queue else False

    original_end = next((e["data"] for e in reversed(events) if e["kind"] == "run_end"), {})
    original = RunReport(original_end.get("status", "unknown"), original_end.get("answer"), original_end.get("reason", ""),
                         original_end.get("steps", 0), original_end.get("tool_calls", 0), original_end.get("denied", 0),
                         original_end.get("approved", 0), original_end.get("injections", 0))
    replay_trace = Trace()
    replayed = run_agent(start["task"], ReplayModel(decisions), registry, policy=policy, budget=budget or Budget(**start["budget"]),
                         approver=approver, trace=replay_trace)

    old, new = _verdicts(events), _verdicts(replay_trace.events)
    changes = [Change(i, o["tool"], o.get("args", {}), o["action"], n["action"], n.get("reason", ""))
               for i, (o, n) in enumerate(zip(old, new, strict=False)) if o["action"] != n["action"]]
    return ReplayResult(changes, original, replayed, min(len(old), len(new)))
