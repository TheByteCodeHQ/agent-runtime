"""The vocabulary of an agent run: what the model decides, what the tools return, and what the runtime says back."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    """What the model wants next: call a tool, or finish with an answer. Exactly one of the two."""

    call: ToolCall | None = None
    final: str | None = None
    usage: dict[str, int] = field(default_factory=dict)       # e.g. {"tokens": 412}

    def __post_init__(self) -> None:
        if (self.call is None) == (self.final is None):
            raise ValueError("A decision is either a tool call or a final answer, not both and not neither")


@dataclass(frozen=True)
class Message:
    role: str                       # user | assistant | tool | system
    content: str
    trusted: bool = True            # False for anything that came from outside: web pages, files, emails, tool output
    tool: str | None = None


class Model(Protocol):
    name: str

    def decide(self, history: list[Message], tools: list[dict[str, Any]]) -> Decision:
        """Look at the conversation and the tools on offer, and choose the next step."""
        ...


@dataclass
class RunReport:
    status: str                     # finished | budget_exceeded | model_error
    answer: str | None
    reason: str = ""
    steps: int = 0
    tool_calls: int = 0
    denied: int = 0
    approved: int = 0
    injections: int = 0
    trace_path: str | None = None
