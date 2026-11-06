"""Hard limits that end a run. An agent that can loop forever will, sooner or later, so the limits are not optional."""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from .types import ToolCall


class BudgetExceeded(Exception):
    pass


@dataclass
class Budget:
    max_steps: int = 20
    max_tool_calls: int = 30
    max_seconds: float = 120.0
    max_tokens: int | None = None
    max_repeats: int = 3                    # the same tool with the same arguments, this many times, is a loop

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_tool_calls", "max_repeats"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")


@dataclass
class BudgetTracker:
    budget: Budget
    clock: Callable[[], float] = time.monotonic
    steps: int = 0
    tool_calls: int = 0
    tokens: int = 0
    started: float = field(default=0.0)
    _seen: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self.started = self.clock()

    def start_step(self) -> None:
        if self.steps >= self.budget.max_steps:
            raise BudgetExceeded(f"step limit of {self.budget.max_steps} reached")
        if self.clock() - self.started > self.budget.max_seconds:
            raise BudgetExceeded(f"time limit of {self.budget.max_seconds:g}s reached")
        self.steps += 1

    def add_usage(self, usage: dict[str, int]) -> None:
        self.tokens += int(usage.get("tokens", 0))
        if self.budget.max_tokens is not None and self.tokens > self.budget.max_tokens:
            raise BudgetExceeded(f"token limit of {self.budget.max_tokens} exceeded ({self.tokens} used)")

    def count_call(self, call: ToolCall) -> None:
        if self.tool_calls >= self.budget.max_tool_calls:
            raise BudgetExceeded(f"tool-call limit of {self.budget.max_tool_calls} reached")
        self.tool_calls += 1
        key = f"{call.tool}:{json.dumps(call.args, sort_keys=True, default=str)}"
        self._seen[key] += 1
        if self._seen[key] >= self.budget.max_repeats:
            raise BudgetExceeded(f"looping: '{call.tool}' was called {self._seen[key]} times with the same arguments")
