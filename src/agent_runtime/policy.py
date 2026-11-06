"""The rules, as code. The model proposes; the policy decides. Nothing in a prompt can change a verdict here."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from .tools import Tool
from .types import ToolCall

ALLOW, ASK, DENY = "allow", "ask", "deny"
URL = re.compile(r"""https?://[^\s"'<>)\]]+""", re.I)
Approver = Callable[[ToolCall, str], bool]


@dataclass(frozen=True)
class Verdict:
    action: str                 # allow | ask | deny
    reason: str = ""


@dataclass
class Policy:
    """Defaults by risk: read is allowed, write needs approval, destructive is denied. Names override the defaults.

    Once untrusted content is in the conversation (``tainted``), anything that writes, deletes or sends data out needs a human, even
    if it would normally be allowed. That is the rule that stops a web page from steering the agent into doing harm.
    """

    allow: set[str] = field(default_factory=set)
    ask: set[str] = field(default_factory=set)
    deny: set[str] = field(default_factory=set)
    allowed_hosts: set[str] | None = None           # None = no URL check; a set = URLs in arguments must point at these hosts
    max_calls_per_tool: dict[str, int] = field(default_factory=dict)
    by_risk: dict[str, str] = field(default_factory=lambda: {"read": ALLOW, "write": ASK, "destructive": DENY})

    def __post_init__(self) -> None:
        both = (self.allow & self.deny) | (self.allow & self.ask) | (self.ask & self.deny)
        if both:
            raise ValueError(f"Tool(s) {sorted(both)} are listed in more than one of allow/ask/deny")
        if self.allowed_hosts is not None:
            self.allowed_hosts = {h.lower() for h in self.allowed_hosts}

    def evaluate(self, tool: Tool, call: ToolCall, *, tainted: bool, calls_so_far: dict[str, int]) -> Verdict:
        if tool.name in self.deny:
            return Verdict(DENY, f"'{tool.name}' is on the deny list")
        limit = self.max_calls_per_tool.get(tool.name)
        if limit is not None and calls_so_far.get(tool.name, 0) >= limit:
            return Verdict(DENY, f"'{tool.name}' reached its limit of {limit} call(s) in this run")
        blocked = self._blocked_url(call.args)
        if blocked:
            return Verdict(DENY, f"the call points at '{blocked}', which is not an allowed host")
        base = ALLOW if tool.name in self.allow else ASK if tool.name in self.ask else self.by_risk[tool.risk]
        if base == DENY:
            return Verdict(DENY, f"'{tool.name}' is {tool.risk} and {tool.risk} tools are denied by default")
        if tainted and (tool.risk != "read" or tool.egress) and base == ALLOW:
            return Verdict(ASK, "untrusted content is in the conversation, so this needs a human (even for an allow-listed tool)")
        return Verdict(base, f"{tool.risk} tool" if base == ASK else "")

    def _blocked_url(self, args: Any) -> str | None:
        if self.allowed_hosts is None:
            return None
        for text in _strings(args):
            for match in URL.findall(text):
                host = (urlparse(match).hostname or "").lower()
                if host not in self.allowed_hosts:
                    return host or match[:40]
        return None


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def deny_all_approver(call: ToolCall, reason: str) -> bool:
    """The default when no human is attached: anything that needs approval is refused."""
    return False


POLICY_KEYS = {"allow", "ask", "deny", "allowed_hosts", "max_calls_per_tool", "by_risk"}


def policy_from_dict(data: dict[str, Any]) -> Policy:
    """Build a policy from a JSON-style dictionary. Unknown keys are errors: a typo must not quietly weaken a safety rule."""
    unknown = set(data) - POLICY_KEYS
    if unknown:
        raise ValueError(f"Unknown policy key(s) {sorted(unknown)}. Allowed: {sorted(POLICY_KEYS)}")
    kwargs: dict[str, Any] = {k: set(data[k]) for k in ("allow", "ask", "deny") if k in data}
    if "allowed_hosts" in data:
        kwargs["allowed_hosts"] = set(data["allowed_hosts"])
    if "max_calls_per_tool" in data:
        kwargs["max_calls_per_tool"] = dict(data["max_calls_per_tool"])
    if "by_risk" in data:
        by_risk = {**{"read": ALLOW, "write": ASK, "destructive": DENY}, **data["by_risk"]}
        bad = {v for v in by_risk.values() if v not in (ALLOW, ASK, DENY)}
        if bad:
            raise ValueError(f"by_risk values must be allow, ask or deny, not {sorted(bad)}")
        kwargs["by_risk"] = by_risk
    return Policy(**kwargs)
