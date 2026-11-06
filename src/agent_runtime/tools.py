"""Tools the agent may call. Every tool declares its risk, and every call's arguments are validated before the function runs."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

RISKS = ("read", "write", "destructive")


class ToolError(Exception):
    """A tool failed in a way the model should be told about (bad input, not found). Not a bug in the runtime."""


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    schema: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    risk: str = "read"
    untrusted_output: bool = False      # the result comes from outside (web, files, mail): treat it as data, never as instructions
    egress: bool = False                # the call sends data out of the system (HTTP, email, chat)

    def __post_init__(self) -> None:
        if self.risk not in RISKS:
            raise ValueError(f"{self.name}: risk must be one of {RISKS}")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", self.name):
            raise ValueError(f"Invalid tool name '{self.name}'")

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.schema, "risk": self.risk}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        return tool

    def tool(self, *, risk: str = "read", schema: dict[str, Any] | None = None, untrusted_output: bool = False, egress: bool = False):
        """Decorator form: the function name is the tool name and its docstring is the description."""

        def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(Tool(func.__name__, (func.__doc__ or func.__name__).strip().splitlines()[0], func,
                               schema or {"type": "object", "properties": {}}, risk, untrusted_output, egress))
            return func

        return wrap

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def describe(self) -> list[dict[str, Any]]:
        return [t.describe() for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def validate_args(args: Any, schema: dict[str, Any], path: str = "args") -> list[str]:
    """A small JSON Schema subset: type, enum, required, properties, additionalProperties, items, bounds, length and pattern."""
    kinds = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}
    kind = schema.get("type")
    if kind == "integer":
        ok = isinstance(args, int) and not isinstance(args, bool)
    elif kind == "number":
        ok = isinstance(args, (int, float)) and not isinstance(args, bool)
    elif kind in kinds:
        ok = isinstance(args, kinds[kind]) and (kind == "boolean" or not isinstance(args, bool))
    else:
        ok = True
    if not ok:
        return [f"{path}: expected {kind}, got {type(args).__name__}"]
    errors: list[str] = []
    if "enum" in schema and args not in schema["enum"]:
        errors.append(f"{path}: {args!r} is not one of {schema['enum']}")
    if isinstance(args, (int, float)) and not isinstance(args, bool):
        if "minimum" in schema and args < schema["minimum"]:
            errors.append(f"{path}: {args} is below the minimum {schema['minimum']}")
        if "maximum" in schema and args > schema["maximum"]:
            errors.append(f"{path}: {args} is above the maximum {schema['maximum']}")
    if isinstance(args, str):
        if "maxLength" in schema and len(args) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']} characters")
        if "minLength" in schema and len(args) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], args):
            errors.append(f"{path}: does not match {schema['pattern']!r}")
    if isinstance(args, dict):
        errors += [f"{path}: missing required field '{n}'" for n in schema.get("required", []) if n not in args]
        props = schema.get("properties", {})
        for name, sub in props.items():
            if name in args:
                errors += validate_args(args[name], sub, f"{path}.{name}")
        if schema.get("additionalProperties") is False:
            errors += [f"{path}: unexpected field '{n}'" for n in args if n not in props]
    if isinstance(args, list):
        if "maxItems" in schema and len(args) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for index, item in enumerate(args[:200]):
                errors += validate_args(item, schema["items"], f"{path}[{index}]")
    return errors
