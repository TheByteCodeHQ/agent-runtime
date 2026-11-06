"""Models to drive the loop: a scripted one for tests and demos, and a Gemini adapter for real use."""

from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Callable
from typing import Any

from .types import Decision, Message, ToolCall


class ScriptedModel:
    """Plays back a fixed list of decisions. Items are Decision objects, or callables ``f(history) -> Decision`` to react to the
    conversation (so a test can check what the model was shown)."""

    name = "scripted"

    def __init__(self, script: list[Decision | Callable[[list[Message]], Decision]]) -> None:
        self._script = deque(script)
        self.seen: list[list[Message]] = []

    def decide(self, history: list[Message], tools: list[dict[str, Any]]) -> Decision:
        self.seen.append(history)
        if not self._script:
            raise RuntimeError("the script ran out of decisions")
        item = self._script.popleft()
        return item(history) if callable(item) else item


def call(tool: str, **args: Any) -> Decision:
    return Decision(call=ToolCall(tool, args))


def finish(answer: str) -> Decision:
    return Decision(final=answer)


DECISION_PROMPT = """Choose the next step. Reply with ONE JSON object and nothing else:
  {"tool": "<name>", "args": {...}}      to call a tool
  {"final": "<answer for the user>"}     when the task is done or cannot be done
Tools:
"""


class GeminiModel:
    """Asks Gemini for one JSON decision per step. Never exercised against the live API in this repo's tests."""

    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None, client: Any = None) -> None:
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        if client is not None:
            self._client, self._types = client, None
            return
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use the Gemini model.")
        try:
            from google import genai
        except ImportError:
            raise RuntimeError("Install the extra: pip install 'agent-runtime[gemini]'") from None
        self._client, self._types = genai.Client(api_key=key), genai.types

    def decide(self, history: list[Message], tools: list[dict[str, Any]]) -> Decision:
        system = "\n".join(m.content for m in history if m.role == "system")
        transcript = "\n\n".join(f"[{m.role}{' ' + m.tool if m.tool else ''}] {m.content}" for m in history if m.role != "system")
        prompt = DECISION_PROMPT + json.dumps(tools, indent=1) + "\n\nConversation so far:\n" + transcript
        config: Any = {"system_instruction": system, "temperature": 0.0, "response_mime_type": "application/json"}
        if self._types is not None:
            config = self._types.GenerateContentConfig(**config)
        response = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
        return parse_decision(response.text or "")


def parse_decision(raw: str) -> Decision:
    try:
        data = json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    except ValueError:
        raise ValueError(f"the model did not return JSON: {raw[:100]!r}") from None
    if not isinstance(data, dict):
        raise ValueError("the model's reply must be a JSON object")
    if isinstance(data.get("final"), str):
        return Decision(final=data["final"])
    if isinstance(data.get("tool"), str) and isinstance(data.get("args", {}), dict):
        return Decision(call=ToolCall(data["tool"], data.get("args", {})))
    raise ValueError(f"the model's reply is neither a tool call nor a final answer: {raw[:100]!r}")
