"""The agent loop: chat, let the model call tools, feed results back."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from .client import Message, MistralClient
from .tools import ToolRegistry

DEFAULT_MAX_STEPS = 8
DEFAULT_HISTORY_LIMIT = 20
DEFAULT_HISTORY_CHAR_BUDGET = 50_000

SYSTEM_PROMPT = (
    "You are FoxNaught, a helpful assistant. Use the available tools when they "
    "answer the user's request more reliably than guessing, then summarise the "
    "results."
)


class Agent:
    def __init__(
        self,
        client: MistralClient,
        registry: ToolRegistry,
        system_prompt: str = SYSTEM_PROMPT,
        max_steps: int = DEFAULT_MAX_STEPS,
        on_step: Callable[[str, object], None] | None = None,
        history_limit: int | None = DEFAULT_HISTORY_LIMIT,
        history_char_budget: int | None = DEFAULT_HISTORY_CHAR_BUDGET,
    ) -> None:
        self.client = client
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.on_step = on_step
        self.history_limit = history_limit
        self.history_char_budget = history_char_budget
        self.history: list[Message] = [Message(role="system", content=system_prompt)]

    def reset(self) -> None:
        self.history = [Message(role="system", content=self.system_prompt)]

    def run_turn(self, user_input: str) -> str:
        """Process one user message, running any tool calls, and reply."""
        self.history.append(Message(role="user", content=user_input))
        for _ in range(self.max_steps):
            response = self.client.chat(self.history, self.registry.schemas())
            message = response.message
            if not message.tool_calls:
                self.history.append(message)
                self._trim_history()
                return message.content or ""
            self.history.append(message)
            for call in message.tool_calls:
                self._execute_call(call)
        raise RuntimeError(
            f"agent exceeded {self.max_steps} tool steps without producing a reply"
        )

    def _trim_history(self) -> None:
        """Keep the system prompt plus recent history within count and char limits."""
        while True:
            over_count = self.history_limit is not None and len(self.history) > self.history_limit
            over_chars = (
                self.history_char_budget is not None
                and self._history_chars() > self.history_char_budget
            )
            if not (over_count or over_chars) or len(self.history) <= 2:
                break
            self._drop_oldest()
        while len(self.history) > 1 and self.history[1].role == "tool":
            self.history.pop(1)

    def _history_chars(self) -> int:
        total = 0
        for message in self.history:
            total += len(message.content or "") + 16
            if message.tool_calls:
                total += sum(len(json.dumps(call)) for call in message.tool_calls)
        return total

    def _drop_oldest(self) -> None:
        dropped = self.history.pop(1)
        if dropped.tool_calls:
            drop_ids = {
                call.get("id") for call in dropped.tool_calls if call.get("id")
            }
            kept = [m for m in self.history[1:] if m.role != "tool" or m.tool_call_id not in drop_ids]
            self.history = [self.history[0], *kept]

    def run_turn_stream(
        self,
        user_input: str,
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        """Like run_turn, but streams reply deltas via on_delta as they arrive."""
        self.history.append(Message(role="user", content=user_input))
        schemas = self.registry.schemas()
        for _ in range(self.max_steps):
            content_parts: list[str] = []
            tool_acc: dict[int, dict[str, Any]] = {}
            for event in self.client.chat_stream(self.history, schemas):
                delta = event.delta
                piece = delta.get("content")
                if piece and on_delta is not None:
                    on_delta(piece)
                if piece:
                    content_parts.append(piece)
                for partial in delta.get("tool_calls") or []:
                    self._merge_partial_tool_call(tool_acc, partial)
            message = Message(
                role="assistant",
                content="".join(content_parts) or None,
                tool_calls=list(tool_acc.values()) or None,
            )
            self.history.append(message)
            if not message.tool_calls:
                self._trim_history()
                return message.content or ""
            for call in message.tool_calls:
                self._execute_call(call)
        raise RuntimeError(
            f"agent exceeded {self.max_steps} tool steps without producing a reply"
        )

    @staticmethod
    def _merge_partial_tool_call(
        tool_acc: dict[int, dict[str, Any]], partial: dict[str, Any]
    ) -> None:
        index = partial.get("index", 0)
        if index not in tool_acc:
            existing = dict(partial)
            if "function" in existing:
                existing["function"] = dict(existing["function"])
            tool_acc[index] = existing
            return
        accumulated = tool_acc[index]
        if partial.get("id"):
            accumulated["id"] = partial["id"]
        if partial.get("type"):
            accumulated["type"] = partial["type"]
        partial_function = partial.get("function")
        if partial_function:
            function = accumulated.setdefault("function", {})
            if partial_function.get("name"):
                function["name"] = partial_function["name"]
            if partial_function.get("arguments"):
                function["arguments"] = function.get("arguments", "") + partial_function["arguments"]

    def _execute_call(self, call: dict) -> None:
        function = call.get("function") or {}
        name = function.get("name", "")
        arguments = function.get("arguments")
        if self.on_step is not None:
            self.on_step(name, arguments)
        try:
            result = self.registry.run(name, arguments)
        except KeyError:
            result = f"Error: unknown tool: {name}"
        self.history.append(
            Message(
                role="tool",
                name=name,
                tool_call_id=call.get("id"),
                content=result,
            )
        )

def save_history(agent: Agent, path: str) -> None:
    """Write the agent conversation history to a JSON file."""
    data = [message.to_dict() for message in agent.history]
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp_path, path)


def load_history(agent: Agent, path: str) -> None:
    """Load conversation history from a JSON file into the agent."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list) or not data:
        raise ValueError(f"history file {path!r} must contain a non-empty list")
    messages: list[Message] = []
    for entry in data:
        if not isinstance(entry, dict) or "role" not in entry:
            raise ValueError(f"history file {path!r} has an invalid message entry")
        entry = dict(entry)
        role = entry.pop("role")
        messages.append(
            Message(
                role=role,
                content=entry.get("content"),
                tool_calls=entry.get("tool_calls"),
                name=entry.get("name"),
                tool_call_id=entry.get("tool_call_id"),
            )
        )
    if messages[0].role != "system":
        messages.insert(0, Message(role="system", content=agent.system_prompt))
    agent.history = messages
    agent._trim_history()
