"""The agent loop: chat, let the model call tools, feed results back."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .client import Message, MistralClient
from .longterm import (
    DEFAULT_LONGTERM_INTERVAL,
    DEFAULT_LONGTERM_MAX_CHARS,
    LONGTERM_DUE_NOTE,
    LONGTERM_MEMO,
    LONGTERM_SNIPPET_CHARS,
    append_longterm,
    longterm_due,
    read_longterm,
    register_longterm_tool,
)
from .tools import ToolRegistry

DEFAULT_MAX_STEPS = 16
DEFAULT_HISTORY_LIMIT = 20
DEFAULT_HISTORY_CHAR_BUDGET = 50_000
ARCHIVE_LINE_CAP = 2_000
DEFAULT_ARCHIVE_MAX_CHARS = 100_000
ARCHIVE_NOTE = (
    "\n\nOlder conversation turns are archived at {path}; use the read_file tool "
    "to consult that archive when details from earlier in the project matter."
)

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
        memory_archive_path: str | None = None,
        memory_archive_max_chars: int | None = DEFAULT_ARCHIVE_MAX_CHARS,
        longterm_path: str | None = None,
        longterm_interval: int = DEFAULT_LONGTERM_INTERVAL,
        longterm_max_chars: int | None = DEFAULT_LONGTERM_MAX_CHARS,
    ) -> None:
        self.client = client
        self.registry = registry
        if memory_archive_path:
            system_prompt += ARCHIVE_NOTE.format(path=memory_archive_path)
        self.longterm_path = longterm_path
        self.longterm_interval = longterm_interval
        self.longterm_max_chars = longterm_max_chars
        self.runs = 0
        self.last_report_run = 0
        if longterm_path:
            system_prompt += "\n\n" + LONGTERM_MEMO.format(
                path=longterm_path, interval=longterm_interval
            )
            self._register_longterm_tool()
        self.system_prompt = system_prompt
        self.memory_archive_path = memory_archive_path
        self.memory_archive_max_chars = memory_archive_max_chars
        self.max_steps = max_steps
        self.on_step = on_step
        self.history_limit = history_limit
        self.history_char_budget = history_char_budget
        self.history: list[Message] = [Message(role="system", content=system_prompt)]

    def reset(self) -> None:
        self.history = [Message(role="system", content=self.system_prompt)]

    def _refresh_system_prompt(self) -> None:
        """Rebuild the system message with the long-term memory snapshot and due state."""
        if not self.longterm_path:
            return
        parts = [self.system_prompt]
        snapshot = read_longterm(self.longterm_path, max_chars=LONGTERM_SNIPPET_CHARS)
        if snapshot:
            parts.append("\n\nCurrent long-term memory snapshot:\n\n" + snapshot)
        if longterm_due(self.runs, self.last_report_run, self.longterm_interval):
            parts.append("\n\n" + LONGTERM_DUE_NOTE.format(number=self.runs))
        self.history[0] = Message(role="system", content="".join(parts))

    def _register_longterm_tool(self) -> None:
        agent = self

        def report(summary: str) -> str:
            text = (summary or "").strip()
            if not text:
                return "Error: summary must be a non-empty string"
            append_longterm(
                agent.longterm_path,
                agent.runs,
                text,
                max_chars=agent.longterm_max_chars,
            )
            agent.last_report_run = agent.runs
            return f"recorded long-term memory update for run {agent.runs}"

        try:
            register_longterm_tool(
                self.registry, report, interval=self.longterm_interval
            )
        except ValueError:
            pass

    def run_turn(self, user_input: str) -> str:
        """Process one user message, running any tool calls, and reply."""
        self.runs += 1
        self._refresh_system_prompt()
        checkpoint = len(self.history)
        self.history.append(Message(role="user", content=user_input))
        try:
            for _ in range(self.max_steps):
                response = self.client.chat(self.history, self.registry.schemas())
                message = response.message
                if not message.tool_calls:
                    self.history.append(message)
                    self._trim_history(archive=True)
                    return message.content or ""
                self.history.append(message)
                for call in message.tool_calls:
                    self._execute_call(call)
            raise RuntimeError(
                f"agent exceeded {self.max_steps} tool steps without producing a reply"
            )
        except Exception:
            del self.history[checkpoint:]
            raise

    def _trim_history(self, archive: bool = False) -> None:
        """Keep the system prompt plus recent history within count and char limits."""
        dropped: list[Message] = []
        while True:
            over_count = self.history_limit is not None and len(self.history) > self.history_limit
            over_chars = (
                self.history_char_budget is not None
                and self._history_chars() > self.history_char_budget
            )
            if not (over_count or over_chars) or len(self.history) <= 2:
                break
            dropped.extend(self._drop_oldest())
        while len(self.history) > 1 and self.history[1].role == "tool":
            dropped.append(self.history.pop(1))
        if archive and dropped and self.memory_archive_path:
            archive_messages(
                self.memory_archive_path,
                dropped,
                max_chars=self.memory_archive_max_chars,
            )

    def _history_chars(self) -> int:
        total = 0
        for message in self.history:
            total += len(message.content or "") + 16
            if message.tool_calls:
                total += sum(len(json.dumps(call)) for call in message.tool_calls)
        return total

    def _drop_oldest(self) -> list[Message]:
        dropped = self.history.pop(1)
        removed = [dropped]
        if dropped.tool_calls:
            drop_ids = {
                call.get("id") for call in dropped.tool_calls if call.get("id")
            }
            kept = []
            for message in self.history[1:]:
                if message.role == "tool" and message.tool_call_id in drop_ids:
                    removed.append(message)
                    continue
                kept.append(message)
            self.history = [self.history[0], *kept]
        return removed

    def run_turn_stream(
        self,
        user_input: str,
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        """Like run_turn, but streams reply deltas via on_delta as they arrive."""
        self.runs += 1
        self._refresh_system_prompt()
        checkpoint = len(self.history)
        self.history.append(Message(role="user", content=user_input))
        schemas = self.registry.schemas()
        try:
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
                    self._trim_history(archive=True)
                    return message.content or ""
                for call in message.tool_calls:
                    self._execute_call(call)
            raise RuntimeError(
                f"agent exceeded {self.max_steps} tool steps without producing a reply"
            )
        except Exception:
            del self.history[checkpoint:]
            raise

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

def archive_messages(
    path: str,
    messages: list[Message],
    max_chars: int | None = DEFAULT_ARCHIVE_MAX_CHARS,
) -> None:
    """Append trimmed conversation messages to a markdown archive file."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    lines: list[str] = []
    for message in messages:
        text = (message.content or "").strip()
        if len(text) > ARCHIVE_LINE_CAP:
            text = text[:ARCHIVE_LINE_CAP] + " …"
        if message.tool_calls:
            calls = "; ".join(
                (call.get("function") or {}).get("name", "unknown")
                for call in message.tool_calls
            )
            lines.append(
                f"- assistant tool_calls: {calls}"
                + (f" (said: {text})" if text else "")
            )
        elif message.role == "tool":
            lines.append(f"- tool result ({message.name}): {text}")
        else:
            lines.append(f"- {message.role}: {text}")
    header = "" if os.path.exists(path) else "# Conversation memory archive\n\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(header + "\n".join(lines) + "\n")
    if max_chars is not None:
        compact_archive(path, max_chars)


def compact_archive(path: str, max_chars: int) -> None:
    """Keep only the most recent archive lines when the file exceeds max_chars."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    if len(content) <= max_chars:
        return
    cut = len(content) - max_chars
    newline = content.find("\n", cut)
    keep_from = newline + 1 if newline != -1 else len(content)
    note = (
        f"\n_(older entries were pruned to stay under {max_chars} characters "
        f"on {datetime.now(timezone.utc).date()})_\n"
    )
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write("# Conversation memory archive\n\n" + note + "\n" + content[keep_from:])
    os.replace(tmp_path, path)


def save_history(agent: Agent, path: str) -> None:
    """Write the agent conversation history and run bookkeeping to a JSON file."""
    data = {
        "runs": agent.runs,
        "last_report_run": agent.last_report_run,
        "messages": [message.to_dict() for message in agent.history],
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp_path, path)


def load_history(agent: Agent, path: str) -> None:
    """Load conversation history from a JSON file into the agent.

    Accepts either the current dict format (with runs bookkeeping) or the
    legacy bare list of messages.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        raw_messages = data.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError(f"history file {path!r} must contain a non-empty list")
        agent.runs = int(data.get("runs", 0))
        agent.last_report_run = int(data.get("last_report_run", 0))
    elif isinstance(data, list):
        raw_messages = data
        if not data:
            raise ValueError(f"history file {path!r} must contain a non-empty list")
    else:
        raise ValueError(f"history file {path!r} must contain a non-empty list")
    messages: list[Message] = []
    for entry in raw_messages:
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
    agent._trim_history(archive=True)
