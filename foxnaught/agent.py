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
        """Keep the system prompt plus recent history within count and char limits.
        
        Trims oldest messages first, preferring to keep tool call/result pairs together.
        Also removes orphaned tool messages (tool results without their call).
        """
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

ARCHIVE_HEADER = "# Conversation memory archive\n\n"
_ARCHIVE_TITLE = ARCHIVE_HEADER.strip()
_PRUNE_NOTE_PREFIX = "_(older entries were pruned"
HISTORY_ROLES = frozenset({"system", "user", "assistant", "tool"})


def archive_messages(
    path: str,
    messages: list[Message],
    max_chars: int | None = DEFAULT_ARCHIVE_MAX_CHARS,
) -> None:
    """Append trimmed conversation messages to a markdown archive file.

    An empty ``messages`` list is a no-op. Text that cannot be encoded as
    UTF-8 (e.g. lone surrogates from a decoded JSON escape) is written with
    backslash escapes instead of aborting the trim that triggered the archive.
    """
    if not messages:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    lines: list[str] = []
    for message in messages:
        text = (message.content or "").strip()
        if len(text) > ARCHIVE_LINE_CAP:
            text = text[:ARCHIVE_LINE_CAP] + " …"
        # Indent continuation lines so only an entry's first line starts with
        # "- ", which lets compaction keep or drop each entry whole.
        text = text.replace("\n", "\n  ")
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
    header = "" if os.path.exists(path) else ARCHIVE_HEADER
    with open(path, "a", encoding="utf-8", errors="backslashreplace") as fh:
        fh.write(header + "\n".join(lines) + "\n")
    if max_chars is not None:
        compact_archive(path, max_chars)


def _archive_body(content: str) -> str:
    """Return the archive entries without the leading title and pruning notes."""
    lines = content.splitlines(keepends=True)
    start = 0
    while start < len(lines):
        stripped = lines[start].strip()
        if stripped and stripped != _ARCHIVE_TITLE and not stripped.startswith(_PRUNE_NOTE_PREFIX):
            break
        start += 1
    return "".join(lines[start:])


def compact_archive(path: str, max_chars: int) -> None:
    """Keep only the most recent whole archive lines when the file exceeds max_chars.

    The rewritten file is the title, a single pruning note and as many of the
    newest entries as fit, so it stays within ``max_chars`` and is left alone by
    the next call. A ``max_chars`` smaller than the title and note keeps those
    alone. Missing files are ignored.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8", errors="replace") as fh:
        content = fh.read()
    if len(content) <= max_chars:
        return
    note = (
        f"\n{_PRUNE_NOTE_PREFIX} to stay under {max_chars} characters "
        f"on {datetime.now(timezone.utc).date()})_\n\n"
    )
    preamble = ARCHIVE_HEADER + note
    body = _archive_body(content)
    budget = max(max_chars - len(preamble), 0)
    if len(body) > budget:
        newline = body.find("\n", len(body) - budget - 1)
        body = body[newline + 1 :] if newline != -1 else ""
        if not body.startswith("- "):
            entry = body.find("\n- ")
            body = body[entry + 1 :] if entry != -1 else ""
    if preamble + body == content:
        return
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(preamble + body)
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


def _history_counter(data: dict[str, Any], key: str, path: str) -> int:
    value = data.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"history file {path!r} has an invalid {key!r}: {value!r}")
    return value


def _history_message(entry: Any, index: int, path: str) -> Message:
    def invalid(reason: str) -> ValueError:
        return ValueError(f"history file {path!r} has an invalid message entry at index {index}: {reason}")

    if not isinstance(entry, dict):
        raise invalid("not an object")
    role = entry.get("role")
    if not isinstance(role, str) or role not in HISTORY_ROLES:
        raise invalid(f"role must be one of {sorted(HISTORY_ROLES)}, got {role!r}")
    for key in ("content", "name", "tool_call_id"):
        if entry.get(key) is not None and not isinstance(entry[key], str):
            raise invalid(f"{key} must be a string")
    tool_calls = entry.get("tool_calls")
    if tool_calls is not None and (
        not isinstance(tool_calls, list) or not all(isinstance(call, dict) for call in tool_calls)
    ):
        raise invalid("tool_calls must be a list of objects")
    for call in tool_calls or []:
        function = call.get("function")
        if call.get("id") is not None and not isinstance(call["id"], str):
            raise invalid("tool call id must be a string")
        if function is not None and (
            not isinstance(function, dict)
            or (function.get("name") is not None and not isinstance(function["name"], str))
        ):
            raise invalid("tool call function must be an object with a string name")
    return Message(
        role=role,
        content=entry.get("content"),
        tool_calls=tool_calls,
        name=entry.get("name"),
        tool_call_id=entry.get("tool_call_id"),
    )


def load_history(agent: Agent, path: str) -> None:
    """Load conversation history from a JSON file into the agent.

    Accepts either the current dict format (with runs bookkeeping) or the
    legacy bare list of messages. The whole file is validated before the agent
    is touched, so a failed load leaves its history and counters unchanged.
    Malformed content raises ``ValueError`` (``json.JSONDecodeError`` and
    ``UnicodeDecodeError`` are subclasses); a missing file raises
    ``FileNotFoundError``.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    runs, last_report_run = agent.runs, agent.last_report_run
    if isinstance(data, dict):
        raw_messages = data.get("messages")
        runs = _history_counter(data, "runs", path)
        last_report_run = _history_counter(data, "last_report_run", path)
    else:
        raw_messages = data
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError(f"history file {path!r} must contain a non-empty list")
    messages = [_history_message(entry, index, path) for index, entry in enumerate(raw_messages)]
    if messages[0].role != "system":
        messages.insert(0, Message(role="system", content=agent.system_prompt))
    agent.runs = runs
    agent.last_report_run = last_report_run
    agent.history = messages
    agent._trim_history(archive=True)
