"""Tests for the conversation memory archive."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from foxnaught.agent import (
    ARCHIVE_LINE_CAP,
    SYSTEM_PROMPT,
    Agent,
    archive_messages,
    load_history,
    save_history,
)
from foxnaught.builtins import register_defaults
from foxnaught.client import ChatResponse, Message, MistralClient
from foxnaught.tools import ToolRegistry


def make_agent(archive_path: str | None) -> Agent:
    registry = ToolRegistry()
    register_defaults(registry)
    client = mock.create_autospec(MistralClient, instance=True)
    return Agent(client, registry, memory_archive_path=archive_path)


class ArchiveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.archive = os.path.join(self.dir.name, "MEMORY.md")

    def test_archive_disabled_leaves_system_prompt_alone(self) -> None:
        agent = make_agent(None)
        self.assertEqual(agent.history[0].content, SYSTEM_PROMPT)
        self.assertIsNone(agent.memory_archive_path)

    def test_archive_enabled_notes_path_in_system_prompt(self) -> None:
        agent = make_agent(self.archive)
        self.assertIn(self.archive, agent.history[0].content)
        self.assertIn("read_file", agent.history[0].content)

    def test_trimmed_messages_are_archived(self) -> None:
        agent = make_agent(self.archive)
        agent.history_limit = 4
        side = [
            ChatResponse(message=Message(role="assistant", content=f"reply {i}"))
            for i in range(6)
        ]
        agent.client.chat.side_effect = side
        for i in range(6):
            agent.run_turn(f"msg {i}")
        with open(self.archive, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("# Conversation memory archive", content)
        self.assertIn("- user: msg 0", content)
        self.assertIn("- assistant: reply 0", content)
        self.assertNotIn("- user: msg 5", content)

    def test_tool_pair_is_archived_together(self) -> None:
        agent = make_agent(self.archive)
        agent.history_limit = 3
        side = [
            ChatResponse(
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {"id": "c1", "type": "function", "function": {"name": "now_utc", "arguments": None}}
                    ],
                )
            ),
            ChatResponse(message=Message(role="assistant", content="done 1")),
            ChatResponse(message=Message(role="assistant", content="done 2")),
            ChatResponse(message=Message(role="assistant", content="done 3")),
        ]
        agent.client.chat.side_effect = side
        agent.run_turn("need the time")
        with open(self.archive, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("assistant tool_calls: now_utc", content)
        self.assertIn("tool result (now_utc):", content)

    def test_long_messages_are_truncated(self) -> None:
        agent = make_agent(self.archive)
        agent.history_limit = 3
        side = [
            ChatResponse(message=Message(role="assistant", content="x" * (ARCHIVE_LINE_CAP + 500)))
            for _ in range(5)
        ]
        agent.client.chat.side_effect = side
        for i in range(5):
            agent.run_turn(f"q{i}")
        with open(self.archive, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("…", content)
        self.assertNotIn("x" * (ARCHIVE_LINE_CAP + 500), content)

    def test_no_archive_written_without_drops(self) -> None:
        agent = make_agent(self.archive)
        agent.client.chat.side_effect = [
            ChatResponse(message=Message(role="assistant", content="hi"))
        ]
        agent.run_turn("hello")
        self.assertFalse(os.path.exists(self.archive))

    def test_archive_messages_direct_call_creates_parent_dirs(self) -> None:
        nested = os.path.join(self.dir.name, "sub", "dir", "MEMORY.md")
        archive_messages(nested, [Message(role="user", content="hello there")])
        with open(nested, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("- user: hello there", content)

    def test_load_history_trims_without_duplicating_stable_archive(self) -> None:
        agent = make_agent(self.archive)
        for i in range(8):
            agent.history.append(Message(role="user", content=f"u{i}"))
            agent.history.append(Message(role="assistant", content=f"r{i}"))
        save_history(agent, os.path.join(self.dir.name, "memory.json"))
        reloaded = make_agent(self.archive)
        before = os.path.getsize(self.archive) if os.path.exists(self.archive) else 0
        load_history(reloaded, os.path.join(self.dir.name, "memory.json"))
        after = os.path.getsize(self.archive) if os.path.exists(self.archive) else 0
        self.assertEqual(before, after)
        self.assertEqual(len(reloaded.history), 17)


if __name__ == "__main__":
    unittest.main()
