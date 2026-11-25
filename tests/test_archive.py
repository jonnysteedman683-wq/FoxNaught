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
    compact_archive,
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
    return Agent(client, registry, memory_archive_path=archive_path, memory_archive_max_chars=None)


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

    def test_archive_compacts_when_over_max_chars(self) -> None:
        archive_messages(
            self.archive,
            [Message(role="user", content=f"entry {i} " + "y" * 100) for i in range(60)],
            max_chars=100_000,
        )
        archive_messages(
            self.archive,
            [Message(role="user", content="final entry " + "z" * 100)],
            max_chars=1000,
        )
        with open(self.archive, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("older entries were pruned", content)
        self.assertIn("- user: final entry", content)
        self.assertNotIn("entry 0", content)
        self.assertLess(len(content), 2000)

    def test_compact_archive_ignores_small_files(self) -> None:
        archive_messages(self.archive, [Message(role="user", content="small")])
        compact_archive(self.archive, 100_000)
        with open(self.archive, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("- user: small", content)
        self.assertNotIn("pruned", content)

    def test_agent_uses_configured_compaction_limit(self) -> None:
        agent = make_agent(self.archive)
        agent.memory_archive_max_chars = 500
        agent.history_limit = 3
        side = [
            ChatResponse(message=Message(role="assistant", content="r" * 100))
            for _ in range(8)
        ]
        agent.client.chat.side_effect = side
        for i in range(8):
            agent.run_turn(f"question {i} " + "w" * 80)
        with open(self.archive, encoding="utf-8") as fh:
            self.assertLess(len(fh.read()), 1000)

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


def entries(content: str) -> list[str]:
    return [line for line in content.splitlines() if line.startswith("- ")]


class ArchiveEdgeCaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.archive = os.path.join(self.dir.name, "MEMORY.md")

    def read(self) -> str:
        with open(self.archive, encoding="utf-8") as fh:
            return fh.read()

    def seed(self, count: int, width: int = 50) -> None:
        archive_messages(
            self.archive,
            [Message(role="user", content=f"e{i} " + "x" * width) for i in range(count)],
            max_chars=None,
        )

    def test_empty_messages_is_noop(self) -> None:
        archive_messages(self.archive, [])
        self.assertFalse(os.path.exists(self.archive))
        self.seed(1)
        before = self.read()
        archive_messages(self.archive, [])
        self.assertEqual(self.read(), before)

    def test_header_written_once_across_appends(self) -> None:
        archive_messages(self.archive, [Message(role="user", content="a")], max_chars=None)
        archive_messages(self.archive, [Message(role="user", content="b")], max_chars=None)
        content = self.read()
        self.assertEqual(content.count("# Conversation memory archive"), 1)
        self.assertEqual(entries(content), ["- user: a", "- user: b"])

    def test_max_chars_none_never_compacts(self) -> None:
        self.seed(200, width=600)
        self.assertGreater(len(self.read()), 100_000)
        self.assertNotIn("pruned", self.read())

    def test_line_cap_boundary(self) -> None:
        archive_messages(
            self.archive,
            [
                Message(role="user", content="a" * ARCHIVE_LINE_CAP),
                Message(role="user", content="b" * (ARCHIVE_LINE_CAP + 1)),
            ],
            max_chars=None,
        )
        exact, over = entries(self.read())
        self.assertEqual(exact, "- user: " + "a" * ARCHIVE_LINE_CAP)
        self.assertEqual(over, "- user: " + "b" * ARCHIVE_LINE_CAP + " …")

    def test_lone_surrogate_content_is_escaped_not_fatal(self) -> None:
        archive_messages(self.archive, [Message(role="assistant", content="bad \ud83d end")], max_chars=None)
        self.assertIn("- assistant: bad \\ud83d end", self.read())

    def test_max_chars_zero_keeps_only_title_and_note(self) -> None:
        self.seed(10)
        archive_messages(self.archive, [Message(role="user", content="newest")], max_chars=0)
        content = self.read()
        self.assertTrue(content.startswith("# Conversation memory archive\n"))
        self.assertEqual(content.count("older entries were pruned"), 1)
        self.assertEqual(entries(content), [])
        archive_messages(self.archive, [Message(role="user", content="again")], max_chars=0)
        self.assertEqual(self.read(), content)

    def test_compaction_stays_within_max_chars_and_keeps_whole_lines(self) -> None:
        self.seed(40)
        compact_archive(self.archive, 600)
        content = self.read()
        self.assertLessEqual(len(content), 600)
        kept = entries(content)
        self.assertTrue(kept)
        self.assertEqual(kept[-1], "- user: e39 " + "x" * 50)
        body = content.split(")_\n\n", 1)[1]
        self.assertEqual(body.splitlines(), kept)

    def test_compaction_is_idempotent(self) -> None:
        self.seed(40)
        compact_archive(self.archive, 600)
        first = self.read()
        compact_archive(self.archive, 600)
        self.assertEqual(self.read(), first)

    def test_repeated_small_appends_keep_single_title_and_note(self) -> None:
        self.seed(5, width=400)
        compact_archive(self.archive, 1000)
        for i in range(60):
            archive_messages(
                self.archive,
                [Message(role="user", content="z" * (i * 13 % 300 + 5))],
                max_chars=1000,
            )
            content = self.read()
            with self.subTest(i=i):
                self.assertEqual(content.count("# Conversation memory archive"), 1)
                self.assertLessEqual(content.count("older entries were pruned"), 1)
                self.assertLessEqual(len(content), 1000)

    def test_compaction_heals_duplicated_notes(self) -> None:
        note = "_(older entries were pruned to stay under 50 characters on 2026-01-01)_"
        with open(self.archive, "w", encoding="utf-8") as fh:
            fh.write(
                "# Conversation memory archive\n\n\n"
                + note + "\n\n\n" + note + "\n\n"
                + "".join(f"- user: line {i}\n" for i in range(50))
            )
        compact_archive(self.archive, 400)
        content = self.read()
        self.assertEqual(content.count("older entries were pruned"), 1)
        self.assertEqual(entries(content)[-1], "- user: line 49")

    def test_tiny_max_chars_on_missing_parent_dir(self) -> None:
        nested = os.path.join(self.dir.name, "a", "b", "MEMORY.md")
        archive_messages(nested, [Message(role="user", content="y" * 500)], max_chars=10)
        with open(nested, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("older entries were pruned", content)
        self.assertFalse(os.path.exists(nested + ".tmp"))

    def test_compact_missing_file_is_noop(self) -> None:
        compact_archive(self.archive, 0)
        self.assertFalse(os.path.exists(self.archive))

    def test_compact_empty_file_is_noop(self) -> None:
        open(self.archive, "w").close()
        compact_archive(self.archive, 0)
        self.assertEqual(self.read(), "")

    def test_compact_tolerates_invalid_utf8(self) -> None:
        with open(self.archive, "wb") as fh:
            fh.write(b"# Conversation memory archive\n\n" + b"- user: \xff\xfe\n" * 100)
        compact_archive(self.archive, 300)
        content = self.read()
        self.assertLessEqual(len(content), 300)
        self.assertIn("�", content)


if __name__ == "__main__":
    unittest.main()
