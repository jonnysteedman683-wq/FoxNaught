"""Tests for the distilled long-term memory tier."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

from foxnaught.agent import Agent, load_history, save_history
from foxnaught.builtins import register_defaults
from foxnaught.client import ChatResponse, Message, MistralClient
from foxnaught.longterm import (
    DEFAULT_LONGTERM_INTERVAL,
    DEFAULT_LONGTERM_MAX_CHARS,
    append_longterm,
    compact_longterm,
    longterm_due,
    read_longterm,
)
from foxnaught.tools import ToolRegistry


def make_agent(**kwargs) -> Agent:
    registry = ToolRegistry()
    register_defaults(registry)
    client = mock.create_autospec(MistralClient, instance=True)
    return Agent(client, registry, **kwargs)


def reply_agent(agent: Agent, text: str = "ok") -> Agent:
    agent.client.chat.return_value = ChatResponse(
        message=Message(role="assistant", content=text)
    )
    return agent


class LongTermFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "LONGTERM.md")

    def test_missing_file_reads_as_empty(self) -> None:
        self.assertEqual(read_longterm(self.path), "")

    def test_append_creates_file_with_header_and_section(self) -> None:
        append_longterm(self.path, 5, "user prefers terse replies")
        content = read_longterm(self.path)
        self.assertIn("# Long-term memory", content)
        self.assertIn("## Run 5 - ", content)
        self.assertIn("- user prefers terse replies", content)

    def test_append_deduplicates_date_for_same_day(self) -> None:
        append_longterm(self.path, 5, "fact a")
        append_longterm(self.path, 10, "fact b")
        content = read_longterm(self.path)
        self.assertIn("## Run 5 - ", content)
        self.assertIn("## Run 10 - ", content)
        self.assertIn("- fact a", content)
        self.assertIn("- fact b", content)

    def test_append_ignores_empty_summary(self) -> None:
        append_longterm(self.path, 3, "   ")
        self.assertFalse(os.path.exists(self.path))

    def test_read_longterm_returns_tail_within_budget(self) -> None:
        for i in range(1, 11):
            append_longterm(self.path, i * 5, f"fact {i} " + "z" * 40)
        tail = read_longterm(self.path, max_chars=300)
        self.assertIn("- fact 10 ", tail)
        self.assertNotIn("fact 1 ", tail)

    def test_compaction_drops_oldest_sections(self) -> None:
        for i in range(1, 6):
            append_longterm(self.path, i * 5, f"fact {i} " + "z" * 150)
        with open(self.path, encoding="utf-8") as fh:
            before = fh.read()
        self.assertIn("## Run 5 - ", before)
        compact_longterm(self.path, max_chars=600)
        with open(self.path, encoding="utf-8") as fh:
            after = fh.read()
        self.assertLessEqual(len(after), 600)
        self.assertIn("pruned", after)
        self.assertNotIn("- fact 1 ", after)

    def test_compaction_noop_for_small_file(self) -> None:
        append_longterm(self.path, 5, "small fact")
        with open(self.path, encoding="utf-8") as fh:
            before = fh.read()
        compact_longterm(self.path, max_chars=DEFAULT_LONGTERM_MAX_CHARS)
        with open(self.path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), before)


class DueLogicTest(unittest.TestCase):
    def test_due_math(self) -> None:
        self.assertFalse(longterm_due(0, 0, 5))
        self.assertFalse(longterm_due(4, 0, 5))
        self.assertTrue(longterm_due(5, 0, 5))
        self.assertFalse(longterm_due(9, 5, 5))
        self.assertTrue(longterm_due(10, 5, 5))
        self.assertTrue(longterm_due(1, 0, 0))


class AgentLongTermTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "LONGTERM.md")
        self.memory = os.path.join(self.dir.name, "memory.json")

    def test_missing_longterm_reads_as_empty(self) -> None:
        self.assertEqual(read_longterm(self.path), "")

    def test_init_appends_longterm_note(self) -> None:
        agent = make_agent(longterm_path=self.path)
        self.assertIn("report_longterm", agent.system_prompt)
        self.assertIn(str(DEFAULT_LONGTERM_INTERVAL), agent.system_prompt)

    def test_report_tool_registered_once_per_registry(self) -> None:
        registry = ToolRegistry()
        register_defaults(registry)
        client = mock.create_autospec(MistralClient, instance=True)
        Agent(client, registry, longterm_path=self.path)
        Agent(client, registry, longterm_path=self.path)
        self.assertEqual(
            len([t for t in registry._tools if t == "report_longterm"]), 1
        )

    def turn(self, agent: Agent, text: str = "ok") -> None:
        agent.client.chat.return_value = ChatResponse(
            message=Message(role="assistant", content=text)
        )
        agent.run_turn("hello")

    def test_due_note_appears_on_interval_run(self) -> None:
        agent = make_agent(longterm_path=self.path)
        reply_agent(agent)
        for _ in range(4):
            agent.run_turn("hello")
        self.assertNotIn("DUE", agent.history[0].content)
        agent.run_turn("hello")
        self.assertIn("DUE", agent.history[0].content)
        self.assertIn("run 5", agent.history[0].content)

    def test_tool_call_resets_due_state(self) -> None:
        agent = make_agent(longterm_path=self.path)
        response = ChatResponse(
            message=Message(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {
                            "name": "report_longterm",
                            "arguments": '{"summary": "durable fact"}',
                        },
                    }
                ],
            )
        )
        agent.client.chat.side_effect = [
            response,
            ChatResponse(message=Message(role="assistant", content="done")),
            ChatResponse(message=Message(role="assistant", content="done again")),
        ]
        agent.run_turn("hello")
        agent.run_turn("hello")
        self.assertEqual(agent.last_report_run, 1)
        self.assertNotIn("DUE", agent.history[0].content)
        with open(self.path, encoding="utf-8") as fh:
            self.assertIn("- durable fact", fh.read())

    def test_tool_call_via_registry_api(self) -> None:
        agent = make_agent(longterm_path=self.path)
        result = agent.registry.run(
            "report_longterm", '{"summary": "kept the repo tidy"}'
        )
        self.assertIn("recorded long-term memory update for run 0", result)
        with open(self.path, encoding="utf-8") as fh:
            self.assertIn("## Run 0 - ", fh.read())

    def test_empty_summary_tool_call_returns_error(self) -> None:
        agent = make_agent(longterm_path=self.path)
        result = agent.registry.run("report_longterm", '{"summary": "  "}')
        self.assertTrue(result.startswith("Error:"))

    def test_snapshot_included_when_not_due(self) -> None:
        append_longterm(self.path, 5, "snapshot fact")
        agent = make_agent(longterm_path=self.path)
        agent.runs = 6
        agent.last_report_run = 5
        self.turn(agent)
        self.assertIn("snapshot fact", agent.history[0].content)

    def test_runs_persist_in_memory_file(self) -> None:
        agent = make_agent(longterm_path=self.path)
        self.turn(agent)
        self.turn(agent)
        save_history(agent, self.memory)
        fresh = make_agent(longterm_path=self.path)
        load_history(fresh, self.memory)
        self.assertEqual(fresh.runs, 2)
        self.assertEqual(fresh.last_report_run, agent.last_report_run)

    def test_legacy_memory_file_loads(self) -> None:
        with open(self.memory, "w", encoding="utf-8") as fh:
            fh.write(
                '[{"role": "user", "content": "hi"},'
                ' {"role": "assistant", "content": "hey"}]'
            )
        agent = make_agent(longterm_path=self.path)
        load_history(agent, self.memory)
        self.assertEqual(agent.runs, 0)
        self.assertEqual(agent.history[-1].content, "hey")

    def test_missing_run_counter_defaults(self) -> None:
        agent = make_agent(longterm_path=self.path)
        self.assertEqual(agent.runs, 0)
        self.assertEqual(agent.last_report_run, 0)

    def test_line_cap_truncation_in_section(self) -> None:
        from foxnaught.longterm import LONGTERM_LINE_CAP

        append_longterm(self.path, 1, "long " * 500)
        with open(self.path, encoding="utf-8") as fh:
            content = fh.read()
        line = [l for l in content.splitlines() if l.startswith("- ")][0]
        self.assertLessEqual(len(line), LONGTERM_LINE_CAP + 4)


class DateHeaderTest(unittest.TestCase):
    def test_section_carries_utc_date(self) -> None:
        dir = tempfile.TemporaryDirectory()
        self.addCleanup(dir.cleanup)
        path = os.path.join(dir.name, "LONGTERM.md")
        append_longterm(path, 7, "fact")
        today = datetime.now(timezone.utc).date().isoformat()
        with open(path, encoding="utf-8") as fh:
            self.assertIn(today, fh.read())


if __name__ == "__main__":
    unittest.main()
