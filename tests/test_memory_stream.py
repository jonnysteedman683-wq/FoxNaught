"""Tests for streaming agent turns and memory persistence."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from foxnaught.agent import Agent, load_history, save_history
from foxnaught.builtins import register_defaults
from foxnaught.client import Message, MistralClient, StreamEvent
from foxnaught.tools import ToolRegistry


def make_agent() -> Agent:
    registry = ToolRegistry()
    register_defaults(registry)
    client = mock.create_autospec(MistralClient, instance=True)
    return Agent(client, registry)


def events_from_chunks(chunks: list[dict]) -> list[StreamEvent]:
    return [StreamEvent(delta=chunk.get("delta") or {}, finish_reason=chunk.get("finish_reason")) for chunk in chunks]


class HistoryTrimTest(unittest.TestCase):
    def fill(self, agent: Agent, turns: int) -> None:
        from foxnaught.client import ChatResponse

        side = [
            ChatResponse(message=Message(role="assistant", content=f"reply {i}"))
            for i in range(turns)
        ]
        agent.client.chat.side_effect = side
        for i in range(turns):
            agent.run_turn(f"msg {i}")

    def test_history_is_trimmed_to_limit(self) -> None:
        agent = make_agent()
        agent.history_limit = 6
        self.fill(agent, 10)
        self.assertEqual(len(agent.history), 6)
        self.assertEqual(agent.history[0].role, "system")
        self.assertEqual(agent.history[-1].content, "reply 9")

    def test_limit_can_be_disabled(self) -> None:
        agent = make_agent()
        agent.history_limit = None
        agent.history_char_budget = None
        self.fill(agent, 6)
        self.assertEqual(len(agent.history), 13)

    def test_char_budget_trims_longest_context(self) -> None:
        from foxnaught.client import ChatResponse

        agent = make_agent()
        agent.history_limit = None
        agent.history_char_budget = 200
        side = [
            ChatResponse(message=Message(role="assistant", content="x" * 90))
            for _ in range(5)
        ]
        agent.client.chat.side_effect = side
        for i in range(5):
            agent.run_turn(f"question {i} " + "y" * 40)
        self.assertEqual(len(agent.history), 2)
        self.assertEqual(agent.history[0].role, "system")
        self.assertEqual([m.role for m in agent.history[1:]][0], "assistant")
        self.assertFalse(any(m.role == "tool" for m in agent.history[1:]))
        self.assertLess(agent._history_chars(), 50_000)

    def test_char_budget_can_be_disabled(self) -> None:
        from foxnaught.client import ChatResponse

        agent = make_agent()
        agent.history_limit = None
        agent.history_char_budget = None
        side = [
            ChatResponse(message=Message(role="assistant", content="x" * 900))
            for _ in range(3)
        ]
        agent.client.chat.side_effect = side
        for i in range(3):
            agent.run_turn(f"q{i}")
        self.assertEqual(len(agent.history), 7)

    def test_dropping_tool_pair_keeps_history_valid(self) -> None:
        agent = make_agent()
        agent.history_limit = 4
        agent.history.append(Message(role="user", content="q1"))
        agent.history.append(
            Message(
                role="assistant",
                content=None,
                tool_calls=[{"id": "c1", "type": "function", "function": {"name": "now_utc", "arguments": None}}],
            )
        )
        agent.history.append(Message(role="tool", name="now_utc", tool_call_id="c1", content="ok"))
        agent.history.append(Message(role="assistant", content="r1"))
        agent.history.append(Message(role="user", content="q2"))
        agent.history.append(Message(role="assistant", content="r2"))
        agent._trim_history()
        self.assertEqual([m.role for m in agent.history], ["system", "assistant", "user", "assistant"])
        self.assertNotIn("tool", [m.role for m in agent.history])


class StreamingAgentTest(unittest.TestCase):
    def test_streams_plain_reply_to_on_delta(self) -> None:
        agent = make_agent()
        agent.client.chat_stream.return_value = events_from_chunks(
            [
                {"delta": {"content": "Hel"}, "finish_reason": None},
                {"delta": {"content": "lo"}, "finish_reason": "stop"},
            ]
        )
        seen: list[str] = []
        reply = agent.run_turn_stream("hi", on_delta=seen.append)
        self.assertEqual(reply, "Hello")
        self.assertEqual(seen, ["Hel", "lo"])
        self.assertEqual([m.role for m in agent.history], ["system", "user", "assistant"])

    def test_accumulates_fragmented_tool_arguments(self) -> None:
        agent = make_agent()
        agent.client.chat_stream.side_effect = [
            events_from_chunks(
                [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "calculate", "arguments": '{"expres'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    },
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": 'sion": "2+3"}'}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    },
                ]
            ),
            events_from_chunks(
                [{"delta": {"content": "It is 5."}, "finish_reason": "stop"}]
            ),
        ]
        seen: list[str] = []
        reply = agent.run_turn_stream("what is two plus three", on_delta=seen.append)
        self.assertEqual(reply, "It is 5.")
        self.assertEqual(seen, ["It is 5."])
        self.assertEqual(agent.history[3].role, "tool")
        self.assertEqual(agent.history[3].content, "5")

    def test_missing_on_delta_still_works(self) -> None:
        agent = make_agent()
        agent.client.chat_stream.return_value = events_from_chunks(
            [{"delta": {"content": "hi"}, "finish_reason": "stop"}]
        )
        self.assertEqual(agent.run_turn_stream("hello"), "hi")

    def test_stream_continues_after_unknown_tool(self) -> None:
        agent = make_agent()
        agent.client.chat_stream.side_effect = [
            events_from_chunks(
                [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_x",
                                    "type": "function",
                                    "function": {"name": "no_such_tool", "arguments": "{}"},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            ),
            events_from_chunks(
                [
                    {
                        "delta": {"content": "could not use that tool"},
                        "finish_reason": "stop",
                    }
                ]
            ),
        ]
        seen: list[str] = []
        reply = agent.run_turn_stream("hi", on_delta=seen.append)
        self.assertEqual(reply, "could not use that tool")
        self.assertEqual(seen, ["could not use that tool"])
        tool_message = agent.history[3]
        self.assertEqual(tool_message.role, "tool")
        self.assertTrue(tool_message.content.startswith("Error: unknown tool"))

    def test_stream_continues_after_tool_error_string(self) -> None:
        agent = make_agent()
        agent.client.chat_stream.side_effect = [
            events_from_chunks(
                [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "calculate", "arguments": '{"expression": "1/0"}'},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            ),
            events_from_chunks(
                [{"delta": {"content": "that did not work"}, "finish_reason": "stop"}]
            ),
        ]
        seen: list[str] = []
        reply = agent.run_turn_stream("div", on_delta=seen.append)
        self.assertEqual(reply, "that did not work")
        self.assertEqual(seen, ["that did not work"])
        self.assertEqual(agent.history[3].role, "tool")
        self.assertTrue(agent.history[3].content.startswith("Error:"))

    def test_stream_exceeding_max_steps_raises(self) -> None:
        agent = make_agent()
        agent.max_steps = 2
        agent.client.chat_stream.return_value = events_from_chunks(
            [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_loop",
                                "type": "function",
                                "function": {"name": "calculate", "arguments": '{"expression": "1+1"}'},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "exceeded 2 tool steps"):
            agent.run_turn_stream("loop")

    def test_stream_propagates_mistral_error(self) -> None:
        from foxnaught.client import MistralError

        agent = make_agent()

        def failing_stream(messages, tools=None):
            raise MistralError("boom")
            yield

        agent.client.chat_stream.side_effect = failing_stream
        with self.assertRaisesRegex(MistralError, "boom"):
            agent.run_turn_stream("hi")

    def test_stream_failure_rolls_back_history(self) -> None:
        from foxnaught.client import MistralError

        agent = make_agent()

        def failing_stream(messages, tools=None):
            raise MistralError("boom")
            yield

        agent.client.chat_stream.side_effect = failing_stream
        with self.assertRaises(MistralError):
            agent.run_turn_stream("hi")
        self.assertEqual([m.role for m in agent.history], ["system"])

    def test_stream_failure_mid_tool_turn_rolls_back_partial_history(self) -> None:
        from foxnaught.client import MistralError

        agent = make_agent()
        agent.client.chat_stream.side_effect = [
            events_from_chunks(
                [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "now_utc", "arguments": None},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            ),
            MistralError("boom"),
        ]
        with self.assertRaises(MistralError):
            agent.run_turn_stream("hi")
        self.assertEqual([m.role for m in agent.history], ["system"])


class MemoryTest(unittest.TestCase):
    def test_save_and_load_round_trip(self) -> None:
        agent = make_agent()
        agent.history.append(Message(role="user", content="remember this"))
        agent.history.append(Message(role="assistant", content="cached"))
        save_history(agent, "/tmp/foxnaught-test-memory.json")
        fresh = make_agent()
        self.assertNotIn("remember this", str(fresh.history))
        load_history(fresh, "/tmp/foxnaught-test-memory.json")
        self.assertEqual(fresh.history[-1].content, "cached")
        self.assertEqual(fresh.history[0].role, "system")

    def test_missing_file_is_ignored_by_cli_helper_but_raises_directly(self) -> None:
        agent = make_agent()
        with self.assertRaises(FileNotFoundError):
            load_history(agent, "/tmp/does-not-exist-foxnaught.json")

    def test_invalid_history_file_raises_value_error(self) -> None:
        agent = make_agent()
        with open("/tmp/foxnaught-bad-history.json", "w") as fh:
            fh.write('{"not": "a list"}')
        with self.assertRaises(ValueError):
            load_history(agent, "/tmp/foxnaught-bad-history.json")

    def test_history_without_system_prompt_gets_one(self) -> None:
        agent = make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "no-system.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump([{"role": "user", "content": "hi"}], fh)
            load_history(agent, path)
        self.assertEqual(agent.history[0].role, "system")
        self.assertEqual(agent.history[1].content, "hi")

    def test_load_history_rejects_empty_file(self) -> None:
        agent = make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.json")
            with open(path, "w", encoding="utf-8"):
                pass
            with self.assertRaises(ValueError):
                load_history(agent, path)

    def test_load_history_rejects_empty_list(self) -> None:
        agent = make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty-list.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump([], fh)
            with self.assertRaisesRegex(ValueError, "must contain a non-empty list"):
                load_history(agent, path)

    def test_load_history_rejects_message_without_role(self) -> None:
        agent = make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "no-role.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"messages": [{"content": "hi"}]}, fh)
            with self.assertRaisesRegex(ValueError, "invalid message entry"):
                load_history(agent, path)

    def test_load_history_trims_oversized_history(self) -> None:
        agent = make_agent()
        for i in range(30):
            agent.history.append(Message(role="user", content=f"u{i}"))
            agent.history.append(Message(role="assistant", content=f"a{i}"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "long.json")
            save_history(agent, path)
            fresh = make_agent()
            fresh.history_limit = 6
            load_history(fresh, path)
        self.assertEqual(len(fresh.history), 6)
        self.assertEqual(fresh.history[0].role, "system")
        self.assertEqual(fresh.history[-1].role, "assistant")
        self.assertEqual(fresh.history[-1].content, "a29")

    def test_save_history_is_atomic(self) -> None:
        agent = make_agent()
        agent.history.append(Message(role="user", content="hi"))
        agent.history.append(Message(role="assistant", content="ok"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "memory.json")
            save_history(agent, path)
            with open(path, encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertFalse(os.path.exists(path + ".tmp"))
        self.assertIn("runs", saved)
        self.assertIn("last_report_run", saved)
        self.assertIsInstance(saved["messages"], list)
        self.assertTrue(saved["messages"])


if __name__ == "__main__":
    unittest.main()
