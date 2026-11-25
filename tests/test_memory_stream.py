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
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "memory.json")
            save_history(agent, path)
            fresh = make_agent()
            self.assertNotIn("remember this", str(fresh.history))
            load_history(fresh, path)
        self.assertEqual(fresh.history[-1].content, "cached")
        self.assertEqual(fresh.history[0].role, "system")

    def test_missing_file_is_ignored_by_cli_helper_but_raises_directly(self) -> None:
        agent = make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "does-not-exist.json")
            with self.assertRaises(FileNotFoundError):
                load_history(agent, path)

    def test_invalid_history_file_raises_value_error(self) -> None:
        agent = make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad-history.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('{"not": "a list"}')
            with self.assertRaises(ValueError):
                load_history(agent, path)

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


class LoadHistoryErrorTest(unittest.TestCase):
    """Malformed memory files raise ValueError and leave the agent untouched."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "memory.json")
        self.agent = make_agent()
        self.agent.runs = 7
        self.agent.last_report_run = 4
        self.agent.history.append(Message(role="user", content="keep me"))
        self.before = list(self.agent.history)

    def write(self, raw: str | bytes) -> None:
        with open(self.path, "wb") as fh:
            fh.write(raw.encode("utf-8") if isinstance(raw, str) else raw)

    def assert_rejected(self, raw: str | bytes, error: type[Exception] = ValueError, pattern: str | None = None) -> None:
        self.write(raw)
        if pattern is None:
            with self.assertRaises(error):
                load_history(self.agent, self.path)
        else:
            with self.assertRaisesRegex(error, pattern):
                load_history(self.agent, self.path)
        self.assertEqual(self.agent.history, self.before)
        self.assertEqual(self.agent.runs, 7)
        self.assertEqual(self.agent.last_report_run, 4)

    def test_truncated_json(self) -> None:
        self.assert_rejected('{"runs": 1, "messages": [', json.JSONDecodeError)

    def test_non_utf8_bytes(self) -> None:
        self.assert_rejected(b"\xff\xfe\x00garbage", UnicodeDecodeError)

    def test_decode_errors_are_value_errors_for_callers(self) -> None:
        self.assertTrue(issubclass(json.JSONDecodeError, ValueError))
        self.assertTrue(issubclass(UnicodeDecodeError, ValueError))

    def test_top_level_scalars(self) -> None:
        for raw in ("42", '"hello"', "null", "true"):
            with self.subTest(raw=raw):
                self.assert_rejected(raw, pattern="must contain a non-empty list")

    def test_dict_with_bad_messages(self) -> None:
        for raw in ('{"runs": 1}', '{"messages": "oops"}', '{"messages": []}', '{"messages": {"role": "user"}}'):
            with self.subTest(raw=raw):
                self.assert_rejected(raw, pattern="must contain a non-empty list")

    def test_invalid_message_entries(self) -> None:
        cases = {
            "not an object": '["hi"]',
            "missing role": '[{"content": "hi"}]',
            "non-string role": '[{"role": "user"}, {"role": 123}]',
            "unknown role": '[{"role": "wizard", "content": "hi"}]',
            "non-string content": '[{"role": "user", "content": 5}]',
            "non-string name": '[{"role": "tool", "name": ["x"], "content": "ok"}]',
            "non-string tool_call_id": '[{"role": "tool", "tool_call_id": 1, "content": "ok"}]',
            "tool_calls not a list": '[{"role": "assistant", "tool_calls": "now_utc"}]',
            "tool call not an object": '[{"role": "assistant", "tool_calls": ["now_utc"]}]',
        }
        for label, raw in cases.items():
            with self.subTest(label):
                self.assert_rejected(raw, pattern=r"invalid message entry at index \d")

    def test_invalid_counters_raise_value_error_not_type_error(self) -> None:
        message = '[{"role": "user", "content": "hi"}]'
        for key in ("runs", "last_report_run"):
            for value in ("null", '"abc"', '"5"', "[1]", "-1", "true", "1.5"):
                with self.subTest(key=key, value=value):
                    self.assert_rejected(
                        f'{{"{key}": {value}, "messages": {message}}}',
                        pattern=f"invalid '{key}'",
                    )

    def test_bad_entry_after_valid_counters_does_not_apply_counters(self) -> None:
        self.assert_rejected(
            '{"runs": 99, "last_report_run": 98, "messages": [{"role": "user", "content": "ok"}, {"content": "x"}]}',
            pattern="index 1",
        )

    def test_legacy_list_keeps_existing_counters(self) -> None:
        self.write('[{"role": "user", "content": "hi"}]')
        load_history(self.agent, self.path)
        self.assertEqual((self.agent.runs, self.agent.last_report_run), (7, 4))
        self.assertEqual([m.role for m in self.agent.history], ["system", "user"])

    def test_dict_without_counters_resets_them(self) -> None:
        self.write('{"messages": [{"role": "user", "content": "hi"}]}')
        load_history(self.agent, self.path)
        self.assertEqual((self.agent.runs, self.agent.last_report_run), (0, 0))

    def test_valid_tool_messages_load(self) -> None:
        self.write(
            json.dumps(
                {
                    "runs": 2,
                    "last_report_run": 1,
                    "messages": [
                        {"role": "user", "content": "time?"},
                        {
                            "role": "assistant",
                            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "now_utc"}}],
                        },
                        {"role": "tool", "name": "now_utc", "tool_call_id": "c1", "content": "12:00"},
                        {"role": "assistant", "content": "noon"},
                    ],
                }
            )
        )
        load_history(self.agent, self.path)
        self.assertEqual([m.role for m in self.agent.history], ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(self.agent.history[3].tool_call_id, "c1")
        self.assertEqual((self.agent.runs, self.agent.last_report_run), (2, 1))

    def test_directory_path_raises_os_error(self) -> None:
        with self.assertRaises(OSError):
            load_history(self.agent, os.path.dirname(self.path))

    def test_server_state_survives_corrupt_memory(self) -> None:
        from foxnaught.server import AgentState

        for raw in ('{"runs": null, "messages": [{"role": "user", "content": "x"}]}', '[{"role": "user", "content": 5}]', b"\xff"):
            with self.subTest(raw=raw):
                self.write(raw)
                agent = make_agent()
                state = AgentState(agent, self.path)
                self.assertEqual([m.role for m in state.agent.history], ["system"])
                self.assertEqual(state.agent.runs, 0)


if __name__ == "__main__":
    unittest.main()
