"""Tests for streaming agent turns and memory persistence."""

from __future__ import annotations

import json
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

    def test_trim_drops_leading_tool_orphans(self) -> None:
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

    def test_limit_can_be_disabled(self) -> None:
        agent = make_agent()
        agent.history_limit = None
        self.fill(agent, 6)
        self.assertEqual(len(agent.history), 13)


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
        with open("/tmp/foxnaught-no-system.json", "w") as fh:
            json.dump([{"role": "user", "content": "hi"}], fh)
        load_history(agent, "/tmp/foxnaught-no-system.json")
        self.assertEqual(agent.history[0].role, "system")
        self.assertEqual(agent.history[1].content, "hi")


if __name__ == "__main__":
    unittest.main()
