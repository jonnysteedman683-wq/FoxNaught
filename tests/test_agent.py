"""Tests for the FoxNaught agent."""

from __future__ import annotations

import unittest
from unittest import mock

from foxnaught.agent import Agent
from foxnaught.builtins import calculate, now_utc, register_defaults
from foxnaught.client import ChatResponse, Message, MistralClient
from foxnaught.tools import ToolRegistry


class ToolRegistryTest(unittest.TestCase):
    def test_run_invokes_registered_function(self) -> None:
        registry = ToolRegistry()
        registry.register(
            "echo",
            "Echo text back.",
            {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            echo,
        )
        self.assertEqual(registry.run("echo", '{"text": "hi"}'), "hi")

    def test_run_returns_error_for_bad_json(self) -> None:
        registry = ToolRegistry()
        registry.register(
            "echo", "Echo text back.", {"type": "object", "properties": {}}, echo
        )
        self.assertTrue(registry.run("echo", "{oops").startswith("Error:"))

    def test_run_surfaces_tool_exceptions(self) -> None:
        registry = ToolRegistry()
        registry.register("boom", "Always fails.", {"type": "object", "properties": {}}, boom)
        self.assertTrue(registry.run("boom", None).startswith("Error:"))

    def test_run_rejects_unknown_tool(self) -> None:
        registry = ToolRegistry()
        with self.assertRaises(KeyError):
            registry.run("missing", None)

    def test_schemas_wrap_tools_as_functions(self) -> None:
        registry = ToolRegistry()
        registry.register(
            "echo", "Echo text back.", {"type": "object", "properties": {}}, echo
        )
        schema = registry.schemas()[0]
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "echo")


def echo(text: str = "") -> str:
    return text


def boom() -> str:
    raise ValueError("kaboom")


class BuiltinsTest(unittest.TestCase):
    def test_calculate_arithmetic(self) -> None:
        self.assertEqual(calculate("1 + 2 * 3"), "7")

    def test_calculate_functions_and_constants(self) -> None:
        self.assertEqual(calculate("sqrt(4)"), "2.0")
        self.assertEqual(calculate("round(pi, 2)"), "3.14")

    def test_calculate_rejects_names(self) -> None:
        self.assertTrue(calculate("__import__('os')").startswith("Error:"))

    def test_calculate_rejects_syntax_error(self) -> None:
        self.assertTrue(calculate("1 +").startswith("Error:"))

    def test_now_utc_supported_timezone(self) -> None:
        self.assertNowValid(now_utc())

    def test_now_utc_rejects_other_timezones(self) -> None:
        self.assertTrue(now_utc("EST").startswith("Error:"))

    def assertNowValid(self, value: str) -> None:
        self.assertIn("T", value)
        self.assertTrue(value.endswith("+00:00"))


class AgentTest(unittest.TestCase):
    def make_agent(self, responses: list[ChatResponse]) -> Agent:
        registry = ToolRegistry()
        register_defaults(registry)
        client = mock.create_autospec(MistralClient, instance=True)
        client.chat.side_effect = responses
        return Agent(client, registry)

    def test_plain_reply_without_tools(self) -> None:
        responses = [
            ChatResponse(message=Message(role="assistant", content="hello"))
        ]
        agent = self.make_agent(responses)
        self.assertEqual(agent.run_turn("hi"), "hello")
        self.assertEqual([m.role for m in agent.history], ["system", "user", "assistant"])

    def test_tool_call_then_reply(self) -> None:
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "calculate", "arguments": '{"expression": "6 * 7"}'},
        }
        responses = [
            ChatResponse(
                message=Message(role="assistant", content=None, tool_calls=[tool_call])
            ),
            ChatResponse(message=Message(role="assistant", content="42")),
        ]
        agent = self.make_agent(responses)
        self.assertEqual(agent.run_turn("what is six times seven"), "42")
        self.assertEqual(agent.history[2].tool_calls, [tool_call])
        tool_message = agent.history[3]
        self.assertEqual(tool_message.role, "tool")
        self.assertEqual(tool_message.name, "calculate")
        self.assertEqual(tool_message.tool_call_id, "call_1")
        self.assertEqual(tool_message.content, "42")

    def test_unknown_tool_returns_error_message(self) -> None:
        tool_call = {
            "id": "call_2",
            "type": "function",
            "function": {"name": "no_such_tool", "arguments": None},
        }
        responses = [
            ChatResponse(
                message=Message(role="assistant", content=None, tool_calls=[tool_call])
            ),
            ChatResponse(message=Message(role="assistant", content="done")),
        ]
        agent = self.make_agent(responses)
        self.assertEqual(agent.run_turn("use it"), "done")
        self.assertEqual(agent.history[3].content, "Error: unknown tool: no_such_tool")

    def test_exceeding_max_steps_raises(self) -> None:
        def looping_response() -> ChatResponse:
            return ChatResponse(
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_x",
                            "type": "function",
                            "function": {"name": "now_utc", "arguments": None},
                        }
                    ],
                )
            )

        responses = [looping_response() for _ in range(3)]
        agent = self.make_agent(responses)
        agent.max_steps = 3
        with self.assertRaises(RuntimeError):
            agent.run_turn("loop")

    def test_reset_restores_system_prompt_only(self) -> None:
        responses = [ChatResponse(message=Message(role="assistant", content="ok"))]
        agent = self.make_agent(responses)
        agent.run_turn("hi")
        agent.reset()
        self.assertEqual(len(agent.history), 1)
        self.assertEqual(agent.history[0].role, "system")


if __name__ == "__main__":
    unittest.main()
