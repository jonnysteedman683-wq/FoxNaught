"""Integration tests for end-to-end workflows."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from foxnaught.agent import Agent, Message, save_history, load_history
from foxnaught.client import MistralClient, MistralError
from foxnaught.tools import ToolRegistry


class MemoryIntegrationTest(unittest.TestCase):
    """Test memory persistence and loading across agent runs."""

    def test_memory_round_trip(self) -> None:
        """Test saving and loading memory preserves conversation state."""
        registry = ToolRegistry()
        client = MagicMock(spec=MistralClient)
        client.chat.return_value = ChatResponse(
            Message(role="assistant", content="Hello!"), finish_reason="stop"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            path = fh.name

        try:
            agent = Agent(client, registry, history_limit=10)
            agent.run_turn("Hi there")
            save_history(agent, path)

            new_agent = Agent(client, registry, history_limit=10)
            load_history(new_agent, path)
            self.assertEqual(len(new_agent.history), 3)  # system + user + assistant
            self.assertEqual(new_agent.history[1].content, "Hi there")
            self.assertEqual(new_agent.history[2].content, "Hello!")
        finally:
            os.unlink(path)

    def test_memory_with_tools(self) -> None:
        """Test memory preserves tool calls."""
        registry = ToolRegistry()
        registry.register(
            name="echo",
            description="Echo a message",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            func=lambda text: f"echoed: {text}",
        )

        call_count = [0]
        
        def mock_chat(messages, tools):
            call_count[0] += 1
            if call_count[0] == 1:
                return ChatResponse(
                    Message(
                        role="assistant",
                        content="Calling tool",
                        tool_calls=[{"id": "1", "function": {"name": "echo", "arguments": '{"text": "test"}'}, "type": "function"}],
                    ),
                    finish_reason="tool_calls",
                )
            else:
                return ChatResponse(
                    Message(role="assistant", content="Done"),
                    finish_reason="stop",
                )
        
        client = MagicMock(spec=MistralClient)
        client.chat.side_effect = mock_chat

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            path = fh.name

        try:
            agent = Agent(client, registry, history_limit=10, max_steps=3)
            agent.run_turn("Use echo tool")
            save_history(agent, path)

            new_agent = Agent(client, registry, history_limit=10)
            load_history(new_agent, path)
            self.assertGreaterEqual(len(new_agent.history), 3)
            found_tool_call = False
            for msg in new_agent.history:
                if msg.tool_calls:
                    found_tool_call = True
                    break
            self.assertTrue(found_tool_call)
        finally:
            os.unlink(path)


class ArchiveIntegrationTest(unittest.TestCase):
    """Test memory archive functionality."""

    def test_archive_written_on_trim(self) -> None:
        """Test that archive is written when history is trimmed."""
        registry = ToolRegistry()
        client = MagicMock(spec=MistralClient)
        client.chat.return_value = ChatResponse(
            Message(role="assistant", content="Reply"), finish_reason="stop"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = os.path.join(tmpdir, "archive.md")
            memory_path = os.path.join(tmpdir, "memory.json")

            agent = Agent(
                client, registry,
                history_limit=3,
                memory_archive_path=archive_path,
            )

            for i in range(5):
                agent.run_turn(f"Message {i}")

            save_history(agent, memory_path)

            self.assertTrue(os.path.exists(archive_path))
            with open(archive_path, encoding="utf-8") as fh:
                content = fh.read()
            self.assertIn("user:", content)
            self.assertIn("assistant:", content)


class LongtermMemoryIntegrationTest(unittest.TestCase):
    """Test long-term memory functionality."""

    def test_longterm_memory_prompt_injected(self) -> None:
        """Test that long-term memory is injected into system prompt."""
        registry = ToolRegistry()
        client = MagicMock(spec=MistralClient)
        client.chat.return_value = ChatResponse(
            Message(role="assistant", content="OK"), finish_reason="stop"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as fh:
            fh.write("# Long-term Memory\n\n- Fact: test fact\n")
            longterm_path = fh.name

        try:
            agent = Agent(
                client, registry,
                longterm_path=longterm_path,
                longterm_interval=1,
            )

            agent.run_turn("Hello")

            self.assertIn("Long-term Memory", agent.history[0].content)
            self.assertIn("test fact", agent.history[0].content)
        finally:
            os.unlink(longterm_path)


class SecurityIntegrationTest(unittest.TestCase):
    """Test security features end-to-end."""

    def test_sandboxed_file_access(self) -> None:
        """Test that file tools cannot escape sandbox."""
        from foxnaught.files import make_read_file, SandboxError

        with tempfile.TemporaryDirectory() as tmpdir:
            read_file = make_read_file(tmpdir)

            result = read_file("test.txt")
            self.assertIn("Error", result)

            with open(os.path.join(tmpdir, "test.txt"), "w") as fh:
                fh.write("Hello")

            result = read_file("test.txt")
            self.assertEqual(result, "Hello")

            result = read_file("../etc/passwd")
            self.assertIn("Error", result)

    def test_ssrf_protection(self) -> None:
        """Test SSRF protection in web fetching."""
        from foxnaught.web import fetch_url, DisallowedHostError

        result = fetch_url("http://localhost/test")
        self.assertIn("Error", result)

        result = fetch_url("http://192.168.1.1/test")
        self.assertIn("Error", result)

        result = fetch_url("http://127.0.0.1/test")
        self.assertIn("Error", result)

        result = fetch_url("http://10.0.0.1/test")
        self.assertIn("Error", result)


class PerformanceIntegrationTest(unittest.TestCase):
    """Test performance characteristics."""

    def test_large_history_trimming(self) -> None:
        """Test that large histories are trimmed efficiently."""
        registry = ToolRegistry()
        client = MagicMock(spec=MistralClient)
        client.chat.return_value = ChatResponse(
            Message(role="assistant", content="X" * 1000), finish_reason="stop"
        )

        agent = Agent(client, registry, history_limit=10, history_char_budget=5000)

        for i in range(100):
            agent.run_turn(f"Message {i}")

        self.assertLessEqual(len(agent.history), 11)  # system + 10 messages

    def test_large_message_handling(self) -> None:
        """Test handling of large messages."""
        registry = ToolRegistry()
        client = MagicMock(spec=MistralClient)
        client.chat.return_value = ChatResponse(
            Message(role="assistant", content="Y" * 1000), finish_reason="stop"
        )

        agent = Agent(client, registry, history_char_budget=5000)

        agent.run_turn("A" * 1000)
        agent.run_turn("B")

        self.assertLessEqual(agent._history_chars(), 5000)


@unittest.skipIf(os.environ.get("MISTRAL_API_KEY") is None, "MISTRAL_API_KEY not set")
class LiveIntegrationTest(unittest.TestCase):
    """Live integration tests with real API calls."""

    def test_real_api_roundtrip(self) -> None:
        """Test a real API call."""
        client = MistralClient(model="mistral-tiny-latest")
        messages = [Message(role="user", content="Say 'test' in 3 letters")]
        response = client.chat(messages, tools=None)
        self.assertIsNotNone(response.message.content)
        self.assertIn("test", (response.message.content or "").lower())

    def test_real_api_with_tools(self) -> None:
        """Test API call with tool definitions."""
        client = MistralClient(model="mistral-tiny-latest")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate an arithmetic expression",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ]
        messages = [Message(role="user", content="What is 2+2?")]
        response = client.chat(messages, tools=tools)
        self.assertIsNotNone(response.message)


class ChatResponse:
    """Helper for mock responses."""

    def __init__(self, message: Message, finish_reason: str | None = None, raw: dict | None = None):
        self.message = message
        self.finish_reason = finish_reason
        self.raw = raw or {}


if __name__ == "__main__":
    unittest.main()
