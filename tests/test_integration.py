"""Integration tests: real tools, mocked model client."""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from foxnaught.agent import Agent
from foxnaught.builtins import register_defaults
from foxnaught.client import ChatResponse, Message, MistralClient
from foxnaught.files import register_files
from foxnaught.tools import ToolRegistry


class FilesToolIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base = tempfile.mkdtemp()

    def make_agent(self) -> Agent:
        registry = ToolRegistry()
        register_defaults(registry)
        register_files(registry, self.base)
        return Agent(mock.create_autospec(MistralClient, instance=True), registry)

    def test_write_then_read_files_end_to_end(self) -> None:
        agent = self.make_agent()
        write_call = {
            "id": "c1",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": '{"path": "notes.md", "content": "kept 10"}',
            },
        }
        read_call = {
            "id": "c2",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "notes.md"}'},
        }
        agent.client.chat.side_effect = [
            ChatResponse(message=Message(role="assistant", content=None, tool_calls=[write_call])),
            ChatResponse(message=Message(role="assistant", content=None, tool_calls=[read_call])),
            ChatResponse(message=Message(role="assistant", content="saved 'kept 10'")),
        ]
        reply = agent.run_turn("take a note and read it back")
        self.assertEqual(reply, "saved 'kept 10'")
        tool_messages = [m for m in agent.history if m.role == "tool"]
        self.assertEqual(len(tool_messages), 2)
        self.assertFalse(tool_messages[0].content.startswith("Error:"))
        self.assertEqual(tool_messages[1].content, "kept 10")
        with open(f"{self.base}/notes.md", encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "kept 10")

    def test_missing_file_error_reaches_model(self) -> None:
        agent = self.make_agent()
        read_call = {
            "id": "c1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "nope.txt"}'},
        }
        agent.client.chat.side_effect = [
            ChatResponse(message=Message(role="assistant", content=None, tool_calls=[read_call])),
            ChatResponse(message=Message(role="assistant", content="the file does not exist yet")),
        ]
        reply = agent.run_turn("read the file")
        self.assertEqual(reply, "the file does not exist yet")
        tool_message = next(m for m in agent.history if m.role == "tool")
        self.assertTrue(tool_message.content.startswith("Error:"))


if __name__ == "__main__":
    unittest.main()
