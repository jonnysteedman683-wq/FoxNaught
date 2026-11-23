"""Live integration tests, skipped unless MISTRAL_API_KEY is set."""

from __future__ import annotations

import os
import tempfile
import unittest

from foxnaught.agent import Agent, load_history, save_history
from foxnaught.builtins import register_defaults
from foxnaught.client import MistralClient
from foxnaught.tools import ToolRegistry

HAS_KEY = bool(os.environ.get("MISTRAL_API_KEY"))


@unittest.skipUnless(HAS_KEY, "requires a live MISTRAL_API_KEY")
class LiveSmokeTest(unittest.TestCase):
    def make_agent(self) -> Agent:
        registry = ToolRegistry()
        register_defaults(registry)
        client = MistralClient()
        return Agent(client, registry, max_steps=4)

    def test_tool_call_round_trip(self) -> None:
        agent = self.make_agent()
        reply = agent.run_turn(
            "Use the calculate tool to compute 2 + 3, then reply with just the number."
        )
        self.assertIn("5", reply)

    def test_streaming_round_trip(self) -> None:
        agent = self.make_agent()
        chunks = []
        reply = agent.run_turn_stream("Reply with exactly: ok", on_delta=chunks.append)
        self.assertIn("ok", reply.lower())
        self.assertTrue(chunks, "expected at least one streamed delta")

    def test_memory_file_round_trip(self) -> None:
        agent = self.make_agent()
        agent.run_turn("Reply with exactly: remembered")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name
        save_history(agent, path)
        fresh = self.make_agent()
        load_history(fresh, path)
        self.assertTrue(len(fresh.history) >= 3)


if __name__ == "__main__":
    unittest.main()
