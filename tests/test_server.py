"""Tests for the HTTP API server."""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from foxnaught.agent import Agent
from foxnaught.client import ChatResponse, Message, MistralClient, MistralError
from foxnaught.server import AgentState, make_handler, make_server
from foxnaught.tools import ToolRegistry


def make_agent(responses: list[str] | None = None) -> Agent:
    if responses is None:
        responses = ["ok"]
    registry = ToolRegistry()
    client = mock.create_autospec(MistralClient, instance=True)
    client.chat.side_effect = [
        ChatResponse(message=Message(role="assistant", content=reply))
        for reply in responses
    ]
    return Agent(client, registry)


class ServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.servers: list[ThreadingHTTPServer] = []
        self.base = self.start_server(make_server(make_agent()))

    def start_server(self, server: ThreadingHTTPServer) -> str:
        self.servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{server.server_address[1]}"

    def tearDown(self) -> None:
        for server in self.servers:
            server.shutdown()
            server.server_close()

    def request(self, base: str, method: str, path: str, payload: dict | None = None, raw: bytes | None = None):
        data = raw if raw is not None else (
            json.dumps(payload).encode() if payload is not None else None
        )
        request = urllib.request.Request(
            base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_health(self) -> None:
        status, payload = self.request(self.base, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok"})

    def test_get_unknown_path_404(self) -> None:
        status, _ = self.request(self.base, "GET", "/nope")
        self.assertEqual(status, 404)

    def test_chat_success(self) -> None:
        status, payload = self.request(self.base, "POST", "/chat", {"message": "hi"})
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"reply": "ok"})

    def test_chat_surfaces_agent_errors_as_502(self) -> None:
        class FailingState(AgentState):
            def chat(self, message: str, reset: bool = False) -> str:
                raise MistralError("boom")

        failing = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(FailingState(Agent(mock.create_autospec(MistralClient, instance=True), ToolRegistry()))),
        )
        base = self.start_server(failing)
        status, payload = self.request(base, "POST", "/chat", {"message": "hi"})
        self.assertEqual(status, 502)
        self.assertEqual(payload, {"error": "boom"})

    def test_chat_requires_non_empty_message(self) -> None:
        status, payload = self.request(self.base, "POST", "/chat", {"message": "  "})
        self.assertEqual(status, 400)
        self.assertIn("message", payload["error"])

    def test_chat_rejects_invalid_json(self) -> None:
        status, payload = self.request(self.base, "POST", "/chat", raw=b"{oops")
        self.assertEqual(status, 400)
        self.assertIn("invalid JSON", payload["error"])

    def test_chat_rejects_non_object_body(self) -> None:
        status, _ = self.request(self.base, "POST", "/chat", raw=b"[1,2]")
        self.assertEqual(status, 400)

    def test_post_unknown_path_404(self) -> None:
        status, _ = self.request(self.base, "POST", "/nope", {"message": "hi"})
        self.assertEqual(status, 404)


class AgentStateTest(unittest.TestCase):
    def test_reset_then_chat(self) -> None:
        agent = make_agent(responses=["one", "two"])
        state = AgentState(agent)
        self.assertEqual(state.chat("first"), "one")
        self.assertEqual(state.chat("second", reset=True), "two")
        self.assertEqual([m.role for m in agent.history], ["system", "user", "assistant"])


if __name__ == "__main__":
    unittest.main()
