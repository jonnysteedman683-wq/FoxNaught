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
from foxnaught.client import ChatResponse, Message, MistralClient, MistralError, StreamEvent
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


class StreamingEndpointTest(unittest.TestCase):
    def test_streaming_chat_returns_sse_events(self) -> None:
        registry = ToolRegistry()
        client = mock.create_autospec(MistralClient, instance=True)
        agent = Agent(client, registry)

        def fake_stream(messages, tools=None):
            yield StreamEvent(delta={"content": "Hel"})
            yield StreamEvent(delta={"content": "lo"}, finish_reason="stop")

        client.chat_stream.side_effect = fake_stream
        server = make_server(agent, host="127.0.0.1", port=0)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)

        request = urllib.request.Request(
            base + "/chat",
            data=json.dumps({"message": "hi", "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertTrue(response.headers["Content-Type"].startswith("text/event-stream"))
            body = response.read().decode()
        events = [
            json.loads(line[5:])
            for line in body.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        self.assertIn(b"data: [DONE]", body.encode())
        self.assertEqual([e.get("delta") for e in events if "delta" in e], ["Hel", "lo"])
        final = [e for e in events if "reply" in e]
        self.assertEqual(final, [{"reply": "Hello"}])

    def test_streaming_error_surfaces_as_sse_event(self) -> None:
        from foxnaught.client import MistralError

        registry = ToolRegistry()
        client = mock.create_autospec(MistralClient, instance=True)

        def failing_stream(messages, tools=None):
            raise MistralError("boom")
            yield

        client.chat_stream.side_effect = failing_stream
        agent = Agent(client, registry)
        server = make_server(agent, host="127.0.0.1", port=0)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)

        request = urllib.request.Request(
            base + "/chat",
            data=json.dumps({"message": "hi", "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read().decode()
        self.assertIn('"error": "boom"', body)


class ServerMemoryTest(unittest.TestCase):
    def test_memory_persists_across_requests(self) -> None:
        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(os.unlink, path)

        server = make_server(make_agent(), host="127.0.0.1", port=0, memory_path=path)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            request = urllib.request.Request(
                base + "/chat",
                data=json.dumps({"message": "hi"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(request, timeout=5).read()
            with open(path, encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertIn("hi", str(saved))

            reloaded_agent = make_agent()
            AgentState(reloaded_agent, memory_path=path)
            self.assertIn("hi", str(reloaded_agent.history))
        finally:
            server.shutdown()
            server.server_close()

    def test_reset_wipes_memory_file(self) -> None:
        import os
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(os.unlink, path)

        server = make_server(make_agent(), host="127.0.0.1", port=0, memory_path=path)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)

        request = urllib.request.Request(
            base + "/chat",
            data=json.dumps({"message": "hi", "reset": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(request, timeout=5).read()
        with open(path, encoding="utf-8") as fh:
            saved = json.load(fh)
        self.assertEqual([m["role"] for m in saved], ["system", "user", "assistant"])


if __name__ == "__main__":
    unittest.main()
