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
from foxnaught.server import MAX_BODY_BYTES, AgentState, make_handler, make_server
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


class SlowClientTest(unittest.TestCase):
    def start_fast_timeout_server(self) -> tuple[str, int]:
        handler = make_handler(AgentState(make_agent()))

        class FastTimeout(handler):
            timeout = 1

        server = ThreadingHTTPServer(("127.0.0.1", 0), FastTimeout)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return "127.0.0.1", server.server_address[1]

    def test_stalled_request_line_closes_connection(self) -> None:
        import socket

        host, port = self.start_fast_timeout_server()
        with socket.create_connection((host, port), timeout=5) as sock:
            sock.sendall(b"POST /chat HTTP/1.1\r\nContent-Type: application/json")
            sock.settimeout(5)
            self.assertEqual(sock.recv(1024), b"")

    def test_slow_body_returns_400(self) -> None:
        import socket

        host, port = self.start_fast_timeout_server()
        with socket.create_connection((host, port), timeout=5) as sock:
            sock.sendall(b"POST /chat HTTP/1.1\r\nContent-Length: 4\r\n\r\n")
            sock.settimeout(5)
            response = b""
            try:
                while True:
                    chunk = sock.recv(2048)
                    if not chunk:
                        break
                    response += chunk
            except socket.timeout:
                self.fail("server did not answer a stalled body read with a 400")
        response = response.decode()
        self.assertIn("400", response)
        self.assertIn("timed out reading request body", response)


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

    def test_chat_rejects_missing_body(self) -> None:
        status, payload = self.request(self.base, "POST", "/chat", raw=b"")
        self.assertEqual(status, 400)
        self.assertIn("request body", payload["error"])

    def test_chat_rejects_oversized_body(self) -> None:
        status, payload = self.request(self.base, "POST", "/chat", raw=b"x" * (MAX_BODY_BYTES + 1))
        self.assertEqual(status, 400)
        self.assertIn("too large", payload["error"])


class AuthTest(unittest.TestCase):
    def start_protected(self) -> tuple[str, ThreadingHTTPServer]:
        server = make_server(
            make_agent(), host="127.0.0.1", port=0, auth_token="s3cret"
        )
        base = f"http://127.0.0.1:{server.server_address[1]}"
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return base, server

    def chat(self, base: str, token: str | None):
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            base + "/chat",
            data=json.dumps({"message": "hi"}).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_missing_token_rejected(self) -> None:
        base, _ = self.start_protected()
        status, payload = self.chat(base, None)
        self.assertEqual(status, 401)
        self.assertIn("bearer", payload["error"])

    def test_wrong_token_rejected(self) -> None:
        base, _ = self.start_protected()
        status, _ = self.chat(base, "wrong")
        self.assertEqual(status, 401)

    def test_correct_token_accepted(self) -> None:
        base, _ = self.start_protected()
        status, payload = self.chat(base, "s3cret")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"reply": "ok"})

    def test_health_stays_open(self) -> None:
        base, _ = self.start_protected()
        with urllib.request.urlopen(base + "/health", timeout=5) as response:
            self.assertEqual(response.status, 200)

    def test_open_when_token_not_configured(self) -> None:
        server = make_server(make_agent(), host="127.0.0.1", port=0, auth_token=None)
        base = f"http://127.0.0.1:{server.server_address[1]}"
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        status, _ = self.chat(base, None)
        self.assertEqual(status, 200)


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

    def chat(self, base: str, payload: dict):
        request = urllib.request.Request(
            base + "/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_agent_state_ignores_corrupt_memory_file(self) -> None:
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "memory.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not json")

            agent = make_agent()
            state = AgentState(agent, memory_path=path)
            self.assertEqual([m.role for m in agent.history], ["system"])

            self.assertEqual(state.chat("hi"), "ok")
            with open(path, encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertEqual(
                [m["role"] for m in saved["messages"]], ["system", "user", "assistant"]
            )

    def test_agent_state_ignores_empty_memory_file(self) -> None:
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "memory.json")
            with open(path, "w", encoding="utf-8"):
                pass

            agent = make_agent()
            state = AgentState(agent, memory_path=path)
            self.assertEqual([m.role for m in agent.history], ["system"])

            self.assertEqual(state.chat("hi"), "ok")
            with open(path, encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertEqual(
                [m["role"] for m in saved["messages"]], ["system", "user", "assistant"]
            )

    def test_reset_clears_previously_saved_conversation(self) -> None:
        import os
        import tempfile
        import urllib.request

        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(os.unlink, path)

        server = make_server(
            make_agent(responses=["saved", "after-reset"]),
            host="127.0.0.1",
            port=0,
            memory_path=path,
        )
        base = f"http://127.0.0.1:{server.server_address[1]}"
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)

        self.chat(base, {"message": "secret-marker"})
        with open(path, encoding="utf-8") as fh:
            saved = json.load(fh)
        self.assertIn("secret-marker", str(saved))

        status, payload = self.chat(
            base, {"message": "after-reset", "reset": True}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"reply": "after-reset"})
        with open(path, encoding="utf-8") as fh:
            saved = json.load(fh)
        self.assertNotIn("secret-marker", str(saved))
        self.assertEqual(
            [m["role"] for m in saved["messages"]], ["system", "user", "assistant"]
        )
        self.assertEqual(saved["messages"][1]["content"], "after-reset")


if __name__ == "__main__":
    unittest.main()
