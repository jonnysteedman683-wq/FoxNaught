"""HTTP API for the FoxNaught agent (stdlib only)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .agent import Agent
from .client import MistralError

MAX_BODY_BYTES = 64_000


class AgentState:
    """A shared agent serialised by a lock, since history is not thread-safe."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.lock = threading.Lock()

    def chat(self, message: str, reset: bool = False) -> str:
        with self.lock:
            if reset:
                self.agent.reset()
            return self.agent.run_turn(message)


def make_handler(state: AgentState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "FoxNaught/0.1"

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/chat":
                self._send_json(404, {"error": "not found"})
                return
            try:
                request = self._read_json()
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            message = request.get("message")
            if not isinstance(message, str) or not message.strip():
                self._send_json(400, {"error": "'message' must be a non-empty string"})
                return
            reset = bool(request.get("reset", False))
            try:
                reply = state.chat(message, reset=reset)
            except (MistralError, RuntimeError) as exc:
                self._send_json(502, {"error": str(exc)})
                return
            self._send_json(200, {"reply": reply})

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                raise ValueError("request body is required")
            if length > MAX_BODY_BYTES:
                raise ValueError("request body too large")
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(f"invalid JSON body: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def make_server(agent: Agent, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(AgentState(agent)))
    server.daemon_threads = True
    return server
