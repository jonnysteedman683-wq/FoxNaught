"""HTTP API for the FoxNaught agent (stdlib only)."""

from __future__ import annotations

import hmac
import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .agent import Agent, load_history, save_history
from .client import MistralError

MAX_BODY_BYTES = 64_000
READ_TIMEOUT_SECONDS = 30
AUTH_ENV_VAR = "FOXNAUGHT_AUTH_TOKEN"


class AgentState:
    """A shared agent serialised by a lock, since history is not thread-safe."""

    def __init__(self, agent: Agent, memory_path: str | None = None) -> None:
        self.agent = agent
        self.lock = threading.Lock()
        self.memory_path = memory_path
        if memory_path:
            self._load_memory()

    def _load_memory(self) -> None:
        try:
            load_history(self.agent, self.memory_path)
        except (FileNotFoundError, ValueError):
            pass

    def _save_memory(self) -> None:
        if self.memory_path:
            save_history(self.agent, self.memory_path)

    def chat(
        self,
        message: str,
        reset: bool = False,
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        with self.lock:
            if reset:
                self.agent.reset()
                self._save_memory()
            if on_delta is not None:
                reply = self.agent.run_turn_stream(message, on_delta=on_delta)
            else:
                reply = self.agent.run_turn(message)
            self._save_memory()
            return reply


def make_handler(state: AgentState, auth_token: str | None = None) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "FoxNaught/0.1"
        timeout = READ_TIMEOUT_SECONDS

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/chat":
                self._send_json(404, {"error": "not found"})
                return
            if auth_token is not None and not self._authorized(auth_token):
                self._send_json(401, {"error": "unauthorized: missing or invalid bearer token"})
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
            if request.get("stream", False):
                self._handle_stream_chat(message, reset)
                return
            try:
                reply = state.chat(message, reset=reset)
            except (MistralError, RuntimeError) as exc:
                self._send_json(502, {"error": str(exc)})
                return
            self._send_json(200, {"reply": reply})

        def _handle_stream_chat(self, message: str, reset: bool) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                reply = state.chat(
                    message,
                    reset=reset,
                    on_delta=lambda piece: self._send_sse({"delta": piece}),
                )
                self._send_sse({"reply": reply})
            except (MistralError, RuntimeError, BrokenPipeError) as exc:
                self._send_sse({"error": str(exc)})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _authorized(self, expected: str) -> bool:
            header = self.headers.get("Authorization") or ""
            if not header.startswith("Bearer "):
                return False
            return hmac.compare_digest(header[7:], expected)

        def _send_sse(self, payload: dict) -> None:
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
            self.wfile.flush()

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                raise ValueError("request body is required")
            if length > MAX_BODY_BYTES:
                raise ValueError("request body too large")
            try:
                body = self.rfile.read(length)
            except TimeoutError as exc:
                raise ValueError("timed out reading request body") from exc
            try:
                data = json.loads(body.decode("utf-8"))
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


def make_server(
    agent: Agent,
    host: str = "127.0.0.1",
    port: int = 0,
    memory_path: str | None = None,
    auth_token: str | None = None,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(
        (host, port),
        make_handler(AgentState(agent, memory_path=memory_path), auth_token=auth_token),
    )
    server.daemon_threads = True
    return server
