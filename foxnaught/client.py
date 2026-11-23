"""Thin client for the Mistral chat-completions API (stdlib only)."""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BASE_URL = "https://api.mistral.ai/v1"
DEFAULT_MODEL = "mistral-small-latest"


@dataclass
class Message:
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        if self.tool_calls is not None:
            payload["tool_calls"] = self.tool_calls
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass
class ChatResponse:
    message: Message
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamEvent:
    delta: dict[str, Any]
    finish_reason: str | None = None

    @property
    def content(self) -> str | None:
        return self.delta.get("content")

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return self.delta.get("tool_calls") or []


class MistralError(RuntimeError):
    """Raised when the API call fails."""


class MistralClient:
    RETRY_DELAYS = (1.0, 2.0)
    RETRY_JITTER_RANGE = (0.5, 1.5)

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        retries_on_rate_limit: bool = True,
    ) -> None:
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise MistralError(
                "no API key: pass api_key or set the MISTRAL_API_KEY environment variable"
            )
        self.model = model or os.environ.get("MISTRAL_MODEL", DEFAULT_MODEL)
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.retries_on_rate_limit = retries_on_rate_limit

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        request = self._build_request(messages, tools)
        response = self._open_with_retries(request)
        try:
            body = json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
        return self._parse_response(body)

    def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamEvent]:
        """Yield StreamEvent chunks from a server-sent-events completion."""
        request = self._build_request(messages, tools, stream=True)
        response = self._open_with_retries(request)
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise MistralError(f"invalid SSE chunk: {data!r}") from exc
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                yield StreamEvent(
                    delta=choice.get("delta") or {},
                    finish_reason=choice.get("finish_reason"),
                )
        finally:
            response.close()

    def _build_request(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> urllib.request.Request:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
        }
        if tools:
            payload["tools"] = tools
        if stream:
            payload["stream"] = True
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

    def _open_with_retries(self, request: urllib.request.Request) -> Any:
        attempts = 1 + (len(self.RETRY_DELAYS) if self.retries_on_rate_limit else 0)
        last_rate_limit: urllib.error.HTTPError | None = None
        for attempt in range(attempts):
            if attempt:
                time.sleep(self._retry_delay(attempt))
            try:
                return urllib.request.urlopen(request, timeout=120)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 and self.retries_on_rate_limit and attempt < len(self.RETRY_DELAYS):
                    last_rate_limit = exc
                    continue
                raise MistralError(f"HTTP {exc.code} from Mistral API: {detail}") from exc
            except urllib.error.URLError as exc:
                raise MistralError(f"could not reach the Mistral API: {exc.reason}") from exc
        raise MistralError("rate limit retries exhausted", ) from last_rate_limit

    def _retry_delay(self, attempt: int) -> float:
        base = self.RETRY_DELAYS[attempt - 1]
        low, high = self.RETRY_JITTER_RANGE
        return base * random.uniform(low, high)

    @staticmethod
    def _parse_response(body: dict[str, Any]) -> ChatResponse:
        choices = body.get("choices") or []
        if not choices:
            raise MistralError(f"unexpected API response: {body}")
        choice = choices[0]
        raw_message = choice.get("message") or {}
        message = Message(
            role=raw_message.get("role", "assistant"),
            content=raw_message.get("content"),
            tool_calls=raw_message.get("tool_calls"),
        )
        return ChatResponse(
            message=message, finish_reason=choice.get("finish_reason"), raw=body
        )
