"""Thin client for the Mistral chat-completions API (stdlib only)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
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


class MistralError(RuntimeError):
    """Raised when the API call fails."""


class MistralClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise MistralError(
                "no API key: pass api_key or set the MISTRAL_API_KEY environment variable"
            )
        self.model = model or os.environ.get("MISTRAL_MODEL", DEFAULT_MODEL)
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_dict() for message in messages],
        }
        if tools:
            payload["tools"] = tools

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise MistralError(f"HTTP {exc.code} from Mistral API: {detail}") from exc
        except urllib.error.URLError as exc:
            raise MistralError(f"could not reach the Mistral API: {exc.reason}") from exc

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
