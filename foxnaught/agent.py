"""The agent loop: chat, let the model call tools, feed results back."""

from __future__ import annotations

from collections.abc import Callable

from .client import Message, MistralClient
from .tools import ToolRegistry

DEFAULT_MAX_STEPS = 8

SYSTEM_PROMPT = (
    "You are FoxNaught, a helpful assistant. Use the available tools when they "
    "answer the user's request more reliably than guessing, then summarise the "
    "results."
)


class Agent:
    def __init__(
        self,
        client: MistralClient,
        registry: ToolRegistry,
        system_prompt: str = SYSTEM_PROMPT,
        max_steps: int = DEFAULT_MAX_STEPS,
        on_step: Callable[[str, object], None] | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.on_step = on_step
        self.history: list[Message] = [Message(role="system", content=system_prompt)]

    def reset(self) -> None:
        self.history = [Message(role="system", content=self.system_prompt)]

    def run_turn(self, user_input: str) -> str:
        """Process one user message, running any tool calls, and reply."""
        self.history.append(Message(role="user", content=user_input))
        for _ in range(self.max_steps):
            response = self.client.chat(self.history, self.registry.schemas())
            message = response.message
            if not message.tool_calls:
                self.history.append(message)
                return message.content or ""
            self.history.append(message)
            for call in message.tool_calls:
                self._execute_call(call)
        raise RuntimeError(
            f"agent exceeded {self.max_steps} tool steps without producing a reply"
        )

    def _execute_call(self, call: dict) -> None:
        function = call.get("function") or {}
        name = function.get("name", "")
        arguments = function.get("arguments")
        if self.on_step is not None:
            self.on_step(name, arguments)
        try:
            result = self.registry.run(name, arguments)
        except KeyError:
            result = f"Error: unknown tool: {name}"
        self.history.append(
            Message(
                role="tool",
                name=name,
                tool_call_id=call.get("id"),
                content=result,
            )
        )
