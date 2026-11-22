"""Tool registry for the agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def to_schema(self) -> dict[str, Any]:
        """Return the function-calling schema expected by the Mistral API."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Holds the available tools and executes calls against them."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., Any],
    ) -> None:
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = Tool(
            name=name, description=description, parameters=parameters, func=func
        )

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    def run(self, name: str, arguments: str | None) -> str:
        """Execute a tool call given a JSON-encoded arguments string."""
        tool = self.get(name)
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            return f"Error: invalid JSON arguments for {name}: {exc}"
        if not isinstance(args, dict):
            return f"Error: arguments for {name} must be a JSON object"
        try:
            result = tool.func(**args)
        except Exception as exc:  # noqa: BLE001 - surface tool errors to the model
            return f"Error: {type(exc).__name__}: {exc}"
        if not isinstance(result, str):
            result = json.dumps(result)
        return result
