"""Command-line interface for the FoxNaught agent."""

from __future__ import annotations

import argparse
import sys

from .agent import DEFAULT_MAX_STEPS, Agent
from .builtins import register_defaults
from .client import MistralClient, MistralError
from .tools import ToolRegistry


def build_agent(model: str | None, max_steps: int) -> Agent:
    client = MistralClient(model=model)
    registry = ToolRegistry()
    register_defaults(registry)

    def on_step(name: str, arguments: object) -> None:
        print(f"[tool] {name}({arguments})")

    return Agent(client, registry, max_steps=max_steps, on_step=on_step)


def chat(agent: Agent) -> None:
    print("FoxNaught agent. Type 'exit' to quit, '/reset' to clear the conversation.")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_input:
            continue
        if user_input == "exit":
            return
        if user_input == "/reset":
            agent.reset()
            print("conversation reset")
            continue
        try:
            print(f"agent> {agent.run_turn(user_input)}")
        except (MistralError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="foxnaught", description=__doc__)
    parser.add_argument("--model", default=None, help="Mistral model to use")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="maximum tool steps per turn",
    )
    args = parser.parse_args(argv)

    try:
        agent = build_agent(args.model, args.max_steps)
    except MistralError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    chat(agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
