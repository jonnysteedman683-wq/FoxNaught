"""Command-line interface for the FoxNaught agent."""

from __future__ import annotations

import argparse
import sys

from .agent import DEFAULT_MAX_STEPS, Agent
from .builtins import register_defaults
from .client import MistralClient, MistralError
from .tools import ToolRegistry
from .web import register_web


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_defaults(registry)
    register_web(registry)
    return registry


def build_agent(model: str | None, max_steps: int) -> Agent:
    client = MistralClient(model=model)

    def on_step(name: str, arguments: object) -> None:
        print(f"[tool] {name}({arguments})")

    return Agent(client, build_registry(), max_steps=max_steps, on_step=on_step)


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


def serve(agent: Agent, host: str, port: int) -> None:
    from .server import make_server

    server = make_server(agent, host=host, port=port)
    print(f"FoxNaught agent listening on http://{host}:{server.server_address[1]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="foxnaught", description=__doc__)
    parser.add_argument("--model", default=None, help="Mistral model to use")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="maximum tool steps per turn",
    )
    parser.add_argument(
        "--one-shot",
        metavar="PROMPT",
        default=None,
        help="run a single prompt headlessly and print the reply",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="run an HTTP API server instead of the interactive chat",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address for --serve")
    parser.add_argument("--port", type=int, default=8080, help="port for --serve")
    args = parser.parse_args(argv)

    try:
        agent = build_agent(args.model, args.max_steps)
    except MistralError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.one_shot is not None:
        try:
            print(agent.run_turn(args.one_shot))
        except (MistralError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.serve:
        serve(agent, args.host, args.port)
        return 0
    chat(agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
