"""Command-line interface for the FoxNaught agent."""

from __future__ import annotations

import argparse
import os
import sys

from .actions import register_actions
from .agent import DEFAULT_ARCHIVE_MAX_CHARS, DEFAULT_HISTORY_CHAR_BUDGET, DEFAULT_HISTORY_LIMIT, DEFAULT_MAX_STEPS, SYSTEM_PROMPT, Agent, load_history, save_history
from .builtins import register_defaults
from .client import MistralClient, MistralError
from .files import register_files
from .gh import register_github
from .tools import ToolRegistry
from .web import register_web


def build_registry(base_dir: str) -> ToolRegistry:
    registry = ToolRegistry()
    register_defaults(registry)
    register_web(registry)
    register_files(registry, base_dir)
    register_github(registry)
    register_actions(registry)
    return registry


def build_agent(
    model: str | None,
    max_steps: int,
    base_dir: str,
    history_limit: int,
    history_char_budget: int,
    memory_archive_path: str | None = None,
    memory_archive_max_chars: int | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> Agent:
    client = MistralClient(model=model)

    def on_step(name: str, arguments: object) -> None:
        print(f"[tool] {name}({arguments})")

    return Agent(
        client,
        build_registry(base_dir),
        system_prompt=system_prompt,
        max_steps=max_steps,
        on_step=on_step,
        history_limit=history_limit,
        history_char_budget=history_char_budget,
        memory_archive_path=memory_archive_path,
        memory_archive_max_chars=memory_archive_max_chars,
    )


def chat(agent: Agent, memory: str | None, stream: bool) -> None:
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
            if memory:
                save_history(agent, memory)
            print("conversation reset")
            continue
        try:
            reply = run_turn(agent, user_input, stream=stream)
        except (MistralError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            continue
        print(f"agent> {reply}")


def run_turn(agent: Agent, prompt: str, stream: bool) -> str:
    if stream:
        reply = agent.run_turn_stream(
            prompt,
            on_delta=lambda piece: print(piece, end="", flush=True),
        )
        if agent.history[-1].role == "assistant":
            print()
        return reply
    return agent.run_turn(prompt)


def save_memory(agent: Agent, memory: str | None) -> None:
    if memory:
        save_history(agent, memory)


def load_memory(agent: Agent, memory: str | None) -> None:
    if memory and os.path.exists(memory):
        load_history(agent, memory)


def serve(agent: Agent, host: str, port: int, memory: str | None, auth_token: str | None) -> None:
    from .server import make_server

    server = make_server(agent, host=host, port=port, memory_path=memory, auth_token=auth_token)
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
    parser.add_argument(
        "--memory",
        metavar="FILE",
        default=None,
        help="persist the conversation history to a JSON file across runs",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="stream replies as they are generated",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=DEFAULT_HISTORY_LIMIT,
        help="keep only this many conversation messages before each turn (0 disables)",
    )
    parser.add_argument(
        "--history-char-budget",
        type=int,
        default=DEFAULT_HISTORY_CHAR_BUDGET,
        help="drop oldest messages until the history fits this many characters (0 disables)",
    )
    parser.add_argument(
        "--memory-archive",
        metavar="FILE",
        default=None,
        help=(
            "append trimmed conversation messages to this markdown file so "
            "the agent can read them back with the read_file tool"
        ),
    )
    parser.add_argument(
        "--system-prompt",
        metavar="TEXT",
        default=SYSTEM_PROMPT,
        help="replace the default system prompt (enables distinct agent roles)",
    )
    parser.add_argument(
        "--archive-max-chars",
        type=int,
        default=DEFAULT_ARCHIVE_MAX_CHARS,
        help=(
            "when the memory archive exceeds this many characters, prune the "
            "oldest entries (0 disables pruning)"
        ),
    )
    parser.add_argument(
        "--auth-token",
        default=None,
        help=(
            "require this bearer token to POST /chat (recommended when binding "
            "publicly); falls back to the FOXNAUGHT_AUTH_TOKEN environment variable"
        ),
    )
    args = parser.parse_args(argv)

    try:
        agent = build_agent(
            args.model,
            args.max_steps,
            base_dir=os.getcwd(),
            history_limit=args.history_limit or None,
            history_char_budget=args.history_char_budget or None,
            memory_archive_path=args.memory_archive,
            memory_archive_max_chars=args.archive_max_chars or None,
            system_prompt=args.system_prompt,
        )
    except MistralError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.one_shot is not None:
        try:
            load_memory(agent, args.memory)
            if args.stream:
                run_turn(agent, args.one_shot, stream=True)
            else:
                print(agent.run_turn(args.one_shot))
            save_memory(agent, args.memory)
        except (MistralError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0
    if args.serve:
        serve(
            agent,
            args.host,
            args.port,
            memory=args.memory,
            auth_token=args.auth_token or os.environ.get("FOXNAUGHT_AUTH_TOKEN"),
        )
        return 0
    chat(agent, memory=args.memory, stream=args.stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
