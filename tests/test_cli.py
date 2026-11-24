"""Tests for the CLI one-shot mode."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from foxnaught.__main__ import load_memory, main
from foxnaught.agent import Agent
from foxnaught.builtins import register_defaults
from foxnaught.client import MistralClient, MistralError
from foxnaught.tools import ToolRegistry


def make_agent() -> Agent:
    registry = ToolRegistry()
    register_defaults(registry)
    client = mock.create_autospec(MistralClient, instance=True)
    return Agent(client, registry)


class MainTest(unittest.TestCase):
    def test_one_shot_prints_reply(self) -> None:
        with (
            mock.patch("sys.argv", ["foxnaught", "--one-shot", "hi"]),
            mock.patch("foxnaught.__main__.MistralClient") as client_cls,
            mock.patch("foxnaught.__main__.Agent") as agent_cls,
            mock.patch("sys.stdout") as stdout,
        ):
            agent_cls.return_value.run_turn.return_value = "hello"
            result = main()
        self.assertEqual(result, 0)
        agent_cls.return_value.run_turn.assert_called_once_with("hi")
        stdout.write.assert_any_call("hello")

    def test_one_shot_error_returns_nonzero(self) -> None:
        with (
            mock.patch("sys.argv", ["foxnaught", "--one-shot", "hi"]),
            mock.patch("foxnaught.__main__.MistralClient") as client_cls,
            mock.patch("foxnaught.__main__.Agent") as agent_cls,
        ):
            agent_cls.return_value.run_turn.side_effect = MistralError("boom")
            result = main()
        self.assertEqual(result, 1)

    def test_serve_starts_server(self) -> None:
        with (
            mock.patch("sys.argv", ["foxnaught", "--serve", "--port", "0"]),
            mock.patch("foxnaught.__main__.MistralClient") as client_cls,
            mock.patch("foxnaught.__main__.Agent") as agent_cls,
            mock.patch("foxnaught.server.make_server") as make_server,
        ):
            make_server.return_value.server_address = ("127.0.0.1", 12345)
            make_server.return_value.serve_forever.side_effect = KeyboardInterrupt
            make_server.return_value.server_close.return_value = None
            result = main()
        self.assertEqual(result, 0)
        make_server.assert_called_once()


class LoadMemoryTest(unittest.TestCase):
    def test_load_memory_noop_when_missing(self) -> None:
        agent = make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "does-not-exist.json")
            load_memory(agent, path)
        self.assertEqual([m.role for m in agent.history], ["system"])

    def test_load_memory_loads_existing_file(self) -> None:
        agent = make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "memory.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('[{"role": "user", "content": "remembered"}]')
            load_memory(agent, path)
        self.assertIn("remembered", str(agent.history))
        self.assertEqual(agent.history[0].role, "system")

    def test_load_memory_raises_on_corrupt_file(self) -> None:
        agent = make_agent()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "corrupt.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            with self.assertRaises(ValueError):
                load_memory(agent, path)


class FlagWiringTest(unittest.TestCase):
    def test_zero_limits_are_passed_as_none(self) -> None:
        with (
            mock.patch("foxnaught.__main__.MistralClient"),
            mock.patch("foxnaught.__main__.Agent") as agent_cls,
        ):
            agent_cls.return_value.run_turn.return_value = "hello"
            result = main(
                argv=[
                    "--one-shot",
                    "hi",
                    "--history-limit",
                    "0",
                    "--archive-max-chars",
                    "0",
                    "--longterm-max-chars",
                    "0",
                ]
            )
        self.assertEqual(result, 0)
        _, kwargs = agent_cls.call_args
        self.assertIsNone(kwargs["history_limit"])
        self.assertIsNone(kwargs["memory_archive_max_chars"])
        self.assertIsNone(kwargs["longterm_max_chars"])

    def test_one_shot_stream_uses_run_turn_stream(self) -> None:
        with (
            mock.patch("foxnaught.__main__.MistralClient"),
            mock.patch("foxnaught.__main__.Agent") as agent_cls,
        ):
            agent_cls.return_value.run_turn_stream.return_value = "hello"
            result = main(argv=["--one-shot", "hi", "--stream"])
        self.assertEqual(result, 0)
        stream_mock = agent_cls.return_value.run_turn_stream
        stream_mock.assert_called_once()
        self.assertEqual(stream_mock.call_args.args[0], "hi")
        agent_cls.return_value.run_turn.assert_not_called()

    def test_one_shot_saves_memory_when_flag_set(self) -> None:
        with (
            mock.patch("foxnaught.__main__.MistralClient"),
            mock.patch("foxnaught.__main__.Agent") as agent_cls,
            mock.patch("foxnaught.__main__.load_memory") as load_memory_mock,
            mock.patch("foxnaught.__main__.save_memory") as save_memory_mock,
        ):
            agent_cls.return_value.run_turn.return_value = "hello"
            parent = mock.Mock()
            parent.attach_mock(load_memory_mock, "load_memory")
            parent.attach_mock(save_memory_mock, "save_memory")
            parent.attach_mock(agent_cls.return_value, "agent")
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "memory.json")
                result = main(argv=["--one-shot", "hi", "--memory", path])
        self.assertEqual(result, 0)
        names = [
            name
            for name, _args, _kwargs in parent.mock_calls
            if name in ("load_memory", "agent.run_turn", "save_memory")
        ]
        self.assertEqual(names, ["load_memory", "agent.run_turn", "save_memory"])
        load_memory_mock.assert_called_once_with(agent_cls.return_value, path)
        save_memory_mock.assert_called_once_with(agent_cls.return_value, path)


if __name__ == "__main__":
    unittest.main()
