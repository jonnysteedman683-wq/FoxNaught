"""Tests for the CLI one-shot mode."""

from __future__ import annotations

import unittest
from unittest import mock

from foxnaught.__main__ import main
from foxnaught.client import MistralError


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


if __name__ == "__main__":
    unittest.main()
