"""Tests for SSE streaming in the Mistral client."""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest import mock

from foxnaught.client import Message, MistralClient, MistralError


def sse_response(chunks: list[str]) -> io.BytesIO:
    lines = "\n\n".join(f"data: {chunk}" for chunk in chunks) + "\n\ndata: [DONE]\n\n"
    return io.BytesIO(lines.encode())


def make_client() -> MistralClient:
    return MistralClient(api_key="test-key")


class ChatStreamTest(unittest.TestCase):
    def test_yields_content_deltas_and_finish_reason(self) -> None:
        chunks = [
            json.dumps({"choices": [{"delta": {"content": "He"}, "finish_reason": None}]}),
            json.dumps({"choices": [{"delta": {"content": "y"}, "finish_reason": None}]}),
            json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ]
        with mock.patch("urllib.request.urlopen", return_value=sse_response(chunks)):
            events = list(make_client().chat_stream([Message(role="user", content="hi")]))
        self.assertEqual([e.content for e in events], ["He", "y", None])
        self.assertEqual([e.finish_reason for e in events], [None, None, "stop"])

    def test_sends_stream_flag(self) -> None:
        with mock.patch("urllib.request.urlopen", return_value=sse_response([])) as urlopen:
            list(make_client().chat_stream([Message(role="user", content="hi")]))
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertTrue(payload["stream"])

    def test_skips_non_data_lines_and_empty_choices(self) -> None:
        body = b": keep-alive\n\ndata: {\"choices\": []}\n\ndata: [DONE]\n\n"
        with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(body)):
            events = list(make_client().chat_stream([Message(role="user", content="hi")]))
        self.assertEqual(events, [])

    def test_invalid_chunk_raises_mistral_error(self) -> None:
        body = b"data: {oops}\n\ndata: [DONE]\n\n"
        with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(body)):
            with self.assertRaises(MistralError):
                list(make_client().chat_stream([Message(role="user", content="hi")]))

    def test_rate_limited_stream_is_retried(self) -> None:
        rate_limited = urllib.error.HTTPError("u", 429, "TLM", {}, mock.MagicMock())
        rate_limited.read = mock.MagicMock(return_value=b"{}")
        chunks = [json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]})]
        with (
            mock.patch("urllib.request.urlopen", side_effect=[rate_limited, sse_response(chunks)]) as urlopen,
            mock.patch("foxnaught.client.time.sleep") as sleep,
        ):
            events = list(make_client().chat_stream([Message(role="user", content="hi")]))
        self.assertEqual([e.content for e in events], ["ok"])
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(1.0)


if __name__ == "__main__":
    unittest.main()
