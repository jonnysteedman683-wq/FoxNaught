"""Tests for the Mistral API client."""

from __future__ import annotations

import json
import unittest
import urllib.error
from unittest import mock

from foxnaught.client import Message, MistralClient, MistralError


def make_response(body: bytes) -> mock.MagicMock:
    response = mock.MagicMock()
    response.read.return_value = body
    response.__enter__ = mock.MagicMock(return_value=response)
    response.__exit__ = mock.MagicMock(return_value=False)
    return response


class MistralClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MistralClient(api_key="test-key")

    def test_missing_api_key_raises(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(MistralError):
                MistralClient()

    def test_chat_sends_model_and_messages(self) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hi there"},
                    }
                ]
            }
        ).encode()
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = make_response(body)
            result = self.client.chat([Message(role="user", content="hello")])
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "mistral-small-latest")
        self.assertEqual(payload["messages"][0], {"role": "user", "content": "hello"})
        self.assertNotIn("tools", payload)
        self.assertEqual(result.message.content, "hi there")
        self.assertEqual(result.finish_reason, "stop")

    def test_chat_omits_tools_when_empty(self) -> None:
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        ).encode()
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = make_response(body)
            self.client.chat([Message(role="user", content="hello")], tools=[])
        self.assertNotIn("tools", json.loads(urlopen.call_args.args[0].data))

    def test_http_error_raises_mistral_error(self) -> None:
        error = urllib.error.HTTPError(
            "https://api.mistral.ai/v1/chat/completions", 401, "Unauthorized", {}, mock.MagicMock()
        )
        error.read = mock.MagicMock(return_value=b'"bad key"')
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(MistralError):
                self.client.chat([Message(role="user", content="hello")])

    def test_url_error_raises_mistral_error(self) -> None:
        failure = urllib.error.URLError("connection refused")
        with mock.patch("urllib.request.urlopen", side_effect=failure):
            with self.assertRaises(MistralError):
                self.client.chat([Message(role="user", content="hello")])

    def test_empty_choices_raises(self) -> None:
        body = json.dumps({"object": "chat.completion"}).encode()
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = make_response(body)
            with self.assertRaises(MistralError):
                self.client.chat([Message(role="user", content="hello")])

    def test_retry_delay_applies_jitter(self) -> None:
        client = MistralClient(api_key="test-key")
        with mock.patch("foxnaught.client.random.uniform", return_value=1.5) as uniform:
            self.assertEqual(client._retry_delay(1), 1.5)
        uniform.assert_called_once_with(0.5, 1.5)
        with mock.patch("foxnaught.client.random.uniform", return_value=0.1):
            self.assertEqual(client._retry_delay(2), 0.2)

    def test_rate_limit_is_retried_then_succeeds(self) -> None:
        rate_limited = urllib.error.HTTPError(
            "u", 429, "Too Many Requests", {}, mock.MagicMock()
        )
        rate_limited.read = mock.MagicMock(return_value=b'{"code": 1300}')
        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        ).encode()
        with (
            mock.patch("urllib.request.urlopen", side_effect=[rate_limited, make_response(body)]) as urlopen,
            mock.patch("foxnaught.client.time.sleep") as sleep,
        ):
            result = self.client.chat([Message(role="user", content="hello")])
        self.assertEqual(result.message.content, "ok")
        self.assertEqual(urlopen.call_count, 2)
        (delay,) = sleep.call_args.args
        self.assertGreaterEqual(delay, 0.5)
        self.assertLessEqual(delay, 1.5)

    def test_rate_limit_exhausts_retries(self) -> None:
        rate_limited = urllib.error.HTTPError(
            "u", 429, "Too Many Requests", {}, mock.MagicMock()
        )
        rate_limited.read = mock.MagicMock(return_value=b'{"code": 1300}')
        with (
            mock.patch("urllib.request.urlopen", side_effect=[rate_limited] * 3) as urlopen,
            mock.patch("foxnaught.client.time.sleep") as sleep,
        ):
            with self.assertRaises(MistralError) as ctx:
                self.client.chat([Message(role="user", content="hello")])
        self.assertIn("429", str(ctx.exception))
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_retries_can_be_disabled(self) -> None:
        rate_limited = urllib.error.HTTPError(
            "u", 429, "Too Many Requests", {}, mock.MagicMock()
        )
        rate_limited.read = mock.MagicMock(return_value=b'{"code": 1300}')
        client = MistralClient(api_key="test-key", retries_on_rate_limit=False)
        with mock.patch("urllib.request.urlopen", side_effect=[rate_limited]) as urlopen:
            with self.assertRaises(MistralError):
                client.chat([Message(role="user", content="hello")])
        self.assertEqual(urlopen.call_count, 1)

    def test_other_http_errors_are_not_retried(self) -> None:
        unauthorised = urllib.error.HTTPError(
            "u", 401, "Unauthorized", {}, mock.MagicMock()
        )
        unauthorised.read = mock.MagicMock(return_value=b'"bad key"')
        with mock.patch("urllib.request.urlopen", side_effect=[unauthorised]) as urlopen:
            with self.assertRaises(MistralError):
                self.client.chat([Message(role="user", content="hello")])
        self.assertEqual(urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
