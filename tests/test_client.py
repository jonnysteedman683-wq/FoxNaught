"""Tests for the Mistral API client."""

from __future__ import annotations

import json
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from email.message import Message as EmailMessage
from email.utils import format_datetime
from unittest import mock

from foxnaught.client import (
    MAX_RETRY_AFTER_SECONDS,
    Message,
    MistralClient,
    MistralError,
)


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

    @staticmethod
    def http_error(status: int, headers: dict[str, str] | None = None) -> urllib.error.HTTPError:
        hdrs = EmailMessage()
        for key, value in (headers or {}).items():
            hdrs[key] = value
        error = urllib.error.HTTPError("u", status, "error", hdrs, mock.MagicMock())
        error.read = mock.MagicMock(return_value=b'{"error": "transient"}')
        return error

    @staticmethod
    def success_response() -> mock.MagicMock:
        return make_response(
            json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
            ).encode()
        )

    def test_server_error_is_retried_then_succeeds(self) -> None:
        with (
            mock.patch(
                "urllib.request.urlopen",
                side_effect=[self.http_error(503), self.success_response()],
            ) as urlopen,
            mock.patch("foxnaught.client.time.sleep") as sleep,
        ):
            result = self.client.chat([Message(role="user", content="hello")])
        self.assertEqual(result.message.content, "ok")
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(sleep.call_count, 1)

    def test_server_error_exhausts_retries(self) -> None:
        with (
            mock.patch(
                "urllib.request.urlopen", side_effect=[self.http_error(500)] * 3
            ) as urlopen,
            mock.patch("foxnaught.client.time.sleep") as sleep,
        ):
            with self.assertRaises(MistralError) as ctx:
                self.client.chat([Message(role="user", content="hello")])
        self.assertIn("500", str(ctx.exception))
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_client_error_is_not_retried(self) -> None:
        with mock.patch(
            "urllib.request.urlopen", side_effect=[self.http_error(404)]
        ) as urlopen:
            with self.assertRaises(MistralError) as ctx:
                self.client.chat([Message(role="user", content="hello")])
        self.assertIn("404", str(ctx.exception))
        self.assertEqual(urlopen.call_count, 1)

    def test_network_failure_is_retried_then_succeeds(self) -> None:
        failure = urllib.error.URLError("connection reset by peer")
        with (
            mock.patch(
                "urllib.request.urlopen",
                side_effect=[failure, self.success_response()],
            ) as urlopen,
            mock.patch("foxnaught.client.time.sleep") as sleep,
        ):
            result = self.client.chat([Message(role="user", content="hello")])
        self.assertEqual(result.message.content, "ok")
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(sleep.call_count, 1)

    def test_network_failure_exhausts_retries(self) -> None:
        failure = urllib.error.URLError("connection refused")
        with (
            mock.patch("urllib.request.urlopen", side_effect=[failure] * 3) as urlopen,
            mock.patch("foxnaught.client.time.sleep") as sleep,
        ):
            with self.assertRaises(MistralError) as ctx:
                self.client.chat([Message(role="user", content="hello")])
        self.assertIn("could not reach the Mistral API", str(ctx.exception))
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_retry_after_seconds_is_honoured(self) -> None:
        error = self.http_error(429, {"Retry-After": "7"})
        with (
            mock.patch(
                "urllib.request.urlopen",
                side_effect=[error, self.success_response()],
            ),
            mock.patch("foxnaught.client.time.sleep") as sleep,
        ):
            self.client.chat([Message(role="user", content="hello")])
        (delay,) = sleep.call_args.args
        self.assertEqual(delay, 7.0)

    def test_retry_after_is_capped(self) -> None:
        error = self.http_error(503, {"Retry-After": "600"})
        with (
            mock.patch(
                "urllib.request.urlopen",
                side_effect=[error, self.success_response()],
            ),
            mock.patch("foxnaught.client.time.sleep") as sleep,
        ):
            self.client.chat([Message(role="user", content="hello")])
        (delay,) = sleep.call_args.args
        self.assertEqual(delay, MAX_RETRY_AFTER_SECONDS)

    def test_retry_after_http_date_is_honoured(self) -> None:
        retry_at = datetime.now(timezone.utc) + timedelta(seconds=5)
        error = self.http_error(429, {"Retry-After": format_datetime(retry_at)})
        with (
            mock.patch(
                "urllib.request.urlopen",
                side_effect=[error, self.success_response()],
            ),
            mock.patch("foxnaught.client.time.sleep") as sleep,
        ):
            self.client.chat([Message(role="user", content="hello")])
        (delay,) = sleep.call_args.args
        self.assertGreaterEqual(delay, 4.0)
        self.assertLessEqual(delay, 5.0)

    def test_invalid_retry_after_falls_back_to_backoff(self) -> None:
        error = self.http_error(429, {"Retry-After": "not-a-date"})
        with (
            mock.patch(
                "urllib.request.urlopen",
                side_effect=[error, self.success_response()],
            ),
            mock.patch("foxnaught.client.time.sleep") as sleep,
        ):
            self.client.chat([Message(role="user", content="hello")])
        (delay,) = sleep.call_args.args
        self.assertGreaterEqual(delay, 0.5)
        self.assertLessEqual(delay, 1.5)

    def test_transient_retries_can_be_disabled(self) -> None:
        client = MistralClient(api_key="test-key", retries_on_rate_limit=False)
        with mock.patch(
            "urllib.request.urlopen", side_effect=[self.http_error(503)]
        ) as urlopen:
            with self.assertRaises(MistralError):
                client.chat([Message(role="user", content="hello")])
        self.assertEqual(urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
