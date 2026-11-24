"""Tests for the web-fetch tool."""

from __future__ import annotations

import socket
import unittest
from unittest import mock

from foxnaught.tools import ToolRegistry
from foxnaught.web import _charset, _extract_text, fetch_url, register_web

PUBLIC_ADDR = [(2, 1, 6, "", ("93.184.216.34", 0))]
PRIVATE_ADDR = [(2, 1, 6, "", ("10.0.0.5", 0))]


def fake_response(
    status: int = 200,
    body: bytes = b"",
    content_type: str = "text/plain",
    location: str | None = None,
) -> mock.MagicMock:
    response = mock.MagicMock()
    response.status = status
    response.reason = "OK" if status < 400 else "Error"
    response.headers = {"Content-Type": content_type}
    if location is not None:
        response.headers["Location"] = location
    response.read.return_value = body
    return response


def fake_connection(response: mock.MagicMock) -> mock.MagicMock:
    connection = mock.MagicMock()
    connection.getresponse.return_value = response
    return connection


class FetchUrlTest(unittest.TestCase):
    def test_pins_connection_to_validated_address(self) -> None:
        with (
            mock.patch("socket.getaddrinfo", return_value=PUBLIC_ADDR),
            mock.patch(
                "foxnaught.web._open_pinned",
                return_value=fake_connection(fake_response(body=b"hello world")),
            ) as open_pinned,
        ):
            result = fetch_url("https://example.com/page?q=1")
        self.assertEqual(result, "hello world")
        address, hostname, scheme, port = open_pinned.call_args.args
        self.assertEqual(address, "93.184.216.34")
        self.assertEqual(hostname, "example.com")
        self.assertEqual(scheme, "https")
        self.assertIsNone(port)
        request = open_pinned.return_value.request.call_args
        self.assertEqual(request.args[0], "GET")
        self.assertEqual(request.args[1], "/page?q=1")
        self.assertEqual(request.kwargs["headers"]["Host"], "example.com")

    def test_rejects_out_of_range_port(self) -> None:
        with mock.patch("socket.getaddrinfo", return_value=PUBLIC_ADDR):
            self.assertTrue(
                fetch_url("https://example.com:99999/").startswith("Error: fetch failed")
            )

    def test_rejects_redirect_without_location(self) -> None:
        with (
            mock.patch("socket.getaddrinfo", return_value=PUBLIC_ADDR),
            mock.patch(
                "foxnaught.web._open_pinned",
                return_value=fake_connection(fake_response(status=302)),
            ),
        ):
            self.assertTrue(
                fetch_url("https://example.com/redirect").startswith(
                    "Error: fetch failed: HTTP 302"
                )
            )

    def test_rejects_non_success_non_redirect_status(self) -> None:
        with (
            mock.patch("socket.getaddrinfo", return_value=PUBLIC_ADDR),
            mock.patch(
                "foxnaught.web._open_pinned",
                return_value=fake_connection(fake_response(status=304)),
            ),
        ):
            self.assertTrue(
                fetch_url("https://example.com/conditional").startswith(
                    "Error: fetch failed: HTTP 304"
                )
            )

    def test_rejects_non_http_schemes(self) -> None:
        self.assertTrue(fetch_url("ftp://example.com/file").startswith("Error:"))
        self.assertTrue(fetch_url("file:///etc/passwd").startswith("Error:"))

    def test_rejects_non_public_addresses(self) -> None:
        with mock.patch("socket.getaddrinfo", return_value=PRIVATE_ADDR):
            self.assertTrue(
                fetch_url("http://localhost/secret").startswith(
                    "Error: refusing to fetch non-public"
                )
            )

    def test_rejects_unresolvable_host(self) -> None:
        gai_error = socket.gaierror("no DNS")
        with mock.patch("socket.getaddrinfo", side_effect=gai_error):
            self.assertTrue(fetch_url("https://no-such-host.invalid/").startswith("Error:"))

    def test_reports_http_error_status(self) -> None:
        with (
            mock.patch("socket.getaddrinfo", return_value=PUBLIC_ADDR),
            mock.patch(
                "foxnaught.web._open_pinned",
                return_value=fake_connection(fake_response(status=404)),
            ),
        ):
            self.assertTrue(
                fetch_url("https://example.com/x").startswith("Error: fetch failed")
            )

    def test_refuses_redirect_to_private_host(self) -> None:
        redirecting = fake_connection(
            fake_response(status=302, location="http://internal.example/secret")
        )
        with (
            mock.patch(
                "socket.getaddrinfo",
                side_effect=[PUBLIC_ADDR, PRIVATE_ADDR],
            ) as getaddrinfo,
            mock.patch("foxnaught.web._open_pinned", side_effect=[redirecting]) as open_pinned,
        ):
            result = fetch_url("https://example.com/redirect")
        self.assertTrue(result.startswith("Error: refusing to fetch non-public"))
        self.assertEqual(getaddrinfo.call_count, 2)
        self.assertEqual(open_pinned.call_count, 1)

    def test_follows_redirect_to_public_host(self) -> None:
        redirecting = fake_connection(
            fake_response(status=302, location="https://example.org/next")
        )
        final = fake_connection(fake_response(body=b"landed"))
        with (
            mock.patch("socket.getaddrinfo", return_value=PUBLIC_ADDR),
            mock.patch(
                "foxnaught.web._open_pinned", side_effect=[redirecting, final]
            ) as open_pinned,
        ):
            result = fetch_url("https://example.com/redirect")
        self.assertEqual(result, "landed")
        self.assertEqual(open_pinned.call_count, 2)
        second_request = final.request.call_args
        self.assertEqual(second_request.args[1], "/next")

    def test_rejects_non_http_redirect_scheme(self) -> None:
        redirecting = fake_connection(
            fake_response(status=302, location="ftp://example.org/next")
        )
        with (
            mock.patch("socket.getaddrinfo", return_value=PUBLIC_ADDR),
            mock.patch("foxnaught.web._open_pinned", return_value=redirecting),
        ):
            self.assertTrue(
                fetch_url("https://example.com/redirect").startswith("Error: only http")
            )

    def test_rejects_too_many_redirects(self) -> None:
        def redirecting_connection(*_args, **_kwargs):
            return fake_connection(
                fake_response(status=302, location="https://example.com/loop")
            )

        with (
            mock.patch("socket.getaddrinfo", return_value=PUBLIC_ADDR),
            mock.patch("foxnaught.web._open_pinned", side_effect=redirecting_connection),
        ):
            self.assertTrue(
                fetch_url("https://example.com/redirect").startswith(
                    "Error: fetch failed: too many redirects"
                )
            )

    def test_truncates_long_content(self) -> None:
        with (
            mock.patch("socket.getaddrinfo", return_value=PUBLIC_ADDR),
            mock.patch(
                "foxnaught.web._open_pinned",
                return_value=fake_connection(fake_response(body=b"x" * 900)),
            ),
        ):
            result = fetch_url("https://example.com/big", max_chars=300)
        self.assertEqual(len(result), 301)
        self.assertTrue(result.endswith("…"))

    def test_uses_declared_charset_even_when_quoted(self) -> None:
        body = "café".encode("iso-8859-1")
        with (
            mock.patch("socket.getaddrinfo", return_value=PUBLIC_ADDR),
            mock.patch(
                "foxnaught.web._open_pinned",
                return_value=fake_connection(
                    fake_response(body=body, content_type='text/html; charset="iso-8859-1"')
                ),
            ),
        ):
            result = fetch_url("https://example.com/encoded")
        self.assertIn("café", result)
        self.assertEqual(_charset('text/html; charset="iso-8859-1"'), "iso-8859-1")
        self.assertEqual(_charset("text/html"), "utf-8")
        self.assertEqual(_charset("text/html; xcharset=utf-16"), "utf-8")

    def test_strips_html_tags(self) -> None:
        html = "<html><head><style>.a{}</style></head><body><h1>Hi</h1><p>there</p><script>bad()</script></body></html>"
        text = _extract_text(html)
        self.assertNotIn("<", text)
        self.assertNotIn("bad()", text)
        self.assertIn("Hi", text)
        self.assertIn("there", text)

    def test_decodes_entities_once(self) -> None:
        self.assertEqual(_extract_text("&amp;lt;"), "&lt;")
        self.assertEqual(_extract_text("&#60;p&amp;gt&#62;"), "<p&gt>")
        self.assertEqual(_extract_text("a&nbsp;b"), "a b")

    def test_registered_in_registry(self) -> None:
        registry = ToolRegistry()
        register_web(registry)
        self.assertEqual(registry.schemas()[0]["function"]["name"], "fetch_url")


if __name__ == "__main__":
    unittest.main()
