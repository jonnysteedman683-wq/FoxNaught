"""Tests for the web-fetch tool."""

from __future__ import annotations

import socket
import unittest
import urllib.error
from unittest import mock

from foxnaught.web import _extract_text, fetch_url, register_web
from foxnaught.tools import ToolRegistry


def urlopen_return(body: bytes, content_type: str = "text/plain") -> mock.MagicMock:
    response = mock.MagicMock()
    response.read.return_value = body
    response.headers = {"Content-Type": content_type}
    response.__enter__ = mock.MagicMock(return_value=response)
    response.__exit__ = mock.MagicMock(return_value=False)
    return response


class FetchUrlTest(unittest.TestCase):
    def test_valid_url_triggers_public_resolution(self) -> None:
        addr_info = [(2, 1, 6, "", ("93.184.216.34", 0))]
        with (
            mock.patch("socket.getaddrinfo", return_value=addr_info),
            mock.patch(
                "urllib.request.urlopen",
                return_value=urlopen_return(b"hello world"),
            ) as urlopen,
        ):
            result = fetch_url("https://example.com/page")
        self.assertEqual(result, "hello world")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.com/page")

    def test_rejects_non_http_schemes(self) -> None:
        self.assertTrue(fetch_url("ftp://example.com/file").startswith("Error:"))
        self.assertTrue(fetch_url("file:///etc/passwd").startswith("Error:"))

    def test_rejects_non_public_addresses(self) -> None:
        addr_info = [(2, 1, 6, "", ("127.0.0.1", 0))]
        with mock.patch("socket.getaddrinfo", return_value=addr_info):
            self.assertTrue(
                fetch_url("http://localhost/secret").startswith(
                    "Error: refusing to fetch non-public"
                )
            )

    def test_rejects_unresolvable_host(self) -> None:
        gai_error = socket.gaierror("no DNS")
        with mock.patch("socket.getaddrinfo", side_effect=gai_error):
            self.assertTrue(fetch_url("https://no-such-host.invalid/").startswith("Error:"))

    def test_reports_fetch_failure(self) -> None:
        addr_info = [(2, 1, 6, "", ("93.184.216.34", 0))]
        with (
            mock.patch("socket.getaddrinfo", return_value=addr_info),
            mock.patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.HTTPError("u", 404, "nf", {}, mock.MagicMock()),
            ),
        ):
            self.assertTrue(fetch_url("https://example.com/x").startswith("Error: fetch failed"))

    def test_truncates_long_content(self) -> None:
        addr_info = [(2, 1, 6, "", ("93.184.216.34", 0))]
        with (
            mock.patch("socket.getaddrinfo", return_value=addr_info),
            mock.patch(
                "urllib.request.urlopen",
                return_value=urlopen_return(b"x" * 900),
            ),
        ):
            result = fetch_url("https://example.com/big", max_chars=300)
        self.assertEqual(len(result), 301)
        self.assertTrue(result.endswith("…"))

    def test_strips_html_tags(self) -> None:
        html = "<html><head><style>.a{}</style></head><body><h1>Hi</h1><p>there</p><script>bad()</script></body></html>"
        text = _extract_text(html)
        self.assertNotIn("<", text)
        self.assertNotIn("bad()", text)
        self.assertIn("Hi", text)
        self.assertIn("there", text)

    def test_registered_in_registry(self) -> None:
        registry = ToolRegistry()
        register_web(registry)
        self.assertEqual(registry.schemas()[0]["function"]["name"], "fetch_url")


if __name__ == "__main__":
    unittest.main()
