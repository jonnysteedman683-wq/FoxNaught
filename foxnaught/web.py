"""Web-fetch tool with SSRF protection."""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request

from .tools import ToolRegistry

MAX_BYTES = 512_000
MAX_CHARS = 4_000
FETCH_TIMEOUT = 15

_TAG_RE = re.compile(r"<(?:script|style)\b[^>]*>.*?</(?:script|style)>", re.DOTALL | re.IGNORECASE)
_STRIP_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t]*\n[ \t\n]*")


class DisallowedHostError(ValueError):
    """Raised when a URL targets a private, loopback, or otherwise unsafe address."""


def _validate_public_host(hostname: str) -> None:
    try:
        addrinfos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise DisallowedHostError(f"cannot resolve host {hostname!r}: {exc}") from exc
    for _, _, _, _, sockaddr in addrinfos:
        ip = ipaddress.ip_address(sockaddr[0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise DisallowedHostError(f"refusing to fetch non-public address {ip}")


def _extract_text(html: str) -> str:
    html = _TAG_RE.sub(" ", html)
    html = _STRIP_RE.sub("\n", html)
    html = html.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
    return _WHITESPACE_RE.sub("\n", html).strip()


def fetch_url(url: str, max_chars: int = MAX_CHARS) -> str:
    """Fetch a public URL and return a text summary of its content."""
    max_chars = max(200, min(int(max_chars), MAX_CHARS))
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError as exc:
        return f"Error: invalid URL: {exc}"
    if parsed.scheme not in ("http", "https"):
        return "Error: only http and https URLs are supported"
    if not parsed.hostname:
        return "Error: URL has no hostname"
    try:
        _validate_public_host(parsed.hostname)
    except DisallowedHostError as exc:
        return f"Error: {exc}"

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FoxNaught-agent/0.1", "Accept": "text/html,text/plain,*/*"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
            content_type = response.headers.get("Content-Type", "")
            body = response.read(MAX_BYTES)
    except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        return f"Error: fetch failed: {exc}"
    try:
        text = body.decode(content_type.rsplit("charset=", 1)[-1] if "charset=" in content_type else "utf-8", errors="replace")
    except LookupError:
        text = body.decode("utf-8", errors="replace")
    if "html" in content_type:
        text = _extract_text(text)
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text or ""


def register_web(registry: ToolRegistry) -> None:
    registry.register(
        name="fetch_url",
        description=(
            "Fetch a public web page or text resource over http(s) and return its "
            "content as plain text (HTML tags removed), truncated to max_chars. "
            "Refuses private, loopback, and non-resolvable hosts."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The http(s) URL to fetch."},
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (200-4000).",
                    "default": MAX_CHARS,
                },
            },
            "required": ["url"],
        },
        func=fetch_url,
    )
