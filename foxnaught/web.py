"""Web-fetch tool with SSRF protection."""

from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import urllib.parse

from .tools import ToolRegistry

MAX_BYTES = 512_000
MAX_CHARS = 4_000
FETCH_TIMEOUT = 15
MAX_REDIRECTS = 5
REDIRECT_STATUSES = {301, 302, 303, 307, 308}

_TAG_RE = re.compile(r"<(?:script|style)\b[^>]*>.*?</(?:script|style)>", re.DOTALL | re.IGNORECASE)
_STRIP_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t]*\n[ \t\n]*")
_CHARSET_RE = re.compile(r"charset=([^;\s]+)", re.IGNORECASE)


class DisallowedHostError(ValueError):
    """Raised when a URL targets a private, loopback, or otherwise unsafe address."""


class FetchHTTPError(RuntimeError):
    """Raised when the remote server returns an HTTP error status."""


def _resolve_public_address(hostname: str) -> str:
    """Resolve hostname and return a validated public IP to pin the connection to."""
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
    if not addrinfos:
        raise DisallowedHostError(f"cannot resolve host {hostname!r}")
    return addrinfos[0][4][0]


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection to a pre-validated IP with SNI/cert checks on the hostname."""

    def __init__(self, address: str, hostname: str, **kwargs: object) -> None:
        super().__init__(address, **kwargs)
        self._pinned_hostname = hostname

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self.host, self.port), self.timeout, self.source_address
        )
        self.sock = self._context.wrap_socket(
            self.sock, server_hostname=self._pinned_hostname
        )


def _open_pinned(
    address: str, hostname: str, scheme: str, port: int | None
) -> http.client.HTTPConnection:
    if ":" in address and not address.startswith("["):
        address = f"[{address}]"
    kwargs: dict = {"port": port, "timeout": FETCH_TIMEOUT}
    if scheme == "https":
        return _PinnedHTTPSConnection(address, hostname, **kwargs)
    return http.client.HTTPConnection(address, **kwargs)


def _follow_redirects(url: str, max_bytes: int) -> tuple[str, bytes]:
    """GET url via a connection pinned to each validated IP; return final content."""
    for _ in range(MAX_REDIRECTS + 1):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https"):
            raise DisallowedHostError(f"only http and https URLs are supported: {url!r}")
        hostname = parsed.hostname
        if not hostname:
            raise DisallowedHostError(f"URL has no hostname: {url!r}")
        address = _resolve_public_address(hostname)
        host_header = hostname if parsed.port is None else f"{hostname}:{parsed.port}"
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        connection = _open_pinned(address, hostname, parsed.scheme, parsed.port)
        try:
            connection.request(
                "GET",
                path,
                headers={
                    "Host": host_header,
                    "User-Agent": "FoxNaught-agent/0.1",
                    "Accept": "text/html,text/plain,*/*",
                },
            )
            response = connection.getresponse()
            content_type = response.headers.get("Content-Type", "")
            location = response.headers.get("Location")
            if response.status in REDIRECT_STATUSES and location:
                response.read(max_bytes)
                url = urllib.parse.urljoin(url, location)
                continue
            if response.status >= 400:
                raise FetchHTTPError(f"HTTP {response.status} {response.reason}")
            return content_type, response.read(max_bytes)
        finally:
            connection.close()
    raise FetchHTTPError(f"too many redirects (more than {MAX_REDIRECTS})")


def _charset(content_type: str) -> str:
    match = _CHARSET_RE.search(content_type or "")
    if not match:
        return "utf-8"
    return match.group(1).strip("\"'").strip() or "utf-8"


def _extract_text(html: str) -> str:
    html = _TAG_RE.sub(" ", html)
    html = _STRIP_RE.sub("\n", html)
    html = html.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
    return _WHITESPACE_RE.sub("\n", html).strip()


def fetch_url(url: str, max_chars: int = MAX_CHARS) -> str:
    """Fetch a public URL and return a text summary of its content."""
    max_chars = max(200, min(int(max_chars), MAX_CHARS))
    url = (url or "").strip()
    try:
        content_type, body = _follow_redirects(url, MAX_BYTES)
    except DisallowedHostError as exc:
        return f"Error: {exc}"
    except (FetchHTTPError, http.client.HTTPException, OSError) as exc:
        return f"Error: fetch failed: {exc}"
    try:
        text = body.decode(_charset(content_type), errors="replace")
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
            "Refuses private, loopback, and non-resolvable hosts; redirects are "
            "followed only after re-validating each target host."
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
