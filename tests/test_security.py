"""Security tests for SSRF protection and sandboxing."""

from __future__ import annotations

import ipaddress
import os
import socket
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from foxnaught.files import make_read_file, make_write_file, make_list_files, SandboxError
from foxnaught.web import (
    fetch_url,
    DisallowedHostError,
    FetchHTTPError,
    _resolve_public_address,
    _follow_redirects,
)


class SandboxSecurityTest(unittest.TestCase):
    """Test sandboxing of file operations."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.subdir = os.path.join(self.tmpdir, "subdir")
        os.makedirs(self.subdir)
        with open(os.path.join(self.tmpdir, "test.txt"), "w") as fh:
            fh.write("test content")

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_read_file_blocked_absolute_escape(self) -> None:
        """Test that absolute paths escaping the base dir are blocked."""
        read_file = make_read_file(self.tmpdir)
        result = read_file("/etc/passwd")
        self.assertIn("Error", result)
        self.assertIn("escapes", result)

    def test_read_file_blocked_relative_escape(self) -> None:
        """Test that relative paths escaping the base dir are blocked."""
        read_file = make_read_file(self.tmpdir)
        result = read_file("../etc/passwd")
        self.assertIn("Error", result)
        self.assertIn("escapes", result)

    def test_read_file_blocked_parent_escape(self) -> None:
        """Test that parent directory traversal is blocked."""
        read_file = make_read_file(self.tmpdir)
        result = read_file("../../../etc/passwd")
        self.assertIn("Error", result)

    def test_read_file_allowed_within_sandbox(self) -> None:
        """Test that files within the sandbox are accessible."""
        read_file = make_read_file(self.tmpdir)
        result = read_file("test.txt")
        self.assertEqual(result, "test content")

    def test_read_file_allowed_in_subdir(self) -> None:
        """Test that files in subdirectories are accessible."""
        with open(os.path.join(self.subdir, "sub.txt"), "w") as fh:
            fh.write("sub content")
        read_file = make_read_file(self.tmpdir)
        result = read_file("subdir/sub.txt")
        self.assertEqual(result, "sub content")

    def test_write_file_blocked_escape(self) -> None:
        """Test that writing outside the sandbox is blocked."""
        write_file = make_write_file(self.tmpdir)
        result = write_file("/etc/test.txt", "content")
        self.assertIn("Error", result)
        self.assertIn("escapes", result)

    def test_write_file_allowed_within_sandbox(self) -> None:
        """Test that writing within the sandbox works."""
        write_file = make_write_file(self.tmpdir)
        result = write_file("new.txt", "new content")
        self.assertIn("wrote", result)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "new.txt")))

    def test_write_file_creates_parent_dirs(self) -> None:
        """Test that parent directories are created."""
        write_file = make_write_file(self.tmpdir)
        result = write_file("newdir/new.txt", "content")
        self.assertIn("wrote", result)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "newdir", "new.txt")))

    def test_list_files_blocked_escape(self) -> None:
        """Test that listing outside the sandbox is blocked."""
        list_files = make_list_files(self.tmpdir)
        result = list_files("../")
        self.assertIn("Error", result)

    def test_list_files_allowed_within_sandbox(self) -> None:
        """Test that listing within the sandbox works."""
        list_files = make_list_files(self.tmpdir)
        result = list_files(".")
        self.assertIn("test.txt", result)

    def test_symlink_escape_blocked(self) -> None:
        """Test that symlink-based escapes are blocked."""
        link_path = os.path.join(self.tmpdir, "link")
        try:
            os.symlink("/etc", link_path)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks not supported on this system")

        read_file = make_read_file(self.tmpdir)
        result = read_file("link/passwd")
        self.assertIn("Error", result)


class SSRFProtectionTest(unittest.TestCase):
    """Test SSRF protection in web fetching."""

    def test_localhost_blocked(self) -> None:
        """Test that localhost is blocked."""
        result = fetch_url("http://localhost/test")
        self.assertIn("Error", result)

    def test_127_0_0_1_blocked(self) -> None:
        """Test that 127.0.0.1 is blocked."""
        result = fetch_url("http://127.0.0.1/test")
        self.assertIn("Error", result)

    def test_private_ipv4_blocked(self) -> None:
        """Test that private IPv4 addresses are blocked."""
        private_ips = [
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.1.1",  # link-local
        ]
        for ip in private_ips:
            with self.subTest(ip=ip):
                result = fetch_url(f"http://{ip}/test")
                self.assertIn("Error", result)

    def test_private_ipv6_blocked(self) -> None:
        """Test that private IPv6 addresses are blocked."""
        private_ipv6 = [
            "::1",  # loopback
            "fe80::1",  # link-local
            "fc00::1",  # unique local
        ]
        for ip in private_ipv6:
            with self.subTest(ip=ip):
                result = fetch_url(f"http://[{ip}]/test")
                self.assertIn("Error", result)

    def test_multicast_blocked(self) -> None:
        """Test that multicast addresses are blocked."""
        result = fetch_url("http://224.0.0.1/test")
        self.assertIn("Error", result)

    def test_reserved_blocked(self) -> None:
        """Test that reserved addresses are blocked."""
        result = fetch_url("http://192.0.2.1/test")  # TEST-NET-1
        self.assertIn("Error", result)

    def test_non_http_scheme_blocked(self) -> None:
        """Test that non-HTTP schemes are blocked."""
        result = fetch_url("ftp://example.com/test")
        self.assertIn("Error", result)
        
        result = fetch_url("file:///etc/passwd")
        self.assertIn("Error", result)

    def test_no_hostname_blocked(self) -> None:
        """Test that URLs without hostname are blocked."""
        result = fetch_url("http:///path")
        self.assertIn("Error", result)

    @patch("socket.getaddrinfo")
    def test_unresolvable_host_blocked(self, mock_getaddrinfo: MagicMock) -> None:
        """Test that unresolvable hosts are blocked."""
        mock_getaddrinfo.side_effect = socket.gaierror(-2, "Name or service not known")
        result = fetch_url("http://nonexistent.example.com/test")
        self.assertIn("Error", result)


class ResolvePublicAddressTest(unittest.TestCase):
    """Test the _resolve_public_address function."""

    @patch("socket.getaddrinfo")
    def test_resolves_public_ip(self, mock_getaddrinfo: MagicMock) -> None:
        """Test that public IPs are resolved."""
        mock_getaddrinfo.return_value = [
            (0, 0, 0, 0, ("93.184.216.34", 0, 0, 0)),
        ]
        result = _resolve_public_address("example.com")
        self.assertEqual(result, "93.184.216.34")

    @patch("socket.getaddrinfo")
    def test_rejects_private_ip(self, mock_getaddrinfo: MagicMock) -> None:
        """Test that private IPs are rejected."""
        mock_getaddrinfo.return_value = [
            (0, 0, 0, 0, ("192.168.1.1", 0, 0, 0)),
        ]
        with self.assertRaises(DisallowedHostError):
            _resolve_public_address("private.example.com")

    @patch("socket.getaddrinfo")
    def test_rejects_loopback(self, mock_getaddrinfo: MagicMock) -> None:
        """Test that loopback is rejected."""
        mock_getaddrinfo.return_value = [
            (0, 0, 0, 0, ("127.0.0.1", 0, 0, 0)),
        ]
        with self.assertRaises(DisallowedHostError):
            _resolve_public_address("localhost")

    @patch("socket.getaddrinfo")
    def test_rejects_multiple_addresses_with_private(self, mock_getaddrinfo: MagicMock) -> None:
        """Test rejection when any resolved address is private."""
        mock_getaddrinfo.return_value = [
            (0, 0, 0, 0, ("93.184.216.34", 0, 0, 0)),
            (0, 0, 0, 0, ("192.168.1.1", 0, 0, 0)),
        ]
        with self.assertRaises(DisallowedHostError):
            _resolve_public_address("multi.example.com")


class RedirectValidationTest(unittest.TestCase):
    """Test redirect validation."""

    @patch("foxnaught.web._follow_redirects")
    @patch("foxnaught.web._resolve_public_address")
    def test_redirect_to_private_blocked(
        self, mock_resolve: MagicMock, mock_follow: MagicMock
    ) -> None:
        """Test that redirects to private addresses are blocked."""
        mock_resolve.side_effect = [
            "93.184.216.34",  # Initial resolution - public
            DisallowedHostError("refusing to fetch non-public address 192.168.1.1"),  # Redirect resolution - private
        ]
        mock_follow.side_effect = FetchHTTPError("test")
        
        result = fetch_url("http://example.com/redirect")
        self.assertIn("Error", result)


if __name__ == "__main__":
    unittest.main()
