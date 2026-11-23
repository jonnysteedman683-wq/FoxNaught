"""Tests for sandboxed file tools."""

from __future__ import annotations

import os
import tempfile
import unittest

from foxnaught.files import (
    make_list_files,
    make_read_file,
    make_write_file,
    register_files,
)
from foxnaught.tools import ToolRegistry


class FileToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base = tempfile.mkdtemp()

    def test_read_write_round_trip(self) -> None:
        write = make_write_file(self.base)
        read = make_read_file(self.base)
        self.assertIn("wrote", write("notes/todo.txt", "buy milk"))
        self.assertEqual(read("notes/todo.txt"), "buy milk")

    def test_read_truncates_large_files(self) -> None:
        write = make_write_file(self.base)
        read = make_read_file(self.base)
        write("big.txt", "x" * 70_000)
        content = read("big.txt")
        self.assertIn("[truncated]", content)

    def test_rejects_path_escape(self) -> None:
        write = make_write_file(self.base)
        read = make_read_file(self.base)
        self.assertTrue(write("../../etc/passwd", "x").startswith("Error:"))
        self.assertTrue(read("../../etc/passwd").startswith("Error:"))

    def test_rejects_absolute_escape(self) -> None:
        read = make_read_file(self.base)
        self.assertTrue(read("/etc/passwd").startswith("Error:"))

    def test_list_files_shows_entries_and_subdirs(self) -> None:
        write = make_write_file(self.base)
        write("a.txt", "x")
        write("sub/b.txt", "y")
        listing = make_list_files(self.base)(".")
        self.assertIn("a.txt", listing)
        self.assertIn("sub/", listing)

    def test_list_files_rejects_non_directory(self) -> None:
        write = make_write_file(self.base)
        write("a.txt", "x")
        self.assertTrue(make_list_files(self.base)("a.txt").startswith("Error:"))

    def test_register_files_adds_three_tools(self) -> None:
        registry = ToolRegistry()
        register_files(registry, self.base)
        names = [schema["function"]["name"] for schema in registry.schemas()]
        self.assertEqual(names, ["read_file", "write_file", "list_files"])


if __name__ == "__main__":
    unittest.main()
