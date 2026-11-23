"""File tools confined to a working directory."""

from __future__ import annotations

import os

from .tools import ToolRegistry

MAX_READ_BYTES = 64_000
MAX_WRITE_BYTES = 262_000


class SandboxError(ValueError):
    """Raised when a path escapes the allowed working directory."""


def _resolve_within(base_dir: str, path: str) -> str:
    base = os.path.realpath(base_dir)
    resolved = os.path.realpath(os.path.join(base, path))
    if resolved != base and not resolved.startswith(base + os.sep):
        raise SandboxError(f"path {path!r} escapes the working directory")
    return resolved


def make_read_file(base_dir: str):
    def read_file(path: str) -> str:
        """Read a file from the working directory."""
        try:
            resolved = _resolve_within(base_dir, path)
            with open(resolved, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read(MAX_READ_BYTES + 1)
        except (SandboxError, OSError) as exc:
            return f"Error: {exc}"
        if len(content) > MAX_READ_BYTES:
            content = content[:MAX_READ_BYTES] + "\n…[truncated]"
        return content

    return read_file


def make_write_file(base_dir: str):
    def write_file(path: str, content: str) -> str:
        """Write (or overwrite) a file inside the working directory."""
        if not isinstance(content, str):
            return "Error: content must be a string"
        if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
            return f"Error: content exceeds the {MAX_WRITE_BYTES}-byte limit"
        try:
            resolved = _resolve_within(base_dir, path)
            os.makedirs(os.path.dirname(resolved) or base_dir, exist_ok=True)
            with open(resolved, "w", encoding="utf-8") as fh:
                fh.write(content)
        except (SandboxError, OSError) as exc:
            return f"Error: {exc}"
        return f"wrote {len(content)} characters to {path}"

    return write_file


def make_list_files(base_dir: str):
    def list_files(directory: str = ".") -> str:
        """List files and directories relative to the working directory."""
        try:
            resolved = _resolve_within(base_dir, directory)
        except (SandboxError, OSError) as exc:
            return f"Error: {exc}"
        if not os.path.isdir(resolved):
            return f"Error: {directory} is not a directory"
        entries = sorted(os.listdir(resolved))
        if not entries:
            return "(empty directory)"
        return "\n".join(
            entry + ("/" if os.path.isdir(os.path.join(resolved, entry)) else "")
            for entry in entries
        )

    return list_files


def register_files(registry: ToolRegistry, base_dir: str) -> None:
    registry.register(
        name="read_file",
        description=(
            "Read a text file relative to the working directory. Cannot read "
            "outside the working directory."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path."}
            },
            "required": ["path"],
        },
        func=make_read_file(base_dir),
    )
    registry.register(
        name="write_file",
        description=(
            "Create or overwrite a text file relative to the working directory. "
            "Cannot write outside the working directory."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path."},
                "content": {"type": "string", "description": "Full file content."},
            },
            "required": ["path", "content"],
        },
        func=make_write_file(base_dir),
    )
    registry.register(
        name="list_files",
        description="List entries of a directory relative to the working directory.",
        parameters={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Relative directory path.",
                    "default": ".",
                }
            },
            "required": [],
        },
        func=make_list_files(base_dir),
    )
