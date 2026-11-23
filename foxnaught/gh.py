"""GitHub tools for the agent, backed by the gh CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

from .tools import ToolRegistry

GH_TIMEOUT_SECONDS = 60
MAX_BODY_CHARS = 20_000
MAX_TITLE_CHARS = 200
MAX_LIMIT = 50
MAX_LABELS = 5
STATE_CHOICES = ("open", "closed", "all")


class GHToolError(RuntimeError):
    """Raised when the gh CLI is missing, unauthenticated, or fails."""


def run_gh(arguments: list[str]) -> str:
    gh_path = shutil.which("gh")
    if gh_path is None:
        raise GHToolError("the 'gh' CLI is not installed or not on PATH")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise GHToolError("no GitHub token: set the GH_TOKEN environment variable")
    try:
        completed = subprocess.run(
            [gh_path, *arguments],
            capture_output=True,
            text=True,
            timeout=GH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise GHToolError(f"gh timed out after {GH_TIMEOUT_SECONDS}s") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GHToolError(f"gh failed: {detail}")
    return completed.stdout


def list_issues(state: str = "open", limit: int = 20) -> str:
    """List repository issues as formatted lines."""
    if state not in STATE_CHOICES:
        return f"Error: state must be one of {STATE_CHOICES}"
    try:
        limit = max(1, min(int(limit), MAX_LIMIT))
    except (TypeError, ValueError):
        return "Error: limit must be an integer"
    try:
        output = run_gh(
            [
                "issue",
                "list",
                "--state",
                state,
                "--limit",
                str(limit),
                "--json",
                "number,title,state,createdAt,url",
            ]
        )
    except GHToolError as exc:
        return f"Error: {exc}"
    try:
        items = json.loads(output)
    except json.JSONDecodeError:
        return "Error: unexpected gh output"
    if not items:
        return f"(no {state} issues)"
    return "\n".join(
        f"#{item['number']} [{item['state']}] {item['title']} — {item['url']}"
        for item in items
    )


def search_issues(query: str, limit: int = 20) -> str:
    """Search issues matching a query."""
    if not isinstance(query, str) or not query.strip():
        return "Error: query must be a non-empty string"
    try:
        limit = max(1, min(int(limit), MAX_LIMIT))
    except (TypeError, ValueError):
        return "Error: limit must be an integer"
    try:
        output = run_gh(
            [
                "search",
                "issues",
                "--limit",
                str(limit),
                "--json",
                "number,title,state,url",
                query,
            ]
        )
    except GHToolError as exc:
        return f"Error: {exc}"
    try:
        items = json.loads(output)
    except json.JSONDecodeError:
        return "Error: unexpected gh output"
    if not items:
        return "(no matching issues)"
    return "\n".join(
        f"#{item['number']} [{item['state']}] {item['title']} — {item['url']}"
        for item in items
    )


def create_issue(title: str, body: str, labels: list[str] | None = None) -> str:
    """Create a repository issue."""
    if not isinstance(title, str) or not title.strip():
        return "Error: title must be a non-empty string"
    if not isinstance(body, str) or not body:
        return "Error: body must be a non-empty string"
    if len(title) > MAX_TITLE_CHARS:
        return f"Error: title exceeds {MAX_TITLE_CHARS} characters"
    if len(body) > MAX_BODY_CHARS:
        return f"Error: body exceeds {MAX_BODY_CHARS} characters"
    arguments = ["issue", "create", "--title", title.strip()]
    body_file = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".md", delete=False
    )
    try:
        body_file.write(body)
        body_file.close()
        arguments += ["--body-file", body_file.name]
        clean_labels = [
            label.strip()
            for label in (labels or [])
            if isinstance(label, str) and label.strip()
        ][:MAX_LABELS]
        for label in clean_labels:
            arguments += ["--label", label]
        url = run_gh(arguments).strip()
    except GHToolError as exc:
        return f"Error: {exc}"
    finally:
        os.unlink(body_file.name)
    return f"created issue: {url}"


def register_github(registry: ToolRegistry) -> None:
    registry.register(
        name="list_issues",
        description="List issues in the current repository. Use to review open work.",
        parameters={
            "type": "object",
            "properties": {
                "state": {
                    "type": "string",
                    "description": "open, closed, or all",
                    "enum": list(STATE_CHOICES),
                    "default": "open",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max issues (1-50).",
                    "default": 20,
                },
            },
            "required": [],
        },
        func=list_issues,
    )
    registry.register(
        name="search_issues",
        description=(
            "Search issues matching a query. ALWAYS call this before create_issue "
            "to avoid duplicates."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text."},
                "limit": {
                    "type": "integer",
                    "description": "Max results (1-50).",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
        func=search_issues,
    )
    registry.register(
        name="create_issue",
        description=(
            "Create one repository issue. Check search_issues first; file one issue "
            "per distinct problem with a concise title and informative body. Never "
            "create duplicate issues."
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Concise issue title."},
                "body": {"type": "string", "description": "Issue body in Markdown."},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional labels for the issue.",
                },
            },
            "required": ["title", "body"],
        },
        func=create_issue,
    )
