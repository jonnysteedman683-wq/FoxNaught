"""GitHub Actions tools for the agent, backed by the gh CLI."""

from __future__ import annotations

import json
import re

from .gh import GHToolError, MAX_LIMIT, run_gh
from .tools import ToolRegistry

MAX_WORKFLOW_CHARS = 100
MAX_REF_CHARS = 200
MAX_INPUTS = 8
MAX_INPUT_VALUE_CHARS = 2_000
INPUT_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
REF_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
WORKFLOW_FILE_RE = re.compile(r"\.(ya?ml|yml)$", re.IGNORECASE)


def dispatch_workflow(
    workflow: str,
    ref: str = "main",
    inputs: dict[str, str] | None = None,
) -> str:
    """Trigger a repository workflow that has a workflow_dispatch trigger."""
    if not isinstance(workflow, str) or not workflow.strip():
        return "Error: workflow must be a non-empty name, id, or filename"
    if not isinstance(ref, str) or not ref.strip():
        return "Error: ref must be a non-empty branch or tag name"
    if len(workflow) > MAX_WORKFLOW_CHARS or len(ref) > MAX_REF_CHARS:
        return "Error: workflow name or ref is too long"
    if not REF_RE.match(ref):
        return "Error: ref must contain only alphanumeric characters, dots, hyphens, and underscores"
    arguments = ["workflow", "run", workflow.strip(), "--ref", ref.strip()]
    clean_inputs: dict[str, str] = {}
    if inputs:
        if not isinstance(inputs, dict):
            return "Error: inputs must be an object of string key/value pairs"
        for key, value in list(inputs.items())[:MAX_INPUTS]:
            if not isinstance(key, str) or not INPUT_KEY_RE.match(key):
                return f"Error: invalid input name: {key!r}"
            if not isinstance(value, str):
                value = str(value)
            if len(value) > MAX_INPUT_VALUE_CHARS:
                return f"Error: input value for {key!r} exceeds {MAX_INPUT_VALUE_CHARS} characters"
            clean_inputs[key] = value
        for key, value in clean_inputs.items():
            arguments += ["-f", f"{key}={value}"]
    try:
        run_gh(arguments)
    except GHToolError as exc:
        return f"Error: {exc}"
    suffix = f" with inputs {clean_inputs}" if clean_inputs else ""
    return f"dispatched workflow {workflow.strip()} on ref {ref.strip()}{suffix}"


def list_workflow_runs(workflow: str, limit: int = 5) -> str:
    """List recent runs of a repository workflow."""
    if not isinstance(workflow, str) or not workflow.strip():
        return "Error: workflow must be a non-empty name, id, or filename"
    if len(workflow) > MAX_WORKFLOW_CHARS:
        return "Error: workflow name is too long"
    try:
        limit = max(1, min(int(limit), MAX_LIMIT))
    except (TypeError, ValueError):
        return "Error: limit must be an integer"
    try:
        output = run_gh(
            [
                "run",
                "list",
                "--workflow",
                workflow.strip(),
                "--limit",
                str(limit),
                "--json",
                "databaseId,displayTitle,event,status,conclusion,createdAt,url",
            ]
        )
    except GHToolError as exc:
        return f"Error: {exc}"
    try:
        items = json.loads(output)
    except json.JSONDecodeError:
        return "Error: unexpected gh output"
    if not items:
        return f"(no runs for {workflow.strip()})"
    lines = []
    for item in items:
        conclusion = item.get("conclusion") or "—"
        lines.append(
            f"{item['databaseId']} [{item.get('event')}] {item.get('status')}"
            f"/{conclusion} {item.get('createdAt')} — {item.get('url')}"
        )
    return "\n".join(lines)


def view_run(run_id: int) -> str:
    """Show status, conclusion, and per-job summary for one workflow run."""
    try:
        run_id = int(run_id)
    except (TypeError, ValueError):
        return "Error: run_id must be an integer"
    if run_id <= 0:
        return "Error: run_id must be a positive integer"
    try:
        output = run_gh(
            [
                "run",
                "view",
                str(run_id),
                "--json",
                "displayTitle,event,status,conclusion,createdAt,updatedAt,jobs,url",
            ]
        )
    except GHToolError as exc:
        return f"Error: {exc}"
    try:
        run = json.loads(output)
    except json.JSONDecodeError:
        return "Error: unexpected gh output"
    header = (
        f"run {run_id} [{run.get('event')}] {run.get('status')}"
        f"/{run.get('conclusion') or '—'}: {run.get('displayTitle')}"
    )
    job_lines = [
        f"  - {job.get('name')}: {job.get('status')}/{job.get('conclusion') or '—'}"
        for job in run.get("jobs") or []
    ]
    footer = f"  {run.get('url')}" if run.get("url") else ""
    return "\n".join([header, *job_lines, footer]).strip()


def register_actions(registry: ToolRegistry) -> None:
    registry.register(
        name="dispatch_workflow",
        description=(
            "Trigger a repository workflow that has a workflow_dispatch "
            "trigger (use list_workflow_runs or view_run first to confirm "
            "state). Only dispatch workflows when the user's request clearly "
            "calls for it."
        ),
        parameters={
            "type": "object",
            "properties": {
                "workflow": {
                    "type": "string",
                    "description": "Workflow name, id, or filename (e.g. agent.yml).",
                },
                "ref": {
                    "type": "string",
                    "description": "Git branch or tag to run against.",
                    "default": "main",
                },
                "inputs": {
                    "type": "object",
                    "description": "Optional workflow_dispatch inputs as an object.",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["workflow"],
        },
        func=dispatch_workflow,
    )
    registry.register(
        name="list_workflow_runs",
        description="List recent runs of a repository workflow.",
        parameters={
            "type": "object",
            "properties": {
                "workflow": {
                    "type": "string",
                    "description": "Workflow name, id, or filename.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max runs (1-50).",
                    "default": 5,
                },
            },
            "required": ["workflow"],
        },
        func=list_workflow_runs,
    )
    registry.register(
        name="view_run",
        description=(
            "Show status, conclusion, and per-job summary for one workflow "
            "run by its numeric run id (from list_workflow_runs)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "integer",
                    "description": "Numeric workflow run id.",
                },
            },
            "required": ["run_id"],
        },
        func=view_run,
    )
