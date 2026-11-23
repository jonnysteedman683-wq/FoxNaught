"""Tests for the gh-backed GitHub Actions tools."""

from __future__ import annotations

import json
import json
import unittest
from unittest import mock

from foxnaught.actions import (
    dispatch_workflow,
    list_workflow_runs,
    register_actions,
    view_run,
)
from foxnaught.gh import GHToolError
from foxnaught.tools import ToolRegistry


def runs_json(*run_ids: int) -> str:
    return json.dumps(
        [
            {
                "databaseId": n,
                "displayTitle": f"Run {n}",
                "event": "workflow_dispatch",
                "status": "completed",
                "conclusion": "success" if n % 2 else "failure",
                "createdAt": "2026-11-23T05:00:00Z",
                "url": f"https://github.com/ex/ex/actions/runs/{n}",
            }
            for n in run_ids
        ]
    )


def view_json(run_id: int) -> str:
    return json.dumps(
        {
            "displayTitle": f"Run {run_id}",
            "event": "schedule",
            "status": "completed",
            "conclusion": "failure",
            "createdAt": "2026-11-23T05:00:00Z",
            "updatedAt": "2026-11-23T05:01:00Z",
            "url": f"https://github.com/ex/ex/actions/runs/{run_id}",
            "jobs": [
                {"name": "run-agent", "status": "completed", "conclusion": "failure"},
                {"name": "upload", "status": "completed", "conclusion": "skipped"},
            ],
        }
    )


class DispatchWorkflowTest(unittest.TestCase):
    def test_builds_gh_command_with_inputs(self) -> None:
        with mock.patch("foxnaught.actions.run_gh", return_value="") as gh:
            result = dispatch_workflow("agent.yml", "main", {"prompt": "do it", "model": "mistral-small-latest"})
        self.assertTrue(result.startswith("dispatched workflow agent.yml on ref main"))
        self.assertIn("{'prompt': 'do it', 'model': 'mistral-small-latest'}", result)
        args = gh.call_args.args[0]
        self.assertEqual(args[:3], ["workflow", "run", "agent.yml"])
        self.assertIn("--ref", args)
        self.assertIn("-f", args)
        self.assertIn("prompt=do it", args)
        self.assertIn("model=mistral-small-latest", args)

    def test_rejects_empty_workflow(self) -> None:
        self.assertTrue(dispatch_workflow("").startswith("Error:"))

    def test_rejects_invalid_input_key(self) -> None:
        with mock.patch("foxnaught.actions.run_gh", return_value="") as gh:
            result = dispatch_workflow("agent.yml", "main", {"bad key!": "v"})
        self.assertTrue(result.startswith("Error: invalid input name"))
        gh.assert_not_called()

    def test_caps_input_count(self) -> None:
        inputs = {f"key{i}": "v" for i in range(12)}
        with mock.patch("foxnaught.actions.run_gh", return_value="") as gh:
            dispatch_workflow("agent.yml", "main", inputs)
        args = gh.call_args.args[0]
        self.assertEqual(args.count("-f"), 8)

    def test_tool_error_surfaces_as_error_string(self) -> None:
        with mock.patch("foxnaught.actions.run_gh", side_effect=GHToolError("denied")):
            result = dispatch_workflow("agent.yml")
        self.assertEqual(result, "Error: denied")


class ListRunsTest(unittest.TestCase):
    def test_formats_run_lines(self) -> None:
        with mock.patch("foxnaught.actions.run_gh", return_value=runs_json(1, 2)) as gh:
            result = list_workflow_runs("agent.yml")
        self.assertIn("1 [workflow_dispatch] completed/success", result)
        self.assertIn("2 [workflow_dispatch] completed/failure", result)
        args = gh.call_args.args[0]
        self.assertEqual(args[:3], ["run", "list", "--workflow"])

    def test_empty_list_message(self) -> None:
        with mock.patch("foxnaught.actions.run_gh", return_value="[]"):
            self.assertEqual(list_workflow_runs("ci.yml"), "(no runs for ci.yml)")

    def test_rejects_non_integer_limit(self) -> None:
        self.assertEqual(list_workflow_runs("ci.yml", limit="x"), "Error: limit must be an integer")

    def test_tool_error_surfaces_as_error_string(self) -> None:
        with mock.patch("foxnaught.actions.run_gh", side_effect=GHToolError("boom")):
            self.assertEqual(list_workflow_runs("ci.yml"), "Error: boom")


class ViewRunTest(unittest.TestCase):
    def test_includes_header_and_jobs(self) -> None:
        with mock.patch("foxnaught.actions.run_gh", return_value=view_json(42)):
            result = view_run(42)
        self.assertIn("run 42 [schedule] completed/failure", result)
        self.assertIn("run-agent: completed/failure", result)
        self.assertIn("upload: completed/skipped", result)
        self.assertIn("https://github.com/ex/ex/actions/runs/42", result)

    def test_rejects_non_integer_run_id(self) -> None:
        self.assertEqual(view_run("abc"), "Error: run_id must be an integer")

    def test_rejects_non_positive_run_id(self) -> None:
        self.assertEqual(view_run(0), "Error: run_id must be a positive integer")

    def test_missing_run_surfaces_as_error_string(self) -> None:
        with mock.patch("foxnaught.actions.run_gh", side_effect=GHToolError("not found")):
            self.assertEqual(view_run(999), "Error: not found")


class RegisterActionsTest(unittest.TestCase):
    def test_registers_three_tools(self) -> None:
        registry = ToolRegistry()
        register_actions(registry)
        schemas = registry.schemas()
        names = {schema["function"]["name"] for schema in schemas}
        self.assertEqual(names, {"dispatch_workflow", "list_workflow_runs", "view_run"})

    def test_dispatch_runs_through_registry(self) -> None:
        registry = ToolRegistry()
        register_actions(registry)
        with mock.patch("foxnaught.actions.run_gh", return_value="") as gh:
            result = registry.run("dispatch_workflow", json.dumps({"workflow": "agent.yml"}))
        self.assertTrue(result.startswith("dispatched workflow agent.yml"))
        gh.assert_called_once()


if __name__ == "__main__":
    unittest.main()
