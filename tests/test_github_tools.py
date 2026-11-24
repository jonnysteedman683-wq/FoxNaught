"""Tests for the gh-backed GitHub tools."""

from __future__ import annotations

import json
import os
import subprocess
import unittest
from unittest import mock

from foxnaught.gh import (
    create_issue,
    list_issues,
    register_github,
    run_gh,
    search_issues,
)
from foxnaught.tools import ToolRegistry


def gh_result(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def issue_json(*numbers: int) -> str:
    return json.dumps(
        [
            {
                "number": n,
                "title": f"Issue {n}",
                "state": "OPEN",
                "createdAt": "2026-11-23T00:00:00Z",
                "url": f"https://github.com/ex/ex/issues/{n}",
            }
            for n in numbers
        ]
    )


class RunGhTest(unittest.TestCase):
    def test_returns_stdout_on_success(self) -> None:
        with (
            mock.patch.dict("os.environ", {"GH_TOKEN": "t"}),
            mock.patch("shutil.which", return_value="/usr/bin/gh"),
            mock.patch("subprocess.run", return_value=gh_result(stdout="ok\n")),
        ):
            self.assertEqual(run_gh(["issue", "list"]), "ok\n")

    def test_accepts_github_token_without_gh_token(self) -> None:
        with (
            mock.patch.dict("os.environ", {"GITHUB_TOKEN": "t"}, clear=True),
            mock.patch("shutil.which", return_value="/usr/bin/gh"),
            mock.patch("subprocess.run", return_value=gh_result(stdout="out")) as run_mock,
        ):
            self.assertEqual(run_gh(["issue", "list"]), "out")
        run_mock.assert_called_once()
    def test_missing_gh_binary_raises(self) -> None:
        with mock.patch.dict("os.environ", {"GH_TOKEN": "t"}), mock.patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(Exception, "not installed"):
                run_gh(["issue", "list"])

    def test_missing_token_raises(self) -> None:
        env = {k: v for k, v in os.environ.items() if k not in ("GH_TOKEN", "GITHUB_TOKEN")}
        with mock.patch.dict("os.environ", env, clear=True):
            with self.assertRaisesRegex(Exception, "no GitHub token"):
                run_gh(["issue", "list"])

    def test_nonzero_exit_includes_stderr(self) -> None:
        with (
            mock.patch.dict("os.environ", {"GH_TOKEN": "t"}),
            mock.patch("shutil.which", return_value="/usr/bin/gh"),
            mock.patch("subprocess.run", return_value=gh_result(stderr="boom", returncode=1)),
        ):
            with self.assertRaises(Exception) as ctx:
                run_gh(["issue", "list"])
        self.assertIn("boom", str(ctx.exception))

    def test_timeout_raises(self) -> None:
        with (
            mock.patch.dict("os.environ", {"GH_TOKEN": "t"}),
            mock.patch("shutil.which", return_value="/usr/bin/gh"),
            mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=[], timeout=60)),
        ):
            with self.assertRaisesRegex(Exception, "timed out"):
                run_gh(["search", "issues"])


class ListIssuesTest(unittest.TestCase):
    def test_formats_issue_lines(self) -> None:
        with mock.patch("foxnaught.gh.run_gh", return_value=issue_json(1, 2)) as run_gh_mock:
            result = list_issues()
        self.assertIn("#1", result)
        self.assertIn("Issue 2", result)
        call = run_gh_mock.call_args.args[0]
        self.assertEqual(call[:3], ["issue", "list", "--state"])
        self.assertIn("open", call)

    def test_rejects_invalid_state(self) -> None:
        self.assertTrue(list_issues(state="not-a-state").startswith("Error:"))

    def test_empty_list_message(self) -> None:
        with mock.patch("foxnaught.gh.run_gh", return_value="[]"):
            self.assertEqual(list_issues(), "(no open issues)")

    def test_gh_tool_errors_surface_as_error_strings(self) -> None:
        from foxnaught.gh import GHToolError

        with mock.patch("foxnaught.gh.run_gh", side_effect=GHToolError("no gh")):
            self.assertTrue(list_issues().startswith("Error: no gh"))

    def test_clamps_limit_and_rejects_non_integer(self) -> None:
        with mock.patch("foxnaught.gh.run_gh", return_value=issue_json(1)) as run_gh_mock:
            list_issues(limit=999)
            call = run_gh_mock.call_args.args[0]
            self.assertEqual(call[call.index("--limit") + 1], "50")
            list_issues(limit=0)
            call = run_gh_mock.call_args.args[0]
            self.assertEqual(call[call.index("--limit") + 1], "1")
            run_gh_mock.reset_mock()
            self.assertEqual(list_issues(limit="nope"), "Error: limit must be an integer")
            run_gh_mock.assert_not_called()

    def test_unexpected_json_is_error(self) -> None:
        with mock.patch("foxnaught.gh.run_gh", return_value="not-json"):
            self.assertEqual(list_issues(), "Error: unexpected gh output")


class SearchIssuesTest(unittest.TestCase):
    def test_passes_query_positionally(self) -> None:
        with mock.patch("foxnaught.gh.run_gh", return_value="[]") as run_gh_mock:
            self.assertEqual(search_issues("flaky test"), "(no matching issues)")
        call = run_gh_mock.call_args.args[0]
        self.assertEqual(call[0], "search")
        self.assertIn("flaky test", call)

    def test_rejects_empty_query(self) -> None:
        self.assertTrue(search_issues("  ").startswith("Error:"))

    def test_formats_results_and_surfaces_tool_error(self) -> None:
        from foxnaught.gh import GHToolError

        with mock.patch("foxnaught.gh.run_gh", return_value=issue_json(3)):
            result = search_issues("flaky test")
        self.assertIn("#3", result)
        self.assertIn("Issue 3", result)
        with mock.patch("foxnaught.gh.run_gh", side_effect=GHToolError("no gh")):
            self.assertEqual(search_issues("q"), "Error: no gh")
        with mock.patch("foxnaught.gh.run_gh", return_value="[]") as run_gh_mock:
            self.assertEqual(search_issues("q", limit="nope"), "Error: limit must be an integer")
        run_gh_mock.assert_not_called()

    def test_unexpected_json_is_error(self) -> None:
        with mock.patch("foxnaught.gh.run_gh", return_value="not-json"):
            self.assertEqual(search_issues("q"), "Error: unexpected gh output")


class CreateIssueTest(unittest.TestCase):
    def test_creates_with_body_file_and_labels(self) -> None:
        with (
            mock.patch("foxnaught.gh.run_gh", return_value="https://github.com/ex/ex/issues/9") as run_gh_mock,
            mock.patch("foxnaught.gh.os.unlink") as unlink,
        ):
            result = create_issue("Bad thing", "It broke", labels=["bug", "ci"])
        self.assertEqual(result, "created issue: https://github.com/ex/ex/issues/9")
        call = run_gh_mock.call_args.args[0]
        self.assertIn("--body-file", call)
        self.assertEqual(call[call.index("--label") + 1], "bug")
        unlink.assert_called_once()

    def test_rejects_empty_title_and_body(self) -> None:
        self.assertTrue(create_issue("  ", "x").startswith("Error:"))
        self.assertTrue(create_issue("t", "").startswith("Error:"))

    def test_caps_title_and_body_length(self) -> None:
        self.assertTrue(create_issue("x" * 300, "b").startswith("Error:"))
        self.assertTrue(create_issue("t", "*" * 30_000).startswith("Error:"))

    def test_ignores_non_string_labels(self) -> None:
        with mock.patch("foxnaught.gh.run_gh", return_value="url") as run_gh_mock:
            create_issue("t", "b", labels=["ok", 5, None])
        call = run_gh_mock.call_args.args[0]
        self.assertEqual(call.count("--label"), 1)

    def test_tool_error_and_body_file_is_unlinked(self) -> None:
        from foxnaught.gh import GHToolError

        def deny(arguments):
            raise GHToolError("denied")

        with mock.patch("foxnaught.gh.run_gh", side_effect=deny) as run_gh_mock:
            result = create_issue("Title", "Body text")
        self.assertEqual(result, "Error: denied")
        run_gh_mock.assert_called_once()
        call = run_gh_mock.call_args.args[0]
        body_file = call[call.index("--body-file") + 1]
        self.assertFalse(os.path.exists(body_file))

    def test_caps_labels_at_five(self) -> None:
        with mock.patch("foxnaught.gh.run_gh", return_value="url") as run_gh_mock:
            create_issue("t", "b", labels=["a", "b", "c", "d", "e", "f"])
        call = run_gh_mock.call_args.args[0]
        positions = [i for i, arg in enumerate(call) if arg == "--label"]
        self.assertEqual(len(positions), 5)
        self.assertEqual([call[i + 1] for i in positions], ["a", "b", "c", "d", "e"])


class RegistrationTest(unittest.TestCase):
    def test_registers_three_tools(self) -> None:
        registry = ToolRegistry()
        register_github(registry)
        self.assertEqual(
            [schema["function"]["name"] for schema in registry.schemas()],
            ["list_issues", "search_issues", "create_issue"],
        )


if __name__ == "__main__":
    unittest.main()
