"""Distilled long-term memory: compacted knowledge refreshed every N runs.

Unlike the verbatim conversation archive, long-term memory holds only a few
bulleted, run-labelled sections the agent itself writes every Nth run.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Callable

DEFAULT_LONGTERM_INTERVAL = 5
DEFAULT_LONGTERM_MAX_CHARS = 20_000
LONGTERM_LINE_CAP = 2_000
LONGTERM_SNIPPET_CHARS = 6_000
SECTION_RE = re.compile(r"^## Run (\d+)")

LONGTERM_MEMO = (
    "Long-term memory: durable, distilled knowledge lives in {path} and a "
    "recent snapshot is included below when present. It is small by design; "
    "the verbatim conversation archive (if any) holds the details. Roughly "
    "every {interval} runs, call the report_longterm tool with a compact "
    "bullet summary of the most important durable facts, decisions, and "
    "lessons learned since your last report. Distil knowledge; never dump "
    "transcripts."
)

LONGTERM_DUE_NOTE = (
    "A long-term memory report is DUE now (run {number}): include a call to "
    "the report_longterm tool in this turn with a short distilled summary."
)

LONGTERM_TOOL_DESCRIPTION = (
    "Record a distilled long-term memory update. Provide a compact bullet "
    "summary of durable facts, decisions, and lessons learned; it becomes a "
    "dated run-labelled section in long-term memory. Call this when the "
    "system prompt says a report is due, roughly every {interval} runs."
)

LONGTERM_TOOL_PARAMETERS: dict = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Compact bullet summary of durable knowledge.",
        },
    },
    "required": ["summary"],
}


def read_longterm(path: str, max_chars: int = LONGTERM_SNIPPET_CHARS) -> str:
    """Return the most recent long-term memory content, newest at the end."""
    if not path or not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    if len(content) <= max_chars:
        return content
    window = content[-max_chars:]
    newline = window.find("\n")
    if newline == -1:
        return window
    tail = window[newline + 1 :]
    return tail if tail.strip() else window


def longterm_due(runs: int, last_report_run: int, interval: int) -> bool:
    """True when at least `interval` runs passed since the last report."""
    return runs - last_report_run >= max(interval, 1)


def append_longterm(
    path: str,
    run_number: int,
    summary: str,
    max_chars: int | None = DEFAULT_LONGTERM_MAX_CHARS,
) -> None:
    """Append a distilled run-labelled section to the long-term memory file."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    lines: list[str] = []
    for raw in (summary or "").splitlines():
        text = raw.strip()
        if not text:
            continue
        if len(text) > LONGTERM_LINE_CAP:
            text = text[:LONGTERM_LINE_CAP] + " …"
        lines.append(text if text.startswith("- ") else f"- {text}")
    if not lines:
        return
    date = datetime.now(timezone.utc).date().isoformat()
    header = "" if os.path.exists(path) else "# Long-term memory\n\n"
    section = f"## Run {run_number} - {date}\n" + "\n".join(lines) + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(header + "\n" + section)
    if max_chars is not None:
        compact_longterm(path, max_chars)


def compact_longterm(path: str, max_chars: int) -> None:
    """Drop the oldest run-labelled sections while the file exceeds max_chars."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    if len(content) <= max_chars:
        return
    preamble: list[str] = []
    sections: list[list[str]] = []
    for line in content.splitlines():
        if SECTION_RE.match(line):
            sections.append([line])
        elif sections:
            sections[-1].append(line)
        else:
            preamble.append(line)
    note = (
        f"\n_(older long-term sections were pruned to stay under {max_chars} "
        f"characters on {datetime.now(timezone.utc).date()})_\n"
    )
    while sections:
        body = "\n\n".join("\n".join(section) for section in sections)
        if len("\n".join(preamble).rstrip()) + len(note) + len(body) + 4 <= max_chars:
            break
        sections.pop(0)
    kept = "\n\n".join("\n".join(section) for section in sections)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(preamble).rstrip() + "\n\n" + note + "\n" + kept + "\n")
    os.replace(tmp_path, path)


def register_longterm_tool(
    registry,
    write_report: Callable[[str], str],
    interval: int = DEFAULT_LONGTERM_INTERVAL,
) -> None:
    """Register the report_longterm tool; the write_report closure records it."""
    registry.register(
        "report_longterm",
        LONGTERM_TOOL_DESCRIPTION.format(interval=interval),
        LONGTERM_TOOL_PARAMETERS,
        func=write_report,
    )
