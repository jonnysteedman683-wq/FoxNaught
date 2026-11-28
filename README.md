# FoxNaught

A minimal, dependency-free tool-using AI agent for the Mistral API, written in
Python using only the standard library.

## Features

- Agentic tool-use loop: the model can call tools, see their results, and continue.
- Built-in tools: safe arithmetic evaluation (`calculate`) and current UTC time (`now_utc`).
- Streaming replies (`--stream`) via SSE deltas from the Mistral API.
- Persistent conversation memory (`--memory FILE`) that survives between one-shot runs.
- Three-tier memory: hot conversation, verbatim archive, and a distilled long-term
  memory the agent reports to every N runs (`--longterm FILE`).
- GitHub integration: issue tools, workflow dispatch/run-inspection tools, and a
  `/foxnaught` comment-triggered Assistant workflow.
- Distinct agent roles via `--system-prompt` (one codebase, many agents).
- Pluggable tool registry for adding your own tools.
- Built-in web fetching (`fetch_url`) with SSRF protection against private/non-public
  hosts: connections are pinned to the validated IP address (with SNI kept on the
  hostname, defeating DNS rebinding) and every redirect target is re-validated.
- Sandboxed file tools (`read_file`, `write_file`, `list_files`) confined to the working directory.
- GitHub tools (`list_issues`, `search_issues`, `create_issue`) backed by the `gh` CLI, with duplicate-issue avoidance baked into the tool guidance.
- Three interfaces: interactive chat CLI, headless one-shot mode, and an HTTP API server.

## Install

No dependencies to install. Requires Python 3.10+ and a Mistral API key.

```bash
export MISTRAL_API_KEY="your-key"
```

### Model Configuration

The default model is `mistral-small-latest`. You can customize it in several ways:

- CLI flag: `--model mistral-large-latest`
- Environment variable: `MISTRAL_MODEL=mistral-large-latest`
- In code: `MistralClient(model="mistral-large-latest")`

Available models: `mistral-tiny-latest`, `mistral-small-latest`, `mistral-medium-latest`, `mistral-large-latest`, and any custom model endpoints you have access to.

### Transient failures

The client retries rate limits (429), server errors (5xx), and dropped
connections up to two times with jittered backoff. Other 4xx responses fail
immediately because retrying them cannot help. When the API sends a
`Retry-After` header (seconds or an HTTP date), the client honours it up to a
30-second cap instead of using its own backoff. Pass
`MistralClient(retries_on_rate_limit=False)` to disable all retries.

For local use of the GitHub tools, install the [gh CLI](https://cli.github.com/) and authenticate (`gh auth login`) or set `GH_TOKEN` to a token with repository access.

## Usage

```bash
python -m foxnaught                 # default model: mistral-small-latest
python -m foxnaught --model mistral-large-latest
python -m foxnaught --max-steps 12  # tool steps allowed per turn
```

In the chat, type `exit` to quit or `/reset` to clear the conversation.

### Headless one-shot mode

```bash
python -m foxnaught --one-shot "What is 6 * 7?"
```

With streaming and persistent memory:

```bash
python -m foxnaught --stream --memory foxnaught-history.json --one-shot "Hello again!"
```

The memory file stores the full conversation as JSON; a new run resumes the
previous conversation (including tool results). Delete the file or `/reset` to
start fresh.

### HTTP API server

```bash
python -m foxnaught --serve --host 0.0.0.0 --port 8080
```

Endpoints:

- `GET /health` → `{"status": "ok"}`
- `POST /chat` with `{"message": "...", "reset": false}` → `{"reply": "..."}`
- `POST /chat` with `"stream": true` → server-sent events: `data: {"delta": ...}` per
  reply fragment, a final `data: {"reply": "..."}`, then `data: [DONE]`; errors
  arrive as `data: {"error": "..."}`

With `--serve --memory FILE` the server persists the conversation after every
request (optionally `"reset": true` to wipe it), so restarts keep their memory.

### Authentication

`POST /chat` can require a bearer token. Pass it via `--auth-token` (or the
`FOXNAUGHT_AUTH_TOKEN` environment variable); `/health` stays open in either
case. When no token is configured the server is unauthenticated — do not bind
it publicly in this mode. Connections have a 30-second read timeout, so a
client that stalls mid-request cannot hold a worker thread indefinitely; a
body that times out is answered with 400.

```bash
python -m foxnaught --serve --host 0.0.0.0 \
  --auth-token "$(openssl rand -hex 32)"
curl -s -X POST localhost:8080/chat \
  -H "Authorization: Bearer $FOXNAUGHT_AUTH_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"message": "hi"}'
```

### Docker

The image runs as a non-root user (`foxnaught`), so mount a directory rather
than a single file if you want the container to rewrite the memory file.

**Important:** For persistent memory across container restarts, mount the memory file:

```bash
docker build -t foxnaught .
docker run -p 8080:8080 \
  -e MISTRAL_API_KEY="your-key" \
  -e FOXNAUGHT_AUTH_TOKEN="shared-secret-here" \
  -e MISTRAL_MODEL="mistral-small-latest" \
  -v "$PWD/agent-memory.json:/app/agent-memory.json" \
  -v "$PWD/memory-archive:/app/memory-archive" \
  foxnaught
```

**Docker Compose example:**

```yaml
version: '3.8'
services:
  foxnaught:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MISTRAL_API_KEY=${MISTRAL_API_KEY}
      - FOXNAUGHT_AUTH_TOKEN=${FOXNAUGHT_AUTH_TOKEN}
      - MISTRAL_MODEL=mistral-small-latest
    volumes:
      - ./agent-memory.json:/app/agent-memory.json
      - ./memory-archive:/app/memory-archive
    restart: unless-stopped
```

Verify with:
```bash
curl -s localhost:8080/health
curl -s -X POST localhost:8080/chat \
  -H "Authorization: Bearer shared-secret-here" \
  -H 'Content-Type: application/json' \
  -d '{"message": "What is the current UTC time?"}'
```

### GitHub Actions

- **CI** (`.github/workflows/ci.yml`): runs the test suite on every push to `main` and pull request.
- **Agent** (`.github/workflows/agent.yml`): runs daily at 06:00 UTC and on manual dispatch.
  Trigger it manually from the Actions tab with:
  - **prompt** — what the agent should do (default: review the repo and write `REPORT.md`)
  - **continue_from** — `previous` (default) resumes the last run's conversation via the `agent-memory` artifact; `fresh` starts clean
  - **max_steps** — maximum tool steps per turn (default 16; raise it for prompts that need to read a lot of the repo)
  - **commit_report** — opt-in commit of `REPORT.md` back to `main`
  Both `REPORT.md` and the conversation memory (including the verbatim archive
  and distilled long-term memory) are uploaded as run artifacts.
  The workflow is scheduled daily at 06:00 UTC (GitHub may delay scheduled runs
  by up to an hour on low-traffic repositories).

  **Required permissions:** `contents: read`, `contents: write` (for commit_report), `issues: write` (for create_issue tool)
  **Required secrets:** `MISTRAL_API_KEY`

- **Assistant** (`.github/workflows/assistant.yml`): a second, distinct agent
  specialised in operating workflows. Two triggers:
  - comment `/foxnaught <task>` on any issue (e.g. `/foxnaught check the last Agent run and re-run it if it failed`) — it interprets the task, uses `dispatch_workflow` / `list_workflow_runs` / `view_run`, and replies on the thread with its transcript
  - manual dispatch with a **prompt** input from the Actions tab
  It has its own conversation memory (`assistant-memory` artifact), a
  dedicated system prompt (see `--system-prompt`), and comments made by bots
  do not re-trigger it. Requires `MISTRAL_API_KEY`; the `actions: write`
  permission is what lets it trigger workflows. Triggering is gated to the
  repository owner, collaborators, or the logins in the
  `ASSISTANT_ALLOWED_LOGINS` repository variable — see
  [docs/grok-bot-integration.md](docs/grok-bot-integration.md) for wiring up
  external bots like Grok.

  **Required permissions:** `contents: read`, `issues: write`, `actions: write`
  **Required secrets:** `MISTRAL_API_KEY`
  **Optional variables:** `ASSISTANT_ALLOWED_LOGINS` (comma-separated list of GitHub logins)

- **Watchdog** (`.github/workflows/watchdog.yml`): two guards covering both
  agents. The `workflow_run` trigger fires on any failed Agent or Assistant
  run and opens an issue titled `<Workflow> run <id> failed` with the run
  link and failed job names (deduplicated per run). A daily 08:00 UTC
  schedule check files an `Agent missed its daily schedule (<date>)` issue
  when no Agent run has fired in the last 26 hours — GitHub sometimes
  silently skips scheduled runs on low-activity repositories, and this
  catches it the next morning.

  **Required permissions:** `actions: read`, `issues: write`

### GitHub Tool Permissions

The GitHub tools (`list_issues`, `search_issues`, `create_issue`, `dispatch_workflow`, 
`list_workflow_runs`, `view_run`) require specific permissions:

| Tool | Required Permission | Required Secret/Variable |
|---|---|---|
| `list_issues` | `issues: read` | `GH_TOKEN` |
| `search_issues` | `issues: read` | `GH_TOKEN` |
| `create_issue` | `issues: write` | `GH_TOKEN` |
| `dispatch_workflow` | `actions: write` | `GH_TOKEN` |
| `list_workflow_runs` | `actions: read` | `GH_TOKEN` |
| `view_run` | `actions: read` | `GH_TOKEN` |

**Setup:**
1. In your repository, go to **Settings > Secrets and variables > Actions**
2. Add a **Repository Secret** named `MISTRAL_API_KEY` with your Mistral API key
3. Add a **Repository Secret** named `GH_TOKEN` with a GitHub Personal Access Token (PAT)
   - The PAT needs the permissions listed above based on which tools you use
   - For full functionality, use: `repo`, `issues: write`, `actions: write`
4. Optionally, add a **Repository Variable** named `ASSISTANT_ALLOWED_LOGINS` with comma-separated GitHub logins that can trigger the Assistant via comments

**Note:** The `gh` CLI must be installed in your environment. The Agent and Assistant workflows include it via `actions/setup-python` and the pre-installed `gh` CLI in GitHub Actions runners.

HTTP 429 responses are retried with backoff (1 s, then 2 s) with ±50 % jitter
so many clients retrying in sync do not create a thundering herd.

Conversation history is trimmed to the most recent messages (default 20, see
`--history-limit`) so long-running memories cannot exceed the model's context
window; the system prompt is always kept.

**Memory archive**: `--memory-archive FILE` appends every trimmed message to a
markdown file. When set, the system prompt gains a note telling the agent the
archive exists so it can `read_file` past turns that left the context window.
The Agent workflow passes `--memory-archive MEMORY.md` and ships the archive
inside the `agent-memory` artifact so it accumulates across daily runs. When
the archive exceeds `--archive-max-chars` (default 100,000, `0` disables), the
oldest whole entries are pruned so the file fits the limit again, leaving a
single inline note; per-message content is capped at 2,000 characters.

**Long-term memory**: `--longterm FILE` maintains a distilled, compacted
knowledge file that is formatted differently from the archive: dated,
run-labelled sections of bullets the agent itself writes. Every
`--longterm-interval` runs (default 5), the system prompt flags that a report
is due and the agent calls the `report_longterm` tool with a short summary of
durable facts, decisions, and lessons learned — never transcripts. The
snapshot is injected into the system prompt each turn, so the agent always
remembers its long-term conclusions even after the conversation and archive
rolled past them. When the file exceeds `--longterm-max-chars` (default
20,000, `0` disables), the oldest sections are pruned. The Agent workflow
passes `--longterm LONGTERM.md` and ships it inside the `agent-memory`
artifact; note that a `continue_from: fresh` run skips the whole restore and
therefore starts without long-term memory too.

**Live integration tests**: `tests/test_live.py` performs real API round trips
(tool call, streaming, and memory) and runs automatically in the `live-test` CI
job when `MISTRAL_API_KEY` is set; without the secret they are skipped.

### GitHub tools

With `gh` installed and `GH_TOKEN` set (the Agent workflow provides both), the
agent can list, search, and create repository issues. Trigger the Agent
workflow with a prompt like:

> Review the repository, then create issues for any distinct problems you find
> (search first to avoid duplicates; label them if useful).

The `create_issue` tool is instructed to search for duplicates first and file
one issue per distinct problem. Issue creation requires the workflow job's
`issues: write` permission, already configured.

Required repository secret: `MISTRAL_API_KEY` (Settings → Secrets and variables → Actions).
Without it, agent runs fail with a clear error message.

## Tool catalog

| Tool | Purpose | Requirements | Notes |
|---|---|---|---|
| `calculate` | Safe arithmetic evaluation | — | Supports +, -, *, /, //, %, **, parentheses, constants pi/e/tau, functions abs, round, min, max, sqrt, log, exp, sin, cos, tan. Rejects anything non-mathematical; expressions are capped at 1000 characters, exponents at 4096, and integer results at 8192 bits to prevent resource exhaustion |
| `now_utc` | Current UTC time, ISO-8601 | — | UTC only; timezone parameter must be "UTC" |
| `fetch_url` | Fetch a public web page as text | network | SSRF-protected; blocks private/loopback/reserved/multicast hosts; redirects re-validated (max 5), connection pinned to the resolved IP; max 512KB download, 4000 char return |
| `read_file` | Read a text file | — | Relative to the working directory; 64 KB cap; cannot escape sandbox |
| `write_file` | Create/overwrite a text file | — | Confined to the working directory; 256 KB cap; creates parent directories |
| `list_files` | List a directory | — | Confined to the working directory; shows files and subdirectories |
| `list_issues` | List repository issues | `gh` + `GH_TOKEN` | Backed by GitHub CLI; returns title, number, labels |
| `search_issues` | Search existing issues | `gh` + `GH_TOKEN` | Call before `create_issue` to avoid duplicates |
| `create_issue` | File a repository issue | `gh` + `GH_TOKEN` | 200-char title, 20 K-char body caps; requires `issues: write` permission |
| `dispatch_workflow` | Trigger a workflow with a `workflow_dispatch` trigger | `gh` + `GH_TOKEN` (+ `actions: write` in CI) | Input names validated (alphanumeric + _.-); max 8 inputs, 2 K-char values each |
| `list_workflow_runs` | List recent runs of a workflow | `gh` + `GH_TOKEN` | 1–50 runs; returns ID, event, status, conclusion |
| `view_run` | Status, conclusion, and per-job summary of one run | `gh` + `GH_TOKEN` | Use with `list_workflow_runs`; shows job-level status |
| `report_longterm` | Write a distilled long-term memory section | `--longterm FILE` | Prompted when due, every N runs (default 5) |

## Debugging

### Tool Call Logging

When running the agent, tool calls are logged to stdout:
```
[tool] fetch_url({"url": "https://example.com"})
```

### Error Messages

All tools return descriptive error messages when they fail:
- Invalid arguments: `Error: <description>`
- Sandbox violations: `Error: path escapes the working directory`
- Network errors: `Error: fetch failed: <reason>`
- SSRF protection: `Error: refusing to fetch non-public address <IP>`

### Verbose Mode

To debug API calls, set the `FOXNAUGHT_DEBUG` environment variable:
```bash
FOXNAUGHT_DEBUG=1 python -m foxnaught --one-shot "Hello"
```

This will print detailed information about HTTP requests and responses.

Tool execution is visible as `[tool] name(arguments)` lines in CLI and workflow logs.

## Memory file format

The `--memory FILE` file is a JSON object holding run bookkeeping plus the
conversation:

```json
{
  "runs": 12,
  "last_report_run": 10,
  "messages": [
    {"role": "system", "content": "You are FoxNaught..."},
    {"role": "user", "content": "What is 6 * 7?"},
    {"role": "assistant", "content": "42"}
  ]
}
```

**Fields:**
- `runs`: Total number of agent runs (turns) in this session
- `last_report_run`: Last run number when long-term memory was reported
- `messages`: Array of conversation messages with `role`, `content`, and optionally `tool_calls`/`tool_call_id`

The older bare-array format is still loaded for backwards compatibility
(runs counters then start from zero).

The whole file is validated before anything is loaded: `runs` and
`last_report_run` must be non-negative integers, each message needs a `role` of
`system`, `user`, `assistant` or `tool`, and `content`, `name`, `tool_call_id`
must be strings and `tool_calls` a list of objects when present. A malformed
file (bad JSON, non-UTF-8 bytes or a bad field) raises `ValueError` and leaves
the agent untouched; `--serve` then starts with a fresh conversation.

Messages also carry `tool_calls` / `tool_call_id` when tools were used, matching
the Mistral chat-completions wire format. Files are written atomically (via
`.tmp` files and `os.replace`), so a crash mid-write cannot corrupt them.

**Memory Archive:** When `--memory-archive FILE` is enabled, trimmed messages are
appended to a markdown file. Each entry includes the role and content (truncated
to 2000 chars). Tool calls are noted with their names.

**Long-term Memory:** When `--longterm FILE` is enabled, the agent is prompted
to write distilled summaries every N runs (default 5). These are stored as
dated, run-labelled sections in a markdown file and injected into the system
prompt for context across long-running conversations.

## Adding a tool

```python
from foxnaught.agent import Agent
from foxnaught.builtins import register_defaults
from foxnaught.client import MistralClient
from foxnaught.tools import ToolRegistry

registry = ToolRegistry()
register_defaults(registry)

registry.register(
    name="word_count",
    description="Count words in a text.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    func=lambda text: str(len(text.split())),
)

agent = Agent(MistralClient(), registry)
print(agent.run_turn("How many words are in 'the quick brown fox'?"))
```

Tool functions receive keyword arguments parsed from the model's JSON
arguments. Return a string (or anything JSON-serialisable); exceptions are
caught and reported back to the model as `Error: ...` messages so the agent can
recover.

## Tests

```bash
python -m unittest discover -s tests
```

## Layout

- `foxnaught/agent.py` — the chat/tool-call loop
- `foxnaught/client.py` — Mistral chat-completions client (stdlib `urllib`)
- `foxnaught/tools.py` — tool registry and JSON-schema tool definitions
- `foxnaught/builtins.py` — built-in tools
- `foxnaught/web.py` — web-fetch tool with SSRF protection (stdlib `urllib`)
- `foxnaught/files.py` — sandboxed file tools (locked to the working directory)
- `foxnaught/gh.py` — GitHub tools via the `gh` CLI (needs `GH_TOKEN` and `gh` installed)
- `foxnaught/server.py` — HTTP API (stdlib `http.server`)
- `foxnaught/__main__.py` — CLI entry point (chat, one-shot, serve)
- `Dockerfile` — container image for the HTTP server
