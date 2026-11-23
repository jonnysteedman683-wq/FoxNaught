# FoxNaught

A minimal, dependency-free tool-using AI agent for the Mistral API, written in
Python using only the standard library.

## Features

- Agentic tool-use loop: the model can call tools, see their results, and continue.
- Built-in tools: safe arithmetic evaluation (`calculate`) and current UTC time (`now_utc`).
- Pluggable tool registry for adding your own tools.
- Built-in web fetching (`fetch_url`) with SSRF protection against private/non-public hosts.
- Three interfaces: interactive chat CLI, headless one-shot mode, and an HTTP API server.

## Install

No dependencies to install. Requires Python 3.10+ and a Mistral API key.

```bash
export MISTRAL_API_KEY="your-key"
```

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

### HTTP API server

```bash
python -m foxnaught --serve --host 0.0.0.0 --port 8080
```

Endpoints:

- `GET /health` → `{"status": "ok"}`
- `POST /chat` with `{"message": "...", "reset": false}` → `{"reply": "..."}`

### Docker

```bash
docker build -t foxnaught .
docker run -p 8080:8080 -e MISTRAL_API_KEY="your-key" foxnaught
curl -s localhost:8080/health
curl -s -X POST localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "What is the current UTC time?"}'
```

### GitHub Actions

- **CI** (`.github/workflows/ci.yml`): runs the test suite on every push to `main` and pull request.
- **Agent** (`.github/workflows/agent.yml`): runs the agent headlessly via `--one-shot`.
  Trigger it manually from the Actions tab with a prompt and optional model, or enable
  the commented-out `schedule` block in the workflow for recurring runs.

Required repository secret: `MISTRAL_API_KEY` (Settings → Secrets and variables → Actions).
Without it, agent runs fail with a clear error message.

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
- `foxnaught/server.py` — HTTP API (stdlib `http.server`)
- `foxnaught/__main__.py` — CLI entry point (chat, one-shot, serve)
- `Dockerfile` — container image for the HTTP server
