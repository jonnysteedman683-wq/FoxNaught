# Connecting an external bot (e.g. Grok) to FoxNaught

FoxNaught is driven entirely through GitHub, so any assistant that can act on
this repository — Grok Bot, a CI chatbot, a scheduled script — already has
everything it needs. This page explains the three ways to wire one up, from
least to most privileged, plus the security posture we recommend.

## Option 1 — let the bot leave instructions (recommended)

The bot needs **only `issues: write`**. It posts a comment starting with
`/foxnaught` on any issue, and the Assistant workflow picks it up, does the
work (dispatch workflows, inspect runs, manage issues), and replies on the
thread. This keeps the bot's permissions minimal and leaves an audit trail.

**Who is allowed to trigger it:** the repository owner, any repository
collaborator, or any GitHub login listed (comma-separated) in the
`ASSISTANT_ALLOWED_LOGINS` repository variable
(Settings → Secrets and variables → Actions → Variables).
Bot accounts (`…[bot]` logins) are ignored unless explicitly allow-listed —
this stops strangers or runaway automations from spending your Mistral API
quota or launching workflows via prompt injection. To allow the Grok bot's
own GitHub identity, find its login (visible in the header of any comment it
posts) and add it to that variable.

In your bot's instructions/skill, give it a task like:

> To run FoxNaught workflows, comment `/foxnaught <task>` on an existing
> FoxNaught issue (e.g. the Assistant smoke-test issue). Reply with the link
> to any comment the Assistant posts back, and to any workflow run it
> mentions. Never push to `main`, never edit workflow files, and never use
> `actions` or `administration` permissions.

## Option 2 — let the bot dispatch workflows itself

If the bot's GitHub connection comes with `actions: write` (the Grok
connector requests very broad permissions by default), it can trigger the
Agent workflow directly via the API:

```bash
curl -X POST \
  -H "Authorization: Bearer $BOT_GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/jonnysteedman683-wq/FoxNaught/actions/workflows/agent.yml/dispatches \
  -d '{
    "ref": "main",
    "inputs": {
      "prompt": "Summarise notable changes since yesterday and update REPORT.md.",
      "continue_from": "previous",
      "max_steps": "16"
    }
  }'
```

Note that Option 2 bypasses the Assistant entirely, so there is no
natural-language arbitration, no run inspection, and no reply on a thread —
prefer Option 1 unless the bot needs fire-and-forget dispatches from outside
GitHub.

## Option 3 — the HTTP server (external chat clients)

For chat clients that speak plain JSON rather than GitHub-native actions,
the packaged HTTP server (`python -m foxnaught --serve`) exposes `POST /chat`
(with optional bearer-token auth, streaming SSE, and persistent memory).
This is still awaiting a deployment decision (Fly.io / Render / a VM); the
relevant config keys are `MISTRAL_API_KEY` and `FOXNAUGHT_AUTH_TOKEN`.

## Security posture (applies to any external bot)

- **Least privilege first.** A connector that only needs to comment should
  only have `issues: write`. Audit which repositories the connector covers —
  scope it to FoxNaught rather than every repo you own.
- **Bots are not a security boundary.** Anyone who can influence what the
  bot reads (issue comments, `REPORT.md`, prompt inputs) can attempt prompt
  injection. The Assistant's authorisation gate is the main defence here;
  keep it enabled.
- **Secrets stay in GitHub.** `MISTRAL_API_KEY` and `FOXNAUGHT_AUTH_TOKEN`
  live in repository secrets, never in bot configuration, chat history, or
  the repo itself. Never paste a key into an issue comment — if you did,
  rotate it immediately.
- **Every dispatch is auditable.** All Agent/Assistant runs are visible in
  the Actions tab, and the watchdog files an issue for every failure.
