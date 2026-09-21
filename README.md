# Agent

`agent` is a local terminal coding agent. It turns a user request into a
controlled loop of model calls, tool calls, permission decisions, sandboxed
execution, durable events, and verification.

It is being built as an OpenAI-first alpha. The design keeps model providers,
tools, UI transports, and persistence separate so the system can grow without
rewriting its core loop.

## Architecture at a glance

```text
                    Interactive path

  User
   |
   v
  Textual TUI ----> RPC client ----> JSON-RPC server
                                      |
                                      v
                              Durable session runtime
                                      |
                                      v
  Provider <---- Agent loop ----> Tool dispatcher
     |              |                    |
     |              |                    v
     |              |             Permissions
     |              |                    |
     v              v                    v
  Streamed       Normalized          Sandbox + tools
  provider       events                   |
  events <--------------------------------+
                  tool results

                    Benchmark path

  Task catalog -> fresh fixture copy -> headless agent attempt
                      |                         |
                      v                         v
               changed-path check         JSONL trajectory
                      |
                      v
          Docker verifier (network disabled) -> scoreboard
```

## One normal agent turn

1. The user submits a prompt through the CLI, TUI, or JSON-RPC client.
2. `core/loop.py` builds a normalized provider request from the durable
   conversation history.
3. A provider streams text, thinking, tool calls, usage, and errors.
4. The provider adapter converts that provider-specific stream into shared
   events from `events.py`.
5. The agent loop forwards requested tools to the dispatcher.
6. The dispatcher checks the permission policy before any side effect.
7. Allowed tools run inside the workspace boundary and, for shell commands,
   the configured sandbox.
8. Tool results return to the loop as new events; the model receives them on
   the next turn and can continue or finish.
9. The TUI renders the same events as transcript text, tool activity, cost,
   status, and approval dialogs.

The event model is the contract between all of these layers. A UI does not
need to understand OpenAI wire JSON, and a provider does not need to know how
SQLite or a diff viewer works.

## Component map

| Area | Main location | Responsibility |
| --- | --- | --- |
| Entry points | `src/agent/__main__.py` | CLI commands: run, TUI, server, auth, benchmark, configuration |
| Event contract | `src/agent/events.py` | Typed, serializable events shared by providers, loop, server, TUI, and persistence |
| Agent brain | `src/agent/core/` | Conversation reconstruction, loop state, retries, context compaction, permissions, cache planning, costs, telemetry |
| Providers | `src/agent/providers/` | Normalized provider interface; Anthropic and OpenAI Responses adapters; SSE parsing and provider registry |
| Tools | `src/agent/tools/` | Workspace-safe read/write/edit/glob/grep/bash tools and process tracking |
| Security | `src/agent/security/` | Secret scanning and redaction before file content reaches a model or log |
| Sandbox | `src/agent/sandbox/` | Policy and macOS/Linux execution backends for constrained shell commands |
| Auth | `src/agent/auth/` | Environment, stored credential, and provider-specific API-key resolution |
| Persistence | `src/agent/persistence/` | SQLite schema migrations, append-only event log, sessions, forks, resume |
| Server | `src/agent/server/` | Newline-delimited JSON-RPC, Unix socket transport, session runtime, approvals |
| TUI | `src/agent/tui/` | Textual interface, RPC client, state reducer, transcript, tools, diffs, approvals |
| Benchmark | `src/agent/benchmark/` | Repeated isolated evaluation, trajectories, Docker verification, reports, scoreboard |

## Provider boundary

The rest of the agent never talks directly to an OpenAI or Anthropic HTTP
payload. It uses the normalized contracts in `providers/base.py`.

```text
AgentLoop
  -> ProviderRequest
  -> provider.stream(...)
  -> shared Event objects
```

This is why a new provider can be added behind an adapter without changing the
agent loop, tools, TUI, or SQLite event log.

### OpenAI path

```text
ProviderRequest
  -> openai_wire.py builds Responses API JSON
  -> openai_responses.py sends and reads the SSE stream
  -> shared events: assistant text, tool calls, usage, end/error
```

Prompt caching is planned before the request is sent. Stable instructions and
tool definitions form a stable prefix; changing conversation messages remain
after that prefix. Telemetry records cache reads/writes, token usage, and
known model cost per turn.

## Tool and safety boundary

The model proposes an action; it never receives direct filesystem or shell
access.

```text
Model tool call
  -> ToolRegistry finds the named tool
  -> ToolDispatcher classifies its risk
  -> PermissionPolicy returns allow / ask / deny
  -> approved tool executes in Workspace + Sandbox boundaries
  -> ToolResult becomes a new model-visible event
```

Important rules:

- `Workspace` rejects paths that escape the selected project root.
- File edits are exact-match operations, preventing accidental broad
  replacements.
- Bash runs asynchronously with output limits, timeout handling, and process
  tracking.
- Reading workspace content marks later risky actions as potentially
  influenced by untrusted text. Normal interactive mode asks the user before
  those actions proceed.
- Sandboxes constrain shell execution according to platform policy and can
  block networking.
- Secrets are redacted at serialization and logging boundaries.

## TUI, RPC, and durable sessions

`agent tui` is an interactive client, not a second implementation of the
agent.

```text
agent tui
  -> AgentRpcClient starts `python -m agent serve`
  -> server receives JSON-RPC over stdin/stdout
  -> SessionService stores session metadata and events in SQLite
  -> server streams the durable events back to the TUI
```

The default database is `.agent/sessions.sqlite3` in the current workspace.
Events are persisted before they are broadcast. Therefore a client can resume
a session after a restart, and a second client can replay durable history.

## Benchmark architecture

Benchmarks measure behavior; they do not merely check whether unit tests pass.

For every attempt, the harness:

1. Reads a task from `benchmarks/tasks/`.
2. Copies its fixture into a fresh disposable workspace.
3. Runs the real headless agent against that copy.
4. Records every event to an append-only JSONL trajectory.
5. Checks that the agent changed only allowed paths.
6. Runs verification commands in a pinned Docker image with network disabled.
7. Writes run evidence and a scoreboard row with commit SHA, model, settings
   fingerprint, costs, tokens, turns, duration, and variance.

```text
benchmark task
  -> isolated workspace
  -> agent attempts (N >= 3)
  -> independent Docker pytest verification
  -> results.json + trajectory.jsonl + scoreboard.md/json
```

Benchmarks have no human available to click an approval dialog. The
`IsolatedBenchmarkApproval` bridge exists only for these disposable fixture
copies. Normal TUI and server sessions keep their interactive approval
behavior. The enforced sandbox and changed-path evaluator remain active during
benchmark attempts.

## Current alpha scope

Working now:

- OpenAI Responses API with streaming, tools, usage, and cost telemetry
- terminal TUI and local JSON-RPC server
- SQLite durable sessions, resume, fork, and replay
- workspace tools, permissions, secret scanning, and sandbox policy
- reproducible Docker-backed benchmark runs

Not yet a production release:

- live Gemini, Bedrock, and Vertex provider validation
- MCP, LSP diagnostics, and subagent orchestration
- signed/notarized installer, self-updates, rollback, and crash reporting

## Development commands

```zsh
# Run directly from this checkout.
uv run agent config
uv run agent tui

# Run the test suite.
uv run pytest -q

# Build a distributable alpha artifact.
uv build
```

For an installed local alpha, use `uv tool install .`, authenticate with
`agent auth --provider openai login`, then launch `agent tui`.

## Repository layout

```text
src/agent/        Application source
tests/            Unit, integration, policy, TUI, and benchmark tests
benchmarks/tasks/ Declarative benchmark tasks
benchmarks/images/Pinned verification-image definition
benchmarks/results/Generated benchmark evidence (ignored by Git)
```
