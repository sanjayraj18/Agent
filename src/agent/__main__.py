from  __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4
from getpass import getpass

from pydantic import SecretStr


from agent import config, logs
from agent.auth.credentials import ApiKey
from agent.auth.resolver import CredentialError, resolve
from agent.auth.store import FileStore, StoreError
from agent.core.costs import calculate_known_model_cost
from agent.core.loop import AgentLoop
from agent.core.telemetry import SessionTelemetry
from agent.events import AssistantEnd, ErrorEvent, Event, SessionStarted,ContextCompacted, TextDelta, ThinkingDelta, ToolCallStart, Usage, UserMessage
from agent.providers.anthropic_raw import AnthropicRawProvider
from agent.providers.base import EventFactory, Message, ProviderRequest, TextPart
from agent.server.jsonrpc import JsonRpcServer
from agent.tools.bash import BashTool
from agent.tools.edit_file import EditFileTool
from agent.tools.glob import GlobTool
from agent.tools.grep import GrepTool
from agent.tools.processes import ProcessRegistry
from agent.tools.registry import ToolRegistry
from agent.tools.read_file import ReadFileTool
from agent.tools.workspace import Workspace
from agent.tools.write_file import WriteFileTool
from agent.core.capabilities import capabilities_for_model


PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

def _log_context_compaction(event: ContextCompacted) -> None:
    logs.get("context").info(
        "context compacted",
        extra={
            "previous_input_tokens": event.previous_input_tokens,
            "new_input_tokens": event.new_input_tokens,
            "discarded_message_count": event.discarded_message_count,
            "preserved_file_edit_count": (
                event.preserved_file_edit_count
            ),
            "todo_count": event.todo_count,
        },
    )

def _cost(model: str, usage) -> float | None:
    if model not in PRICING:
        return None
    inp, out = PRICING[model]
    return (
        usage.input_tokens * inp
        + usage.cache_read_input_tokens * inp * 0.10
        + usage.cache_creation_input_tokens * inp * 1.25
        + usage.output_tokens * out
    ) / 1_000_000

DIM, RESET = "\033[2m", "\033[0m"


def _print_turn_telemetry(model: str, usage: Usage) -> None:
    total_input_tokens = (
        usage.input_tokens
        + usage.cache_read_input_tokens
        + usage.cache_creation_input_tokens
    )

    print(
        f"tokens  input={total_input_tokens} "
        f"output={usage.output_tokens} "
        f"cache_read={usage.cache_read_input_tokens} "
        f"cache_write={usage.cache_creation_input_tokens}"
    )

    cost = calculate_known_model_cost(model, usage)

    if cost is None:
        print("cost    unavailable: unknown model pricing")
        return

    print(f"cost    ${cost.total_usd:.5f}")


def _log_live_telemetry(telemetry: SessionTelemetry) -> None:
    """Write Phase 5 telemetry to stderr-safe structured logs."""
    if not telemetry.turns:
        return

    turn = telemetry.turns[-1]

    logs.get("telemetry").info(
        "LLM turn complete",
        extra={
            "turn": turn.turn_number,
            "model": turn.model,
            "input_tokens": turn.usage.input_tokens,
            "output_tokens": turn.usage.output_tokens,
            "cache_read_tokens": turn.usage.cache_read_input_tokens,
            "cache_creation_tokens": (
                turn.usage.cache_creation_input_tokens
            ),
            "turn_cost_usd": (
                str(turn.cost.total_usd)
                if turn.cost is not None
                else None
            ),
            "session_cost_usd": (
                str(telemetry.total_cost_usd)
                if telemetry.total_cost_usd is not None
                else None
            ),
            "cache_hit_rate": str(telemetry.cache_hit_rate),
            "cache_prefix_changed": telemetry.cache_prefix_changed,
        },
    )


async def _headless_run(
    prompt: str,
    credential,
    settings: dict,
) -> AsyncIterator[Event]:
  
    provider = AnthropicRawProvider(credential)
    emit = EventFactory(session_id=uuid4().hex[:12])

    workspace = Workspace(Path.cwd())
    processes = ProcessRegistry()

    session_started = emit(
        SessionStarted,
        cwd=str(workspace.root),
        model=settings["model"],
    )
    user_message = emit(UserMessage, text=prompt)

    request_template = ProviderRequest(
        model=settings["model"],
        max_tokens=settings["max_tokens"],
        effort=settings["effort"],
        messages=[],
        cache_stable_prefix=True
    )

    loop = AgentLoop(
        provider=provider,
        request_template=request_template,
        registry=ToolRegistry(
            [
                ReadFileTool(workspace),
                GlobTool(workspace),
                GrepTool(workspace),
                WriteFileTool(workspace),
                EditFileTool(workspace),
                BashTool(workspace, registry=processes),
            ]
        ),
    )

    try:
        yield session_started
        yield user_message

        async for event in loop.run(
            [session_started, user_message],
            emit,
        ):
            if isinstance(event, AssistantEnd):
                _log_live_telemetry(loop.telemetry)
            elif isinstance(event, ContextCompacted):
                _log_context_compaction(event)
            yield event

    finally:
        await processes.close()
        await provider.aclose()


async def _run(prompt: str, credential, settings: dict) -> int:
    provider = AnthropicRawProvider(credential)
    request = ProviderRequest(
        model=settings["model"],
        max_tokens=settings["max_tokens"],
        effort=settings["effort"],
        messages=[Message(role="user", content=[TextPart(text=prompt)])],
    )
    emit = EventFactory(session_id=uuid4().hex[:12])
    thinking_open = False

    try:
        async for event in provider.stream(request, emit):
            if isinstance(event, ThinkingDelta):
                if not thinking_open:
                    print(f"{DIM}[thinking] ", end="", flush=True)
                    thinking_open = True
                print(f"{DIM}{event.text}{RESET}", end="", flush=True)

            elif isinstance(event, TextDelta):
                if thinking_open:
                    print(f"{RESET}\n")
                    thinking_open = False
                print(event.text, end="", flush=True)

            elif isinstance(event, ToolCallStart):
                print(f"\n{DIM}[tool: {event.name}]{RESET}", flush=True)

            elif isinstance(event, ErrorEvent):
                print(file=sys.stderr)
                print(f"error [{event.kind}]: {event.message}", file=sys.stderr)
                if event.retryable:
                    after = f" after {event.retry_after}s" if event.retry_after else ""
                    print(f"  retryable{after}", file=sys.stderr)
                return 1

            elif isinstance(event, AssistantEnd):
               print(f"\n\n{DIM}stop: {event.stop_reason}")
               _print_turn_telemetry(settings["model"], event.usage)
               print(RESET, end="")
               
        return 0
    finally:
        await provider.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("config", help="show resolved settings and where they came from")
    serve = sub.add_parser("serve", help="run the headless JSON-RPC server over stdin/stdout",)
    serve.add_argument("--model")
    serve.add_argument("--effort",choices=["low", "medium", "high", "xhigh", "max"])
    serve.add_argument("--api-key", dest="api_key")

    auth = sub.add_parser("auth", help="credential commands")
    auth_sub = auth.add_subparsers(dest="auth_command", required=True)
    auth_sub.add_parser("status", help="show the active credential")
    auth_sub.add_parser("login", help="store a credential")
    auth_sub.add_parser("logout", help="remove the stored credential")

    run = sub.add_parser("run", help="send one prompt and stream the reply")
    run.add_argument("prompt")
    run.add_argument("--model")
    run.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    run.add_argument("--api-key", dest="api_key")

    args = parser.parse_args()

    try:
        resolved_config = config.load(
            Path.cwd(),
            {
                "model": getattr(args, "model", None),
                "effort": getattr(args, "effort", None),
            },
        )
    except config.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        raise SystemExit(2)

    settings = config.values(resolved_config)
    logs.setup(level=settings["log_level"])

    if args.command in {"run", "serve"}:
        capabilities = capabilities_for_model(settings["model"])

    if capabilities is None:
        logs.get("context").warning(
            "context management disabled for unknown model",
            extra={"model": settings["model"]},
        )

    if args.command == "config":
        print(config.render(resolved_config))
        return

    store = FileStore()

    # login/logout must come BEFORE resolve() — neither needs an existing
    # credential, and login exists precisely for when there isn't one.
    if args.command == "auth" and args.auth_command == "login":
        raw = getpass("Anthropic API key: ").strip()
        if not raw:
            print("aborted: no key entered", file=sys.stderr)
            raise SystemExit(1)
        credential = ApiKey(value=SecretStr(raw))
        store.save(credential)
        print(f"saved {credential.describe()} to {store.path}")
        return

    if args.command == "auth" and args.auth_command == "logout":
        print("removed stored credential" if store.delete() else "nothing stored")
        return

    try:
        resolved_credential = resolve(
            api_key=getattr(args, "api_key", None), store=store.load
        )
    except (CredentialError, StoreError) as exc:
        print(f"auth error: {exc}", file=sys.stderr)
        raise SystemExit(2)

    if args.command == "auth":         
        print(resolved_credential.describe())
        return

    if args.command == "serve":
        def run_agent(prompt: str) -> AsyncIterator[Event]:
            return _headless_run(
                prompt,
                resolved_credential.credential,
                settings,
            )

        server = JsonRpcServer(run_agent)
        asyncio.run(server.serve(sys.stdin, sys.stdout))
        return

    raise SystemExit(
        asyncio.run(_run(args.prompt, resolved_credential.credential, settings))
    )
