from  __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
from collections.abc import Callable
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
from agent.core.grants import GrantStoreError, ProjectGrantStore
from agent.core.loop import AgentLoop
from agent.core.permission import PermissionPolicy
from agent.core.telemetry import SessionTelemetry
from agent.events import AssistantEnd, ErrorEvent, Event, SessionStarted,ContextCompacted, TextDelta, ThinkingDelta, ToolCallStart, Usage, UserMessage
from agent.providers.anthropic_raw import AnthropicRawProvider
from agent.providers.base import EventFactory, Message, ProviderRequest, TextPart
from agent.persistence.database import SqliteDatabase
from agent.persistence.event_log import EventLog
from agent.persistence.migrations import migrate
from agent.persistence.models import SessionRecord
from agent.persistence.sessions import SessionStore
from agent.sandbox import (
    SandboxBackend,
    SandboxMode,
    SandboxPolicy,
    SandboxPolicyError,
    SandboxUnavailableError,
)
from agent.sandbox.runtime import create_sandbox
from agent.security import SecretScanner
from agent.server.jsonrpc import JsonRpcServer
from agent.server.session_runtime import AgentRunner
from agent.server.session_service import SessionService
from agent.server.unix_socket import UnixSocketServer
from agent.tools.bash import BashTool
from agent.tools.dispatcher import ApprovalHandler
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


def _sandbox_runtime_read_paths() -> tuple[Path, ...]:
    """Return the minimal command runtimes needed inside the sandbox."""

    paths: list[Path] = []

    def include(path: Path) -> None:
        resolved = path.expanduser().resolve()

        if resolved.exists() and resolved not in paths:
            paths.append(resolved)

    python = Path(sys.executable).resolve()
    include(python.parent.parent)

    for command_name in (
        "uv",
        "python",
        "python3",
        "pytest",
        "git",
        "rg",
    ):
        location = shutil.which(command_name)

        if location is not None:
            include(Path(location))

    uv_runtime = Path.home() / ".local" / "share" / "uv"
    if uv_runtime.exists():
        include(uv_runtime)

    return tuple(paths)


def _create_workspace_sandbox(
    workspace: Workspace,
    settings: dict,
) -> tuple[
    SandboxBackend,
    tempfile.TemporaryDirectory[str],
]:
    """Create one enforced sandbox and private temporary directory."""

    temporary_directory = tempfile.TemporaryDirectory(
        prefix=".agent-sandbox-",
        dir=workspace.root,
    )

    try:
        mode_value = settings["sandbox_mode"]
        network_value = settings["sandbox_network_allowed"]

        if not isinstance(mode_value, str):
            raise ValueError("sandbox_mode must be a string")

        if not isinstance(network_value, bool):
            raise ValueError(
                "sandbox_network_allowed must be a boolean"
            )

        policy = SandboxPolicy(
            workspace_root=workspace.root,
            network_allowed=network_value,
            temporary_directory=Path(temporary_directory.name),
            runtime_read_paths=_sandbox_runtime_read_paths(),
            mode=SandboxMode(mode_value),
        )

        return create_sandbox(policy), temporary_directory
    except Exception:
        temporary_directory.cleanup()
        raise


async def _headless_run(
    prompt: str,
    credential,
    settings: dict,
    approval_handler: ApprovalHandler | None = None,
) -> AsyncIterator[Event]:
    provider = AnthropicRawProvider(credential)
    emit = EventFactory(session_id=uuid4().hex[:12])

    workspace = Workspace(Path.cwd())
    processes = ProcessRegistry()
    sandbox_temporary_directory = None

    session_started = emit(
        SessionStarted,
        cwd=str(workspace.root),
        model=settings["model"],
    )
    user_message = emit(UserMessage, text=prompt)

    try:
        try:
            grants = ProjectGrantStore.load(workspace.root)
            permissions = PermissionPolicy(
                default_mode=settings["permission_mode"],
                tool_modes=settings["tool_permission_modes"],
                grants=grants,
            )
        except (GrantStoreError, ValueError) as exc:
            yield session_started
            yield user_message
            yield emit(
                ErrorEvent,
                kind="permission_configuration_error",
                message=str(exc),
                retryable=False,
            )
            return

        logs.get("permissions").info(
            "permission policy loaded",
            extra={
                "default_mode": settings["permission_mode"],
                "tool_overrides": sorted(
                    settings["tool_permission_modes"]
                ),
                "project_grant_count": len(grants.grants),
            },
        )

        try:
            sandbox, sandbox_temporary_directory = (
                _create_workspace_sandbox(
                    workspace,
                    settings,
                )
            )
        except (
            SandboxPolicyError,
            SandboxUnavailableError,
            OSError,
            ValueError,
        ) as exc:
            yield session_started
            yield user_message
            yield emit(
                ErrorEvent,
                kind="sandbox_configuration_error",
                message=str(exc),
                retryable=False,
            )
            return

        logs.get("sandbox").info(
            "sandbox enabled",
            extra={
                "backend": sandbox.name,
                "mode": settings["sandbox_mode"],
                "network_allowed": settings[
                    "sandbox_network_allowed"
                ],
            },
        )

        secret_scanner = SecretScanner()

        request_template = ProviderRequest(
            model=settings["model"],
            max_tokens=settings["max_tokens"],
            effort=settings["effort"],
            messages=[],
            cache_stable_prefix=True,
        )

        loop = AgentLoop(
            provider=provider,
            request_template=request_template,
            registry=ToolRegistry(
                [
                    ReadFileTool(
                        workspace,
                        secret_scanner=secret_scanner,
                    ),
                    GlobTool(workspace),
                    GrepTool(workspace),
                    WriteFileTool(workspace),
                    EditFileTool(workspace),
                    BashTool(
                        workspace,
                        registry=processes,
                        sandbox=sandbox,
                        secret_scanner=secret_scanner,
                    ),
                ]
            ),
            permissions=permissions,
            approval_handler=approval_handler,
        )

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
        if sandbox_temporary_directory is not None:
            sandbox_temporary_directory.cleanup()
        await provider.aclose()


def _persistent_runner_factory(
    credential,
    settings: dict,
) -> Callable[[SessionRecord, ApprovalHandler | None], AgentRunner]:
    """Build AgentLoop runners that continue a durable event history.

    ``SessionRuntime`` creates and persists ``session.started`` and the new
    user message. This adapter owns only the ephemeral resources needed while
    AgentLoop turns that already-durable history into more events.
    """

    def build_runner(
        session: SessionRecord,
        approval_handler: ApprovalHandler | None,
    ) -> AgentRunner:
        async def run(
            history: tuple[Event, ...],
            emit: EventFactory,
        ) -> AsyncIterator[Event]:
            provider = AnthropicRawProvider(credential)
            processes = ProcessRegistry()
            sandbox_temporary_directory = None

            try:
                workspace = Workspace(Path(session.workspace))

                try:
                    grants = ProjectGrantStore.load(workspace.root)
                    permissions = PermissionPolicy(
                        default_mode=settings["permission_mode"],
                        tool_modes=settings["tool_permission_modes"],
                        grants=grants,
                    )
                except (GrantStoreError, ValueError) as exc:
                    yield emit(
                        ErrorEvent,
                        kind="permission_configuration_error",
                        message=str(exc),
                        retryable=False,
                    )
                    return

                try:
                    sandbox, sandbox_temporary_directory = (
                        _create_workspace_sandbox(workspace, settings)
                    )
                except (
                    SandboxPolicyError,
                    SandboxUnavailableError,
                    OSError,
                    ValueError,
                ) as exc:
                    yield emit(
                        ErrorEvent,
                        kind="sandbox_configuration_error",
                        message=str(exc),
                        retryable=False,
                    )
                    return

                secret_scanner = SecretScanner()
                request_template = ProviderRequest(
                    model=session.model,
                    max_tokens=settings["max_tokens"],
                    effort=settings["effort"],
                    messages=[],
                    cache_stable_prefix=True,
                )
                loop = AgentLoop(
                    provider=provider,
                    request_template=request_template,
                    registry=ToolRegistry(
                        [
                            ReadFileTool(
                                workspace,
                                secret_scanner=secret_scanner,
                            ),
                            GlobTool(workspace),
                            GrepTool(workspace),
                            WriteFileTool(workspace),
                            EditFileTool(workspace),
                            BashTool(
                                workspace,
                                registry=processes,
                                sandbox=sandbox,
                                secret_scanner=secret_scanner,
                            ),
                        ]
                    ),
                    permissions=permissions,
                    approval_handler=approval_handler,
                )

                async for event in loop.run(history, emit):
                    if isinstance(event, AssistantEnd):
                        _log_live_telemetry(loop.telemetry)
                    elif isinstance(event, ContextCompacted):
                        _log_context_compaction(event)
                    yield event

            finally:
                await processes.close()
                if sandbox_temporary_directory is not None:
                    sandbox_temporary_directory.cleanup()
                await provider.aclose()

        return run

    return build_runner


def _session_database_path(settings: dict) -> Path:
    value = settings["session_database_path"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "session_database_path must be a non-empty string"
        )

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _create_session_service(
    credential,
    settings: dict,
) -> SessionService:
    """Open, migrate, and wire the one SQLite source of truth."""

    database = SqliteDatabase(_session_database_path(settings))
    migration = migrate(database)
    logs.get("persistence").info(
        "session database ready",
        extra={
            "path": str(database.path),
            "schema_version": migration.current_version,
            "applied_versions": migration.applied_versions,
        },
    )

    return SessionService(
        sessions=SessionStore(database),
        event_log=EventLog(database),
        runner_factory=_persistent_runner_factory(credential, settings),
    )


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

    permission_modes = [
        "readonly",
        "ask",
        "auto",
        "full",
    ]
    sandbox_modes = [
        "enforced",
        "disabled",
        "container",
    ]

    sub.add_parser("config", help="show resolved settings and where they came from")
    sub.add_parser("tui", help="start the interactive terminal interface")
    serve = sub.add_parser(
        "serve",
        help="run the durable JSON-RPC server over stdin/stdout",
    )
    serve.add_argument( "--permission-mode",dest="permission_mode",choices=permission_modes)
    serve.add_argument("--model")
    serve.add_argument("--effort",choices=["low", "medium", "high", "xhigh", "max"])
    serve.add_argument(
        "--sandbox-mode",
        choices=sandbox_modes,
    )
    serve.add_argument(
        "--sandbox-network-allowed",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    serve.add_argument("--api-key", dest="api_key")
    serve.add_argument(
        "--session-database-path",
        dest="session_database_path",
    )
    serve.add_argument(
        "--unix-socket",
        type=Path,
        help="serve the same JSON-RPC protocol on a private Unix socket",
    )

    auth = sub.add_parser("auth", help="credential commands")
    auth_sub = auth.add_subparsers(dest="auth_command", required=True)
    auth_sub.add_parser("status", help="show the active credential")
    auth_sub.add_parser("login", help="store a credential")
    auth_sub.add_parser("logout", help="remove the stored credential")

    run = sub.add_parser("run", help="send one prompt and stream the reply")
    run.add_argument("prompt")
    run.add_argument("--model")
    run.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    run.add_argument( "--permission-mode", dest="permission_mode", choices=permission_modes,)
    run.add_argument(
        "--sandbox-mode",
        choices=sandbox_modes,
    )
    run.add_argument(
        "--sandbox-network-allowed",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    run.add_argument("--api-key", dest="api_key")

    args = parser.parse_args()

    try:
        resolved_config = config.load(
            Path.cwd(),
            {
                "model": getattr(args, "model", None),
                "effort": getattr(args, "effort", None),
                "permission_mode": getattr(args, "permission_mode", None),
                "sandbox_mode": getattr(
                    args,
                    "sandbox_mode",
                    None,
                ),
                "sandbox_network_allowed": getattr(
                    args,
                    "sandbox_network_allowed",
                    None,
                ),
                "session_database_path": getattr(
                    args,
                    "session_database_path",
                    None,
                ),
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

    if args.command == "tui":
        from agent.tui.app import run_tui

        run_tui()
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
        session_service = _create_session_service(
            resolved_credential.credential,
            settings,
        )
        server = JsonRpcServer(
            session_service=session_service,
            default_workspace=Path.cwd(),
            default_model=settings["model"],
        )

        if args.unix_socket is not None:
            socket_server = UnixSocketServer(
                rpc=server,
                path=args.unix_socket,
            )
            asyncio.run(socket_server.serve_forever())
        else:
            asyncio.run(server.serve(sys.stdin, sys.stdout))
        return

    raise SystemExit(
        asyncio.run(_run(args.prompt, resolved_credential.credential, settings))
    )
