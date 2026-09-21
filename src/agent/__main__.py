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
from agent.auth.credentials import ApiKey, Credential, OpenAIApiKey
from agent.auth.resolver import CredentialError, resolve_for_provider
from agent.auth.store import FileStore, StoreError
from agent.core.costs import calculate_known_model_cost
from agent.core.grants import GrantStoreError, ProjectGrantStore
from agent.core.loop import AgentLoop
from agent.core.permission import PermissionPolicy
from agent.core.telemetry import SessionTelemetry
from agent.events import AssistantEnd, ErrorEvent, Event, SessionStarted,ContextCompacted, TextDelta, ThinkingDelta, ToolCallStart, Usage, UserMessage
from agent.providers.base import (
    EventFactory,
    Message,
    Provider,
    ProviderRequest,
    TextPart,
)
from agent.providers.profiles import ProviderId, resolve_provider
from agent.providers.registry import ProviderRegistryError, create_provider
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

DIM, RESET = "\033[2m", "\033[0m"


def _provider_id(settings: dict) -> ProviderId:
    """Resolve once at the boundary before we construct a wire adapter."""

    provider = settings["provider"]
    model = settings["model"]
    if not isinstance(provider, str) or not isinstance(model, str):
        raise ValueError("provider and model must be strings")
    return resolve_provider(provider, model)


def _credential_provider_id(settings: dict) -> ProviderId:
    """Choose a credential namespace without requiring a runnable model.

    `agent auth --provider openai login` must work even when the project's
    default model is still Claude. Model/provider compatibility is enforced
    later for `run` and `serve`, where a request will actually be sent.
    """

    provider = settings["provider"]
    if provider == "anthropic":
        return "anthropic"
    if provider == "openai":
        return "openai"
    return _provider_id(settings)


def _provider_base_url(settings: dict) -> str | None:
    base_url = settings["provider_base_url"]
    return base_url if isinstance(base_url, str) else None


def _create_provider(
    credential: Credential,
    settings: dict,
    *,
    provider_id: ProviderId | None = None,
) -> Provider:
    return create_provider(
        provider_id or _provider_id(settings),
        credential,
        base_url=_provider_base_url(settings),
    )


async def _close_provider(provider: Provider) -> None:
    """Close transport-capable production adapters without enlarging Provider.

    The minimal Provider protocol is intentionally stream-only so unit-test
    fakes and alternative adapters do not need to own network resources.
    """

    close = getattr(provider, "aclose", None)
    if close is not None:
        await close()


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
    credential: Credential,
    settings: dict,
    approval_handler: ApprovalHandler | None = None,
    *,
    workspace_root: Path | None = None,
) -> AsyncIterator[Event]:
    provider = _create_provider(credential, settings)
    emit = EventFactory(session_id=uuid4().hex[:12])

    workspace = Workspace(workspace_root or Path.cwd())
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
        await _close_provider(provider)


def _persistent_runner_factory(
    credential: Credential,
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
            provider: Provider | None = None
            processes = ProcessRegistry()
            sandbox_temporary_directory = None

            try:
                configured_provider = _provider_id(settings)
                try:
                    session_provider = resolve_provider(
                        session.provider,
                        session.model,
                    )
                except ValueError as exc:
                    yield emit(
                        ErrorEvent,
                        kind="provider_configuration_error",
                        message=str(exc),
                        retryable=False,
                    )
                    return

                if session_provider != configured_provider:
                    yield emit(
                        ErrorEvent,
                        kind="provider_configuration_error",
                        message=(
                            "session was created with "
                            f"{session.provider!r}, but this server is "
                            f"configured for {configured_provider!r}"
                        ),
                        retryable=False,
                    )
                    return

                provider = _create_provider(
                    credential,
                    settings,
                    provider_id=session_provider,
                )
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
                if provider is not None:
                    await _close_provider(provider)

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
    provider = _create_provider(credential, settings)
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
        await _close_provider(provider)


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
    serve.add_argument(
        "--provider",
        choices=["anthropic", "openai", "auto"],
    )
    serve.add_argument("--provider-base-url")
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
    auth.add_argument(
        "--provider",
        choices=["anthropic", "openai", "auto"],
    )
    auth_sub = auth.add_subparsers(dest="auth_command", required=True)
    auth_sub.add_parser("status", help="show the active credential")
    auth_sub.add_parser("login", help="store a credential")
    auth_sub.add_parser("logout", help="remove the stored credential")

    bench = sub.add_parser(
        "bench",
        help="run reproducible coding-agent benchmarks",
    )
    bench_sub = bench.add_subparsers(
        dest="bench_command",
        required=True,
    )
    bench_run = bench_sub.add_parser(
        "run",
        help="run one task repeatedly in isolated workspaces",
    )
    bench_run.add_argument("task_id")
    bench_run.add_argument(
        "--provider",
        choices=["anthropic", "openai", "auto"],
    )
    bench_run.add_argument("--provider-base-url")
    bench_run.add_argument("--model")
    bench_run.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
    )
    bench_run.add_argument(
        "--permission-mode",
        dest="permission_mode",
        choices=permission_modes,
    )
    bench_run.add_argument(
        "--sandbox-mode",
        choices=sandbox_modes,
    )
    bench_run.add_argument(
        "--sandbox-network-allowed",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    bench_run.add_argument("--api-key", dest="api_key")
    bench_run.add_argument(
        "--container-image",
        required=True,
        help="verification image pinned as name@sha256:<digest>",
    )
    bench_run.add_argument(
        "--attempts",
        dest="benchmark_attempts",
        type=int,
    )
    bench_run.add_argument(
        "--parallelism",
        dest="benchmark_parallelism",
        type=int,
    )
    bench_run.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("benchmarks"),
    )
    bench_run.add_argument(
        "--results-root",
        type=Path,
        default=Path("benchmarks/results"),
    )
    bench_run.add_argument(
        "--keep-workspaces",
        action="store_true",
        help="preserve copied workspaces after the benchmark completes",
    )

    bench_report = bench_sub.add_parser(
        "report",
        help="render Markdown from a saved JSON scoreboard",
    )
    bench_report.add_argument(
        "--scoreboard-json",
        type=Path,
        default=Path("benchmarks/results/scoreboard.json"),
    )
    bench_report.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/scoreboard.md"),
    )

    run = sub.add_parser("run", help="send one prompt and stream the reply")
    run.add_argument("prompt")
    run.add_argument(
        "--provider",
        choices=["anthropic", "openai", "auto"],
    )
    run.add_argument("--provider-base-url")
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
                "provider": getattr(args, "provider", None),
                "provider_base_url": getattr(
                    args,
                    "provider_base_url",
                    None,
                ),
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
                "benchmark_attempts": getattr(
                    args,
                    "benchmark_attempts",
                    None,
                ),
                "benchmark_parallelism": getattr(
                    args,
                    "benchmark_parallelism",
                    None,
                ),
            },
        )
    except config.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        raise SystemExit(2)

    settings = config.values(resolved_config)
    logs.setup(level=settings["log_level"])

    if args.command in {"run", "serve", "bench"} and not (
        args.command == "bench" and args.bench_command == "report"
    ):
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

    if args.command == "bench" and args.bench_command == "report":
        from agent.benchmark.report import read_scoreboard, render_markdown

        try:
            rows = read_scoreboard(args.scoreboard_json)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(render_markdown(rows), encoding="utf-8")
        except ValueError as exc:
            print(f"benchmark report error: {exc}", file=sys.stderr)
            raise SystemExit(2)
        print(f"wrote {args.output}")
        return

    try:
        selected_provider = (
            _credential_provider_id(settings)
            if args.command == "auth"
            else _provider_id(settings)
        )
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        raise SystemExit(2)

    store = FileStore()

    # login/logout must come BEFORE resolve() — neither needs an existing
    # credential, and login exists precisely for when there isn't one.
    if args.command == "auth" and args.auth_command == "login":
        label = "OpenAI" if selected_provider == "openai" else "Anthropic"
        raw = getpass(f"{label} API key: ").strip()
        if not raw:
            print("aborted: no key entered", file=sys.stderr)
            raise SystemExit(1)
        credential: Credential
        if selected_provider == "openai":
            credential = OpenAIApiKey(value=SecretStr(raw))
        else:
            credential = ApiKey(value=SecretStr(raw))
        store.save(credential, provider=selected_provider)
        print(f"saved {credential.describe()} to {store.path}")
        return

    if args.command == "auth" and args.auth_command == "logout":
        removed = store.delete_for_provider(selected_provider)
        print("removed stored credential" if removed else "nothing stored")
        return

    try:
        resolved_credential = resolve_for_provider(
            selected_provider,
            api_key=getattr(args, "api_key", None),
            store=lambda: store.load_for_provider(selected_provider),
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
            default_provider=selected_provider,
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

    if args.command == "bench":
        from agent.benchmark.approvals import IsolatedBenchmarkApproval
        from agent.benchmark.cli import run_benchmark

        # Benchmarks run against fresh fixture copies, not the user's project,
        # and retain an enforced execution sandbox.  They cannot pause for a
        # TUI approval, so this bridge resolves policy "ask" decisions only
        # for the benchmark attempt.
        benchmark_approval = IsolatedBenchmarkApproval()

        async def benchmark_agent_attempt(
            workspace_root: Path,
            prompt: str,
        ) -> AsyncIterator[Event]:
            async for event in _headless_run(
                prompt,
                resolved_credential.credential,
                settings,
                approval_handler=benchmark_approval,
                workspace_root=workspace_root,
            ):
                yield event

        try:
            execution = asyncio.run(
                run_benchmark(
                    project_root=Path.cwd(),
                    benchmark_root=args.benchmark_root,
                    results_root=args.results_root,
                    task_id=args.task_id,
                    provider=selected_provider,
                    model=settings["model"],
                    container_image=args.container_image,
                    attempts=settings["benchmark_attempts"],
                    parallelism=settings["benchmark_parallelism"],
                    agent_settings=settings,
                    agent_attempt=benchmark_agent_attempt,
                    keep_workspaces=args.keep_workspaces,
                )
            )
        except (ValueError, RuntimeError) as exc:
            print(f"benchmark error: {exc}", file=sys.stderr)
            raise SystemExit(2)

        row = execution.scoreboard
        print(
            f"{row.task_id}: {row.passed_attempts}/{row.attempts} passed "
            f"({row.pass_rate:.2%})"
        )
        print(f"results: {execution.results_path}")
        print(f"scoreboard: {execution.scoreboard_path}")
        return

    raise SystemExit(
        asyncio.run(_run(args.prompt, resolved_credential.credential, settings))
    )


if __name__ == "__main__":
    main()
