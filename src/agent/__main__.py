from  __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import uuid4
from getpass import getpass

from pydantic import SecretStr


from agent import config, logs
from agent.auth.credentials import ApiKey
from agent.auth.resolver import CredentialError, resolve
from agent.auth.store import FileStore, StoreError
from agent.events import AssistantEnd, ErrorEvent, TextDelta, ThinkingDelta, ToolCallStart
from agent.providers.anthropic_raw import AnthropicRawProvider
from agent.providers.base import EventFactory, Message, ProviderRequest, TextPart

PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

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
                u = event.usage
                print(f"\n\n{DIM}stop: {event.stop_reason}")
                print(
                    f"tokens  in={u.input_tokens} out={u.output_tokens} "
                    f"cache_read={u.cache_read_input_tokens} "
                    f"cache_write={u.cache_creation_input_tokens}"
                )
                if (c := _cost(settings["model"], u)) is not None:
                    print(f"cost    ${c:.5f}")
                print(RESET, end="")
        return 0
    finally:
        await provider.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("config", help="show resolved settings and where they came from")

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

    raise SystemExit(
        asyncio.run(_run(args.prompt, resolved_credential.credential, settings))
    )