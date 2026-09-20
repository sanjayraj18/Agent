from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, TextIO

from agent.events import ErrorEvent, Event, dumps
from agent.server.approval import ApprovalBroker
from agent.tools.dispatcher import ApprovalHandler


logger = logging.getLogger(__name__)

AgentRun = Callable[[str], AsyncIterator[Event]]
InteractiveAgentRun = Callable[
    [str, ApprovalHandler],
    AsyncIterator[Event],
]
JsonRpcMessage = dict[str, Any]


@dataclass(frozen=True, slots=True)
class _RunRequest:
    request_id: str | int | None
    prompt: str


@dataclass(frozen=True, slots=True)
class _ApprovalResponse:
    request_id: str | int | None
    approval_id: str
    allow: bool


class JsonRpcServer:
    """
    Stream agent events and accept human decisions on the same stdio channel.

    ``run_agent`` preserves the non-interactive server API used by scripts.
    ``run_agent_with_approval`` is supplied by the headless entry point once
    it constructs an AgentLoop with this server's ApprovalBroker.
    """

    def __init__(
        self,
        run_agent: AgentRun,
        *,
        run_agent_with_approval: InteractiveAgentRun | None = None,
        approval_timeout_seconds: float = 120.0,
    ) -> None:
        if approval_timeout_seconds <= 0:
            raise ValueError(
                "approval_timeout_seconds must be greater than zero"
            )

        self._run_agent = run_agent
        self._run_agent_with_approval = run_agent_with_approval
        self._approval_timeout_seconds = approval_timeout_seconds

    async def serve(self, reader: TextIO, writer: TextIO) -> None:
        """Serve one client while its run streams and approvals arrive."""

        outgoing: asyncio.Queue[JsonRpcMessage | None] = (
            asyncio.Queue()
        )
        active_run: asyncio.Task[None] | None = None
        active_broker: ApprovalBroker | None = None

        async def publish(message: JsonRpcMessage) -> None:
            await outgoing.put(message)

        async def notify_approval(event: object) -> None:
            # ``event`` is a validated Pydantic model. JSON mode makes its
            # primitive values safe for the newline-delimited transport.
            payload = event.model_dump(mode="json")  # type: ignore[attr-defined]
            await publish(
                {
                    "jsonrpc": "2.0",
                    "method": "agent.approval_requested",
                    "params": payload,
                }
            )

        async def write_messages() -> None:
            while True:
                message = await outgoing.get()

                if message is None:
                    return

                writer.write(
                    json.dumps(
                        message,
                        separators=(",", ":"),
                    )
                )
                writer.write("\n")
                writer.flush()

        async def stream_run(
            request: _RunRequest,
            broker: ApprovalBroker,
        ) -> None:
            async for message in self._run_messages(
                request,
                approval_handler=broker,
            ):
                await publish(message)

        writer_task = asyncio.create_task(write_messages())

        try:
            while True:
                line = await asyncio.to_thread(reader.readline)

                if line == "":
                    break

                parsed = self._parse_line(line)

                if isinstance(parsed, dict):
                    await publish(parsed)
                    continue

                if isinstance(parsed, _ApprovalResponse):
                    if active_broker is None:
                        response = self._error(
                            request_id=parsed.request_id,
                            code=-32010,
                            message=(
                                "approval request is no longer pending"
                            ),
                        )
                    elif active_broker.resolve(
                        parsed.approval_id,
                        allow=parsed.allow,
                    ):
                        response = {
                            "jsonrpc": "2.0",
                            "id": parsed.request_id,
                            "result": {"accepted": True},
                        }
                    else:
                        response = self._error(
                            request_id=parsed.request_id,
                            code=-32010,
                            message=(
                                "approval request is no longer pending"
                            ),
                        )

                    if parsed.request_id is not None:
                        await publish(response)

                    continue

                if active_run is not None and active_run.done():
                    active_run = None
                    active_broker = None

                if active_run is not None:
                    await publish(
                        self._error(
                            request_id=parsed.request_id,
                            code=-32000,
                            message="agent run already active",
                        )
                    )
                    continue

                active_broker = ApprovalBroker(
                    request_id=parsed.request_id
                    if parsed.request_id is not None
                    else "notification",
                    notify=notify_approval,
                    timeout_seconds=self._approval_timeout_seconds,
                )
                active_run = asyncio.create_task(
                    stream_run(parsed, active_broker)
                )

        finally:
            if active_broker is not None:
                active_broker.close()

            if active_run is not None:
                await active_run

            await outgoing.put(None)
            await writer_task

    async def handle_line(
        self,
        line: str,
    ) -> AsyncIterator[JsonRpcMessage]:
        """
        Handle one complete non-interactive request for compatibility.

        A one-line caller cannot reply to an approval request while the agent
        is paused, so this method intentionally uses the fail-closed runner.
        ``serve`` is the bidirectional interactive API.
        """

        parsed = self._parse_line(line)

        if isinstance(parsed, dict):
            yield parsed
            return

        if isinstance(parsed, _ApprovalResponse):
            yield self._error(
                request_id=parsed.request_id,
                code=-32010,
                message="approval request is no longer pending",
            )
            return

        async for message in self._run_messages(parsed):
            yield message

    async def _run_messages(
        self,
        request: _RunRequest,
        *,
        approval_handler: ApprovalHandler | None = None,
    ) -> AsyncIterator[JsonRpcMessage]:
        failed = False

        try:
            if (
                approval_handler is not None
                and self._run_agent_with_approval is not None
            ):
                events = self._run_agent_with_approval(
                    request.prompt,
                    approval_handler,
                )
            else:
                events = self._run_agent(request.prompt)

            async for event in events:
                if isinstance(event, ErrorEvent):
                    failed = True

                yield {
                    "jsonrpc": "2.0",
                    "method": "agent.event",
                    "params": {
                        "request_id": request.request_id,
                        "event": json.loads(dumps(event)),
                    },
                }

        except Exception:
            logger.exception("agent execution failed")
            yield self._error(
                request_id=request.request_id,
                code=-32603,
                message="internal error",
            )
            return

        if request.request_id is not None:
            yield {
                "jsonrpc": "2.0",
                "id": request.request_id,
                "result": {
                    "status": "failed" if failed else "completed",
                },
            }

    def _parse_line(
        self,
        line: str,
    ) -> _RunRequest | _ApprovalResponse | JsonRpcMessage:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            return self._error(
                request_id=None,
                code=-32700,
                message="parse error",
            )

        if not isinstance(request, dict):
            return self._error(
                request_id=None,
                code=-32600,
                message="invalid request",
            )

        request_id = request.get("id")

        if request.get("jsonrpc") != "2.0":
            return self._error(
                request_id=request_id,
                code=-32600,
                message="invalid request",
            )

        method = request.get("method")
        if not isinstance(method, str):
            return self._error(
                request_id=request_id,
                code=-32600,
                message="invalid request",
            )

        params = request.get("params", {})
        if not isinstance(params, dict):
            return self._error(
                request_id=request_id,
                code=-32602,
                message="invalid params",
            )

        if method == "agent.run":
            prompt = params.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                return self._error(
                    request_id=request_id,
                    code=-32602,
                    message="params.prompt must be a non-empty string",
                )

            return _RunRequest(
                request_id=request_id,
                prompt=prompt,
            )

        if method == "agent.approval":
            approval_id = params.get("approval_id")
            allow = params.get("allow")

            if not isinstance(approval_id, str) or not approval_id:
                return self._error(
                    request_id=request_id,
                    code=-32602,
                    message=(
                        "params.approval_id must be a non-empty string"
                    ),
                )

            if not isinstance(allow, bool):
                return self._error(
                    request_id=request_id,
                    code=-32602,
                    message="params.allow must be a boolean",
                )

            return _ApprovalResponse(
                request_id=request_id,
                approval_id=approval_id,
                allow=allow,
            )

        return self._error(
            request_id=request_id,
            code=-32601,
            message=f"method not found: {method}",
        )

    @staticmethod
    def _error(
        request_id: str | int | None,
        code: int,
        message: str,
    ) -> JsonRpcMessage:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }
