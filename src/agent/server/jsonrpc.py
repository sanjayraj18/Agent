from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from agent.events import ErrorEvent, Event, dumps
from agent.server.approval import ApprovalBroker
from agent.server.session_service import SessionService
from agent.tools.dispatcher import ApprovalHandler


logger = logging.getLogger(__name__)

AgentRun = Callable[[str], AsyncIterator[Event]]
InteractiveAgentRun = Callable[
    [str, ApprovalHandler],
    AsyncIterator[Event],
]
JsonRpcMessage = dict[str, Any]
ReadLine = Callable[[], Awaitable[str]]
SendMessage = Callable[[JsonRpcMessage], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class _RunRequest:
    request_id: str | int | None
    prompt: str
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class _ApprovalResponse:
    request_id: str | int | None
    approval_id: str
    allow: bool


@dataclass(frozen=True, slots=True)
class _SessionRequest:
    request_id: str | int | None
    method: str
    params: dict[str, Any]


class JsonRpcServer:
    """Serve legacy one-shot runs and durable multi-client sessions.

    The newline-delimited protocol is deliberately transport-neutral. The
    same server can use stdin/stdout for the local TUI or an AF_UNIX socket
    for another local process. SQLite is below this layer, which means neither
    transport is allowed to broadcast an event before it is durable.
    """

    def __init__(
        self,
        run_agent: AgentRun | None = None,
        *,
        run_agent_with_approval: InteractiveAgentRun | None = None,
        session_service: SessionService | None = None,
        default_workspace: Path | str | None = None,
        default_model: str | None = None,
        approval_timeout_seconds: float = 120.0,
    ) -> None:
        if run_agent is None and session_service is None:
            raise ValueError("provide run_agent or session_service")
        if approval_timeout_seconds <= 0:
            raise ValueError(
                "approval_timeout_seconds must be greater than zero"
            )
        if session_service is not None and default_model is None:
            raise ValueError(
                "default_model is required with session_service"
            )

        self._run_agent = run_agent
        self._run_agent_with_approval = run_agent_with_approval
        self._session_service = session_service
        self._default_workspace = str(
            Path(default_workspace or Path.cwd()).expanduser().resolve()
        )
        self._default_model = default_model
        self._approval_timeout_seconds = approval_timeout_seconds

    async def serve(self, reader: TextIO, writer: TextIO) -> None:
        """Serve a blocking stdin/stdout-like text connection."""

        async def read_line() -> str:
            return await asyncio.to_thread(reader.readline)

        async def send(message: JsonRpcMessage) -> None:
            writer.write(json.dumps(message, separators=(",", ":")))
            writer.write("\n")
            writer.flush()

        await self._serve_lines(read_line, send)

    async def serve_stream(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Serve an asyncio byte stream, used by ``UnixSocketServer``."""

        async def read_line() -> str:
            raw = await reader.readline()
            return raw.decode("utf-8", errors="replace")

        async def send(message: JsonRpcMessage) -> None:
            payload = json.dumps(message, separators=(",", ":"))
            writer.write((payload + "\n").encode("utf-8"))
            await writer.drain()

        await self._serve_lines(read_line, send)

    async def handle_line(
        self,
        line: str,
    ) -> AsyncIterator[JsonRpcMessage]:
        """Handle one non-interactive request for scripts and unit tests."""

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

        if isinstance(parsed, _SessionRequest):
            if parsed.method == "session.attach":
                yield self._error(
                    request_id=parsed.request_id,
                    code=-32020,
                    message=(
                        "session.attach requires a streaming connection"
                    ),
                )
                return

            response = await self._session_response(parsed)
            if response is not None:
                yield response
            return

        async for message in self._run_messages(parsed):
            yield message

    async def _serve_lines(
        self,
        read_line: ReadLine,
        send: SendMessage,
    ) -> None:
        outgoing: asyncio.Queue[JsonRpcMessage | None] = asyncio.Queue()
        active_run: asyncio.Task[None] | None = None
        active_broker: ApprovalBroker | None = None
        attachments: dict[str, asyncio.Task[None]] = {}

        async def publish(message: JsonRpcMessage) -> None:
            await outgoing.put(message)

        async def notify_approval(event: object) -> None:
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
                await send(message)

        async def stream_run(
            request: _RunRequest,
            broker: ApprovalBroker,
        ) -> None:
            try:
                async for message in self._run_messages(
                    request,
                    approval_handler=broker,
                ):
                    await publish(message)
            finally:
                broker.close()

        writer_task = asyncio.create_task(write_messages())

        try:
            while True:
                line = await read_line()
                if line == "":
                    break

                parsed = self._parse_line(line)

                if isinstance(parsed, dict):
                    await publish(parsed)
                    continue

                if isinstance(parsed, _ApprovalResponse):
                    await self._handle_approval_response(
                        parsed,
                        active_broker,
                        publish,
                    )
                    continue

                if isinstance(parsed, _SessionRequest):
                    if parsed.method == "session.attach":
                        response = await self._start_attachment(
                            parsed,
                            attachments,
                            publish,
                        )
                    elif parsed.method == "session.detach":
                        response = await self._stop_attachment(
                            parsed,
                            attachments,
                        )
                    else:
                        response = await self._session_response(parsed)

                    if response is not None:
                        await publish(response)
                    continue

                if active_run is not None and active_run.done():
                    await active_run
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
                    request_id=(
                        parsed.request_id
                        if parsed.request_id is not None
                        else "notification"
                    ),
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

            for attachment in attachments.values():
                attachment.cancel()
            if attachments:
                await asyncio.gather(
                    *attachments.values(),
                    return_exceptions=True,
                )

            await outgoing.put(None)
            await writer_task

    async def _handle_approval_response(
        self,
        parsed: _ApprovalResponse,
        active_broker: ApprovalBroker | None,
        publish: Callable[[JsonRpcMessage], Awaitable[None]],
    ) -> None:
        if (
            active_broker is not None
            and active_broker.resolve(
                parsed.approval_id,
                allow=parsed.allow,
            )
        ):
            response: JsonRpcMessage = {
                "jsonrpc": "2.0",
                "id": parsed.request_id,
                "result": {"accepted": True},
            }
        else:
            response = self._error(
                request_id=parsed.request_id,
                code=-32010,
                message="approval request is no longer pending",
            )

        if parsed.request_id is not None:
            await publish(response)

    async def _start_attachment(
        self,
        request: _SessionRequest,
        attachments: dict[str, asyncio.Task[None]],
        publish: Callable[[JsonRpcMessage], Awaitable[None]],
    ) -> JsonRpcMessage | None:
        if self._session_service is None:
            return self._method_not_found(request)

        session_id = request.params.get("session_id")
        after_sequence = request.params.get("after_sequence", 0)

        if not isinstance(session_id, str) or not session_id.strip():
            return self._invalid_params(
                request,
                "params.session_id must be a non-empty string",
            )
        if (
            not isinstance(after_sequence, int)
            or isinstance(after_sequence, bool)
            or after_sequence < 0
        ):
            return self._invalid_params(
                request,
                "params.after_sequence must be a non-negative integer",
            )

        existing = attachments.pop(session_id, None)
        if existing is not None:
            existing.cancel()
            await asyncio.gather(existing, return_exceptions=True)

        try:
            await self._session_service.get_session(session_id)
        except Exception as exc:
            return self._service_error(request, exc)

        async def stream_attachment() -> None:
            assert self._session_service is not None
            try:
                async for event in self._session_service.subscribe(
                    session_id,
                    after_sequence=after_sequence,
                ):
                    await publish(
                        self._event_notification(request.request_id, event)
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "session attachment ended",
                    exc_info=exc,
                )
                await publish(
                    {
                        "jsonrpc": "2.0",
                        "method": "session.subscription_failed",
                        "params": {
                            "session_id": session_id,
                            "message": str(exc),
                        },
                    }
                )

        attachments[session_id] = asyncio.create_task(stream_attachment())
        return self._result(
            request.request_id,
            {
                "attached": True,
                "session_id": session_id,
                "after_sequence": after_sequence,
            },
        )

    async def _stop_attachment(
        self,
        request: _SessionRequest,
        attachments: dict[str, asyncio.Task[None]],
    ) -> JsonRpcMessage | None:
        session_id = request.params.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            return self._invalid_params(
                request,
                "params.session_id must be a non-empty string",
            )

        attachment = attachments.pop(session_id, None)
        if attachment is not None:
            attachment.cancel()
            await asyncio.gather(attachment, return_exceptions=True)

        return self._result(
            request.request_id,
            {"detached": attachment is not None, "session_id": session_id},
        )

    async def _session_response(
        self,
        request: _SessionRequest,
    ) -> JsonRpcMessage | None:
        if self._session_service is None:
            return self._method_not_found(request)

        try:
            result = await self._call_session_service(request)
        except ValueError as exc:
            return self._invalid_params(request, str(exc))
        except Exception as exc:
            return self._service_error(request, exc)

        return self._result(request.request_id, result)

    async def _call_session_service(
        self,
        request: _SessionRequest,
    ) -> dict[str, Any]:
        assert self._session_service is not None
        params = request.params

        if request.method == "session.create":
            workspace = params.get("workspace", self._default_workspace)
            model = params.get("model", self._default_model)
            title = params.get("title")

            if not isinstance(workspace, str) or not workspace.strip():
                raise ValueError(
                    "params.workspace must be a non-empty string"
                )
            if not isinstance(model, str) or not model.strip():
                raise ValueError(
                    "params.model must be a non-empty string"
                )
            if title is not None and not isinstance(title, str):
                raise ValueError("params.title must be a string")

            session = await self._session_service.create_session(
                workspace=workspace,
                model=model,
                title=title,
            )
            return {"session": session.model_dump(mode="json")}

        if request.method == "session.list":
            include_archived = params.get("include_archived", False)
            limit = params.get("limit", 100)
            if not isinstance(include_archived, bool):
                raise ValueError("params.include_archived must be a boolean")
            if (
                not isinstance(limit, int)
                or isinstance(limit, bool)
                or limit < 1
                or limit > 1_000
            ):
                raise ValueError(
                    "params.limit must be an integer from 1 to 1000"
                )
            sessions = await self._session_service.list_sessions(
                include_archived=include_archived,
                limit=limit,
            )
            return {
                "sessions": [
                    session.model_dump(mode="json")
                    for session in sessions
                ]
            }

        session_id = params.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError(
                "params.session_id must be a non-empty string"
            )

        if request.method == "session.get":
            session = await self._session_service.get_session(session_id)
            return {"session": session.model_dump(mode="json")}

        if request.method == "session.archive":
            session = await self._session_service.archive_session(session_id)
            return {"session": session.model_dump(mode="json")}

        if request.method == "session.fork":
            at_sequence = params.get("at_sequence")
            title = params.get("title")
            if (
                at_sequence is not None
                and (
                    not isinstance(at_sequence, int)
                    or isinstance(at_sequence, bool)
                    or at_sequence < 1
                )
            ):
                raise ValueError(
                    "params.at_sequence must be a positive integer"
                )
            if title is not None and not isinstance(title, str):
                raise ValueError("params.title must be a string")
            session = await self._session_service.fork_session(
                session_id,
                at_sequence=at_sequence,
                title=title,
            )
            return {"session": session.model_dump(mode="json")}

        if request.method == "session.replay":
            after_sequence = params.get("after_sequence", 0)
            if (
                not isinstance(after_sequence, int)
                or isinstance(after_sequence, bool)
                or after_sequence < 0
            ):
                raise ValueError(
                    "params.after_sequence must be a non-negative integer"
                )
            events = await self._session_service.replay(
                session_id,
                after_sequence=after_sequence,
            )
            return {
                "events": [json.loads(dumps(event)) for event in events]
            }

        raise ValueError(
            f"unsupported session method: {request.method}"
        )

    async def _run_messages(
        self,
        request: _RunRequest,
        *,
        approval_handler: ApprovalHandler | None = None,
    ) -> AsyncIterator[JsonRpcMessage]:
        failed = False

        try:
            if request.session_id is not None:
                if self._session_service is None:
                    yield self._error(
                        request_id=request.request_id,
                        code=-32601,
                        message="method not found: session.run",
                    )
                    return
                events = self._session_service.run_prompt(
                    request.session_id,
                    request.prompt,
                    approval_handler=approval_handler,
                )
            elif (
                approval_handler is not None
                and self._run_agent_with_approval is not None
            ):
                events = self._run_agent_with_approval(
                    request.prompt,
                    approval_handler,
                )
            elif self._run_agent is not None:
                events = self._run_agent(request.prompt)
            else:
                yield self._error(
                    request_id=request.request_id,
                    code=-32601,
                    message="method not found: agent.run",
                )
                return

            async for event in events:
                if isinstance(event, ErrorEvent):
                    failed = True
                yield self._event_notification(request.request_id, event)

        except Exception:
            logger.exception("agent execution failed")
            yield self._error(
                request_id=request.request_id,
                code=-32603,
                message="internal error",
            )
            return

        if request.request_id is not None:
            result: dict[str, Any] = {
                "status": "failed" if failed else "completed",
            }
            if request.session_id is not None:
                result["session_id"] = request.session_id
            yield self._result(request.request_id, result)

    def _parse_line(
        self,
        line: str,
    ) -> (
        _RunRequest
        | _ApprovalResponse
        | _SessionRequest
        | JsonRpcMessage
    ):
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

        if method in {"agent.run", "session.run"}:
            prompt = params.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                return self._error(
                    request_id=request_id,
                    code=-32602,
                    message="params.prompt must be a non-empty string",
                )

            if method == "session.run":
                session_id = params.get("session_id")
                if not isinstance(session_id, str) or not session_id.strip():
                    return self._error(
                        request_id=request_id,
                        code=-32602,
                        message=(
                            "params.session_id must be a non-empty string"
                        ),
                    )
                return _RunRequest(
                    request_id=request_id,
                    prompt=prompt,
                    session_id=session_id,
                )

            return _RunRequest(request_id=request_id, prompt=prompt)

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

        if method.startswith("session."):
            return _SessionRequest(
                request_id=request_id,
                method=method,
                params=params,
            )

        return self._error(
            request_id=request_id,
            code=-32601,
            message=f"method not found: {method}",
        )

    def _method_not_found(
        self,
        request: _SessionRequest,
    ) -> JsonRpcMessage:
        return self._error(
            request_id=request.request_id,
            code=-32601,
            message=f"method not found: {request.method}",
        )

    def _invalid_params(
        self,
        request: _SessionRequest,
        message: str,
    ) -> JsonRpcMessage:
        return self._error(
            request_id=request.request_id,
            code=-32602,
            message=message,
        )

    def _service_error(
        self,
        request: _SessionRequest,
        exc: Exception,
    ) -> JsonRpcMessage:
        logger.info("session request failed", exc_info=exc)
        return self._error(
            request_id=request.request_id,
            code=-32004,
            message=str(exc) or "session operation failed",
        )

    @staticmethod
    def _event_notification(
        request_id: str | int | None,
        event: Event,
    ) -> JsonRpcMessage:
        return {
            "jsonrpc": "2.0",
            "method": "agent.event",
            "params": {
                "request_id": request_id,
                "event": json.loads(dumps(event)),
            },
        }

    @staticmethod
    def _result(
        request_id: str | int | None,
        result: dict[str, Any],
    ) -> JsonRpcMessage:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }

    @staticmethod
    def _error(
        request_id: str | int | None,
        code: int,
        message: str,
    ) -> JsonRpcMessage:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
