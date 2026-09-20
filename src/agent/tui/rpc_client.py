from __future__ import annotations

import asyncio
import json
import sys
from collections import deque
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from typing import Any

from agent.events import (
    Event,
    EventAdapter,
    PermissionApprovalRequested,
)
from agent.persistence.models import SessionRecord


AgentStreamItem = Event | PermissionApprovalRequested


class RpcClientError(RuntimeError):
    """Base error for TUI-to-agent JSON-RPC communication."""


class AgentProcessError(RpcClientError):
    """Raised when the headless agent process cannot serve requests."""


class RpcResponseError(RpcClientError):
    """Raised when the server returns a JSON-RPC error response."""

    def __init__(
        self,
        code: int,
        message: str,
    ) -> None:
        super().__init__(f"JSON-RPC error {code}: {message}")
        self.code = code
        self.message = message


class AgentRpcClient:
    """
    A TUI client for one local `agent serve` child process.

    One client process can create, resume, fork, and archive durable sessions.
    A session run is still serialized per client because stdout is one ordered
    JSON-RPC stream; other clients may attach through the Unix socket server.
    """

    def __init__(
        self,
        command: tuple[str, ...] | None = None,
        cwd: Path | None = None,
    ) -> None:
        self._command = command or (
            sys.executable,
            "-m",
            "agent",
            "serve",
        )
        self._cwd = cwd
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_lines: deque[str] = deque(maxlen=200)
        self._deferred_messages: deque[dict[str, Any]] = deque()
        self._request_number = 0
        self._run_lock = asyncio.Lock()
        self._active_run_request_id: str | None = None
        self.active_session_id: str | None = None
        self.last_run_status: str | None = None

    @property
    def is_connected(self) -> bool:
        return (
            self._process is not None
            and self._process.returncode is None
        )

    @property
    def recent_stderr(self) -> tuple[str, ...]:
        """Recent agent diagnostics, kept bounded for a UI error panel."""

        return tuple(self._stderr_lines)

    async def start(self) -> None:
        """Start the child agent process once."""

        if self.is_connected:
            return

        await self.close()

        try:
            process = await asyncio.create_subprocess_exec(
                *self._command,
                cwd=str(self._cwd) if self._cwd is not None else None,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise AgentProcessError(
                "could not start headless agent: "
                f"{exc}"
            ) from exc

        self._process = process
        self._stderr_task = asyncio.create_task(
            self._capture_stderr(process)
        )

    async def run(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
    ) -> AsyncIterator[AgentStreamItem]:
        """Run a prompt, optionally inside a durable session."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        async with self._run_lock:
            await self.start()
            self.last_run_status = None

            request_id = self._next_request_id()
            self._active_run_request_id = request_id
            self.active_session_id = session_id

            params: dict[str, Any] = {"prompt": prompt}
            method = "agent.run"

            if session_id is not None:
                if not session_id.strip():
                    raise ValueError(
                        "session_id must be a non-empty string"
                    )
                method = "session.run"
                params["session_id"] = session_id

            try:
                await self._send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": method,
                        "params": params,
                    }
                )

                while True:
                    message = await self._read_message()
                    method = message.get("method")

                    if method == "agent.event":
                        event = self._event_from_notification(
                            message,
                            request_id=request_id,
                        )

                        if event is not None:
                            yield event

                        continue

                    if method == "agent.approval_requested":
                        approval = self._approval_from_notification(
                            message,
                            request_id=request_id,
                        )

                        if approval is not None:
                            yield approval

                        continue

                    if message.get("id") != request_id:
                        raise RpcClientError(
                            "received a response for an unexpected request"
                        )

                    error = message.get("error")
                    if error is not None:
                        raise self._response_error(error)

                    result = message.get("result")
                    if not isinstance(result, dict):
                        raise RpcClientError(
                            "JSON-RPC response is missing a result object"
                        )

                    status = result.get("status")
                    if not isinstance(status, str):
                        raise RpcClientError(
                            "JSON-RPC result is missing a status"
                        )

                    self.last_run_status = status
                    return
            finally:
                self._active_run_request_id = None

    async def create_session(
        self,
        *,
        workspace: Path | None = None,
        model: str | None = None,
        title: str | None = None,
    ) -> SessionRecord:
        """Create a session using server defaults when values are omitted."""

        params: dict[str, Any] = {}
        if workspace is not None:
            params["workspace"] = str(workspace)
        if model is not None:
            params["model"] = model
        if title is not None:
            params["title"] = title

        result = await self._request("session.create", params)
        return self._session_from_result(result)

    async def list_sessions(
        self,
        *,
        include_archived: bool = False,
        limit: int = 100,
    ) -> tuple[SessionRecord, ...]:
        """Fetch sessions for a local picker without loading full histories."""

        result = await self._request(
            "session.list",
            {
                "include_archived": include_archived,
                "limit": limit,
            },
        )
        values = result.get("sessions")
        if not isinstance(values, list):
            raise RpcClientError("session.list returned invalid sessions")

        try:
            return tuple(SessionRecord.model_validate(value) for value in values)
        except Exception as exc:
            raise RpcClientError(
                "session.list returned an invalid session"
            ) from exc

    async def replay_session(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
    ) -> tuple[Event, ...]:
        """Load a finite durable history before resuming a session."""

        result = await self._request(
            "session.replay",
            {
                "session_id": session_id,
                "after_sequence": after_sequence,
            },
        )
        values = result.get("events")
        if not isinstance(values, list):
            raise RpcClientError("session.replay returned invalid events")

        try:
            return tuple(EventAdapter.validate_python(value) for value in values)
        except Exception as exc:
            raise RpcClientError(
                "session.replay returned an invalid event"
            ) from exc

    async def fork_session(
        self,
        session_id: str,
        *,
        at_sequence: int | None = None,
        title: str | None = None,
    ) -> SessionRecord:
        """Branch a session without mutating its original history."""

        params: dict[str, Any] = {"session_id": session_id}
        if at_sequence is not None:
            params["at_sequence"] = at_sequence
        if title is not None:
            params["title"] = title

        result = await self._request("session.fork", params)
        return self._session_from_result(result)

    async def archive_session(
        self,
        session_id: str,
    ) -> SessionRecord:
        """Archive history from the normal picker without deleting it."""

        result = await self._request(
            "session.archive",
            {"session_id": session_id},
        )
        return self._session_from_result(result)

    async def respond_to_approval(
        self,
        approval_id: str,
        *,
        allow: bool,
    ) -> None:
        """Reply to the approval request currently yielded by ``run``."""

        if self._active_run_request_id is None:
            raise RpcClientError("there is no active approval request")

        if not isinstance(approval_id, str) or not approval_id:
            raise ValueError("approval_id must be a non-empty string")
        if not isinstance(allow, bool):
            raise ValueError("allow must be a boolean")

        response_id = self._next_request_id()
        await self._send(
            {
                "jsonrpc": "2.0",
                "id": response_id,
                "method": "agent.approval",
                "params": {
                    "approval_id": approval_id,
                    "allow": allow,
                },
            }
        )

        while True:
            message = await self._read_wire_message()

            if message.get("id") != response_id:
                if message.get("method") in {
                    "agent.event",
                    "agent.approval_requested",
                }:
                    # A parallel tool can finish, or request approval, while
                    # the server acknowledges this decision. Preserve its
                    # order for the paused ``run`` iterator to yield next.
                    self._deferred_messages.append(message)
                    continue

                raise RpcClientError(
                    "approval response arrived out of order"
                )

            error = message.get("error")
            if error is not None:
                raise self._response_error(error)

            result = message.get("result")
            if (
                not isinstance(result, dict)
                or result.get("accepted") is not True
            ):
                raise RpcClientError("approval response was not accepted")

            return

    async def close(self) -> None:
        """Stop the child process and release all local resources."""

        process = self._process
        stderr_task = self._stderr_task

        self._process = None
        self._stderr_task = None
        self._deferred_messages.clear()
        self.active_session_id = None

        if process is not None:
            if process.stdin is not None:
                process.stdin.close()

                with suppress(
                    BrokenPipeError,
                    ConnectionResetError,
                ):
                    await process.stdin.wait_closed()

            if process.returncode is None:
                try:
                    await asyncio.wait_for(
                        process.wait(),
                        timeout=2,
                    )
                except TimeoutError:
                    process.terminate()

                    try:
                        await asyncio.wait_for(
                            process.wait(),
                            timeout=2,
                        )
                    except TimeoutError:
                        process.kill()
                        await process.wait()

        if stderr_task is not None:
            stderr_task.cancel()

            with suppress(asyncio.CancelledError):
                await stderr_task

    async def _send(self, message: dict[str, Any]) -> None:
        process = self._require_process()
        stdin = process.stdin

        if stdin is None:
            raise AgentProcessError(
                "headless agent has no stdin pipe"
            )

        encoded = (
            json.dumps(
                message,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

        stdin.write(encoded)

        try:
            await stdin.drain()
        except (
            BrokenPipeError,
            ConnectionResetError,
        ) as exc:
            raise self._exited_error() from exc

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Make a finite control request while no prompt is streaming."""

        async with self._run_lock:
            await self.start()
            request_id = self._next_request_id()
            await self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                }
            )

            while True:
                message = await self._read_message()

                if message.get("id") != request_id:
                    if message.get("method") in {
                        "agent.event",
                        "agent.approval_requested",
                    }:
                        self._deferred_messages.append(message)
                        continue
                    raise RpcClientError(
                        "received a response for an unexpected request"
                    )

                error = message.get("error")
                if error is not None:
                    raise self._response_error(error)

                result = message.get("result")
                if not isinstance(result, dict):
                    raise RpcClientError(
                        "JSON-RPC response is missing a result object"
                    )
                return result

    async def _read_message(self) -> dict[str, Any]:
        if self._deferred_messages:
            return self._deferred_messages.popleft()

        return await self._read_wire_message()

    async def _read_wire_message(self) -> dict[str, Any]:
        process = self._require_process()
        stdout = process.stdout

        if stdout is None:
            raise AgentProcessError(
                "headless agent has no stdout pipe"
            )

        raw = await stdout.readline()

        if not raw:
            raise self._exited_error()

        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RpcClientError(
                "headless agent wrote invalid JSON-RPC output"
            ) from exc

        if not isinstance(message, dict):
            raise RpcClientError(
                "headless agent wrote a non-object JSON-RPC message"
            )

        return message

    def _event_from_notification(
        self,
        message: dict[str, Any],
        *,
        request_id: str,
    ) -> Event | None:
        params = message.get("params")

        if not isinstance(params, dict):
            raise RpcClientError(
                "agent.event notification has invalid params"
            )

        if params.get("request_id") != request_id:
            return None

        event_payload = params.get("event")

        if not isinstance(event_payload, dict):
            raise RpcClientError(
                "agent.event notification is missing an event"
            )

        try:
            return EventAdapter.validate_python(event_payload)
        except Exception as exc:
            raise RpcClientError(
                "agent.event contains an invalid event payload"
            ) from exc

    def _approval_from_notification(
        self,
        message: dict[str, Any],
        *,
        request_id: str,
    ) -> PermissionApprovalRequested | None:
        params = message.get("params")

        if not isinstance(params, dict):
            raise RpcClientError(
                "approval notification has invalid params"
            )

        try:
            approval = PermissionApprovalRequested.model_validate(params)
        except Exception as exc:
            raise RpcClientError(
                "approval notification has an invalid payload"
            ) from exc

        if approval.request_id != request_id:
            return None

        return approval

    def _next_request_id(self) -> str:
        self._request_number += 1
        return f"tui-{self._request_number}"

    def _require_process(self) -> asyncio.subprocess.Process:
        if self._process is None:
            raise AgentProcessError(
                "headless agent process is not running"
            )

        return self._process

    def _response_error(self, value: object) -> RpcResponseError:
        if not isinstance(value, dict):
            raise RpcClientError(
                "JSON-RPC error response has invalid shape"
            )

        code = value.get("code")
        message = value.get("message")

        if not isinstance(code, int) or not isinstance(message, str):
            raise RpcClientError(
                "JSON-RPC error response is missing code or message"
            )

        return RpcResponseError(code, message)

    @staticmethod
    def _session_from_result(result: dict[str, Any]) -> SessionRecord:
        value = result.get("session")
        if not isinstance(value, dict):
            raise RpcClientError("session result is missing a session object")

        try:
            return SessionRecord.model_validate(value)
        except Exception as exc:
            raise RpcClientError("session result is invalid") from exc

    async def _capture_stderr(
        self,
        process: asyncio.subprocess.Process,
    ) -> None:
        stderr = process.stderr

        if stderr is None:
            return

        while line := await stderr.readline():
            self._stderr_lines.append(
                line.decode(
                    "utf-8",
                    errors="replace",
                ).rstrip("\n")
            )

    def _exited_error(self) -> AgentProcessError:
        details = "\n".join(self._stderr_lines)

        if details:
            return AgentProcessError(
                "headless agent exited unexpectedly:\n"
                f"{details}"
            )

        return AgentProcessError(
            "headless agent exited unexpectedly"
        )
