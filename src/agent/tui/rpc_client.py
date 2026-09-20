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

    Phase 9 starts with one active request at a time. Phase 10 will
    extend this boundary for durable sessions and multiple clients.
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
    ) -> AsyncIterator[AgentStreamItem]:
        """Send one prompt and yield engine events plus approval requests."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        async with self._run_lock:
            await self.start()
            self.last_run_status = None

            request_id = self._next_request_id()
            self._active_run_request_id = request_id

            try:
                await self._send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": "agent.run",
                        "params": {"prompt": prompt},
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
