from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import uuid4

from agent.core.permission import PermissionResult, ToolAction
from agent.events import PermissionApprovalRequested


ApprovalNotifier = Callable[
    [PermissionApprovalRequested],
    Awaitable[None] | None,
]


@dataclass(frozen=True, slots=True)
class PendingApproval:
    """One user decision that is currently blocking a tool execution."""

    approval_id: str
    action: ToolAction
    decision: PermissionResult

    def to_event(
        self,
        *,
        request_id: str | int,
    ) -> PermissionApprovalRequested:
        """Create the validated control message sent to the TUI."""

        return PermissionApprovalRequested(
            approval_id=self.approval_id,
            request_id=request_id,
            tool_name=self.action.tool_name,
            arguments=dict(self.action.arguments),
            reason=self.decision.reason,
            risk=self.decision.risk,
        )


class ApprovalBroker:
    """
    Bridge a dispatcher ``ask`` decision to a single JSON-RPC client.

    Every proposed action receives a unique ID and its own Future, so two
    concurrent tool calls cannot accidentally share a decision. Closing,
    timing out, malformed input, and unknown IDs all fail closed.
    """

    def __init__(
        self,
        *,
        request_id: str | int,
        notify: ApprovalNotifier,
        timeout_seconds: float = 120.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self._request_id = request_id
        self._notify = notify
        self._timeout_seconds = timeout_seconds
        self._pending: dict[str, asyncio.Future[bool]] = {}
        self._closed = False

    @property
    def pending_count(self) -> int:
        """Number of tool calls still waiting for a human decision."""

        return len(self._pending)

    async def approve(
        self,
        action: ToolAction,
        decision: PermissionResult,
    ) -> bool:
        """Publish one request, then wait for exactly its matching reply."""

        if self._closed:
            return False

        approval_id = uuid4().hex
        pending = PendingApproval(
            approval_id=approval_id,
            action=action,
            decision=decision,
        )
        future = asyncio.get_running_loop().create_future()
        self._pending[approval_id] = future

        try:
            notification = pending.to_event(
                request_id=self._request_id,
            )
            result = self._notify(notification)

            if inspect.isawaitable(result):
                await result

            return await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self._timeout_seconds,
            )

        except (asyncio.TimeoutError, asyncio.CancelledError):
            return False
        except Exception:
            # A disconnected or misbehaving UI must not become permission.
            return False
        finally:
            self._pending.pop(approval_id, None)

    def resolve(
        self,
        approval_id: str,
        *,
        allow: bool,
    ) -> bool:
        """Resolve one live request; unknown or already-resolved IDs fail."""

        future = self._pending.get(approval_id)

        if future is None or future.done() or self._closed:
            return False

        future.set_result(allow)
        return True

    def close(self) -> None:
        """Deny every pending action when the client connection ends."""

        self._closed = True

        for future in self._pending.values():
            if not future.done():
                future.set_result(False)
