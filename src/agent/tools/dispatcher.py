from __future__ import annotations

import asyncio
import logging
from typing import Protocol, Sequence

from agent.core.permission import (
    PermissionPolicy,
    PermissionResult,
    ToolAction,
)
from agent.providers.base import ToolResultPart, ToolUsePart
from agent.tools.registry import ToolRegistry, UnknownToolError


logger = logging.getLogger(__name__)


class ApprovalHandler(Protocol):
    """
    Optional bridge to a human approval UI.

    The headless agent has no approval UI yet, so an `ask` decision safely
    returns an error result. Phase 9's TUI can implement this protocol.
    """

    async def approve(
        self,
        action: ToolAction,
        decision: PermissionResult,
    ) -> bool:
        ...


class ToolDispatcher:
    """
    Execute registered tools only after the permission policy allows them.

    The real headless agent explicitly supplies a project policy from
    `main.py`. A direct caller that supplies no policy preserves the legacy
    dispatcher behavior; this keeps the permission boundary explicit rather
    than silently guessing a policy for embedded callers.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        permissions: PermissionPolicy | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        self._registry = registry
        self._permissions = permissions
        self._approval_handler = approval_handler

    @property
    def permissions(self) -> PermissionPolicy | None:
        return self._permissions

    async def dispatch(
        self,
        call: ToolUsePart,
        *,
        contains_untrusted_content: bool = False,
    ) -> ToolResultPart:
        """
        Run one tool call, or return a permission result without running it.
        """
        try:
            tool = self._registry.get(call.name)
        except UnknownToolError:
            return ToolResultPart(
                call_id=call.call_id,
                content=f"unknown tool: {call.name}",
                is_error=True,
            )

        if self._permissions is not None:
            action = ToolAction(
                tool_name=call.name,
                arguments=call.arguments,
                contains_untrusted_content=contains_untrusted_content,
            )
            decision = self._permissions.decide(action)

            logger.info(
                "tool permission decision",
                extra={
                    "tool_name": call.name,
                    "call_id": call.call_id,
                    "risk": decision.risk,
                    "verdict": decision.verdict,
                },
            )

            if decision.verdict == "deny":
                return _permission_result(
                    call,
                    prefix="permission denied",
                    decision=decision,
                )

            if decision.verdict == "ask":
                approved = await self._request_approval(
                    action,
                    decision,
                )

                if not approved:
                    return _permission_result(
                        call,
                        prefix="permission required",
                        decision=decision,
                    )

        try:
            result = await tool.execute(call.arguments)

        except Exception:
            logger.exception(
                "tool execution failed",
                extra={
                    "tool_name": call.name,
                    "call_id": call.call_id,
                },
            )
            return ToolResultPart(
                call_id=call.call_id,
                content=f"tool failed unexpectedly: {call.name}",
                is_error=True,
            )

        return ToolResultPart(
            call_id=call.call_id,
            content=result.content,
            is_error=result.is_error,
            file_change=(
                result.file_change.event_payload()
                if result.file_change is not None
                else None
            ),
        )

    async def dispatch_all(
        self,
        calls: Sequence[ToolUsePart],
        *,
        contains_untrusted_content: bool = False,
    ) -> list[ToolResultPart]:
        """Dispatch calls concurrently while preserving their request order."""
        return list(
            await asyncio.gather(
                *(
                    self.dispatch(
                        call,
                        contains_untrusted_content=(
                            contains_untrusted_content
                        ),
                    )
                    for call in calls
                )
            )
        )

    async def _request_approval(
        self,
        action: ToolAction,
        decision: PermissionResult,
    ) -> bool:
        """
        Ask an optional UI bridge.

        Without a UI, failing closed is essential: `ask` must never become
        accidental permission to execute.
        """
        if self._approval_handler is None:
            return False

        try:
            return await self._approval_handler.approve(
                action,
                decision,
            )
        except Exception:
            logger.exception(
                "approval handler failed",
                extra={"tool_name": action.tool_name},
            )
            return False


def _permission_result(
    call: ToolUsePart,
    *,
    prefix: str,
    decision: PermissionResult,
) -> ToolResultPart:
    return ToolResultPart(
        call_id=call.call_id,
        content=(
            f"{prefix} for {call.name}: {decision.reason} "
            f"(risk={decision.risk})"
        ),
        is_error=True,
    )
