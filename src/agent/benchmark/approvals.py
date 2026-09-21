"""Approval policy used only by isolated, non-interactive benchmarks."""

from __future__ import annotations

from agent.core.permission import PermissionResult, ToolAction


class IsolatedBenchmarkApproval:
    """Approve tool actions within one disposable benchmark workspace.

    A benchmark has no person at the keyboard to answer an ``ask`` decision.
    Its runner instead creates a fresh fixture copy for every attempt, keeps
    the agent's execution sandbox enforced, and rejects unexpected changed
    paths before verification.  This bridge belongs only to ``agent bench``;
    the interactive TUI continues to require a user's decision.
    """

    async def approve(
        self,
        action: ToolAction,
        decision: PermissionResult,
    ) -> bool:
        """Allow an already-isolated action so the benchmark can progress."""
        del action, decision
        return True
