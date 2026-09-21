from __future__ import annotations

import pytest

from agent.benchmark.approvals import IsolatedBenchmarkApproval
from agent.core.permission import PermissionResult, ToolAction


@pytest.mark.anyio
async def test_isolated_benchmark_approval_allows_a_noninteractive_ask():
    handler = IsolatedBenchmarkApproval()
    action = ToolAction(
        tool_name="edit_file",
        arguments={"path": "src/math_utils.py"},
        contains_untrusted_content=True,
    )
    decision = PermissionResult(
        verdict="ask",
        risk="workspace_write",
        reason="risky action may have been influenced by untrusted content",
    )

    assert await handler.approve(action, decision) is True
