from typing import Any

from agent.core.loop import AgentLoop
from agent.core.permission import PermissionPolicy
from agent.events import (
    AssistantEnd,
    AssistantStart,
    Event,
    ToolCallEnd,
    ToolCallStart,
    Usage,
    UserMessage,
)
from agent.providers.base import EventFactory, ProviderRequest
from agent.tools.base import Tool, ToolExecutionResult
from agent.tools.registry import ToolRegistry


class OneWriteProvider:
    name = "one-write-provider"

    def __init__(self) -> None:
        self.turns = 0

    async def stream(
        self,
        request: ProviderRequest,
        emit: EventFactory,
    ):
        self.turns += 1

        if self.turns == 1:
            yield emit(AssistantStart)
            yield emit(
                ToolCallStart,
                index=0,
                call_id="write-1",
                name="write_file",
            )
            yield emit(
                ToolCallEnd,
                index=0,
                call_id="write-1",
                arguments={"path": "notes.txt"},
            )
            yield emit(
                AssistantEnd,
                stop_reason="tool_use",
                usage=Usage(),
            )
            return

        yield emit(AssistantStart)
        yield emit(
            AssistantEnd,
            stop_reason="end_turn",
            usage=Usage(),
        )


class CountingWriteTool(Tool):
    name = "write_file"
    description = "Test workspace write."
    input_schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls = 0

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        self.calls += 1
        return ToolExecutionResult(content="write completed")


class AllowOnce:
    def __init__(self) -> None:
        self.requested_tools: list[str] = []

    async def approve(self, action, decision) -> bool:
        self.requested_tools.append(action.tool_name)
        return True


async def test_loop_passes_the_approval_handler_to_the_dispatcher():
    emit = EventFactory("approval-session")
    provider = OneWriteProvider()
    tool = CountingWriteTool()
    approvals = AllowOnce()
    loop = AgentLoop(
        provider=provider,
        request_template=ProviderRequest(
            model="test-model",
            max_tokens=128,
            messages=[],
        ),
        registry=ToolRegistry([tool]),
        permissions=PermissionPolicy(default_mode="ask"),
        approval_handler=approvals,
    )

    events = [
        event
        async for event in loop.run(
            [emit(UserMessage, text="write notes")],
            emit,
        )
    ]

    assert approvals.requested_tools == ["write_file"]
    assert tool.calls == 1
    assert [event.type for event in events].count("tool.result") == 1
