from typing import Any

from agent.core.permission import PermissionPolicy
from agent.providers.base import ToolResultPart, ToolUsePart
from agent.tools.base import Tool, ToolExecutionResult
from agent.tools.dispatcher import ToolDispatcher
from agent.tools.registry import ToolRegistry


class ResultTool(Tool):
    def __init__(
        self,
        name: str,
        result: ToolExecutionResult,
    ) -> None:
        self.name = name
        self.description = f"Test tool named {name}."
        self.input_schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
        self._result = result

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        return self._result


class ExplodingTool(Tool):
    name = "exploding_tool"
    description = "A tool that raises an unexpected exception."
    input_schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        raise RuntimeError("boom")


def _call(
    call_id: str,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> ToolUsePart:
    return ToolUsePart(
        call_id=call_id,
        name=name,
        arguments=arguments or {},
    )


async def test_dispatcher_returns_a_successful_tool_result():
    registry = ToolRegistry(
        [
            ResultTool(
                "echo",
                ToolExecutionResult(content="hello"),
            )
        ]
    )
    dispatcher = ToolDispatcher(registry)

    result = await dispatcher.dispatch(_call("call-1", "echo"))

    assert result == ToolResultPart(
        call_id="call-1",
        content="hello",
        is_error=False,
    )


async def test_dispatcher_preserves_an_expected_tool_error():
    registry = ToolRegistry(
        [
            ResultTool(
                "validation_tool",
                ToolExecutionResult(
                    content="path must be relative",
                    is_error=True,
                ),
            )
        ]
    )
    dispatcher = ToolDispatcher(registry)

    result = await dispatcher.dispatch(
        _call("call-1", "validation_tool")
    )

    assert result == ToolResultPart(
        call_id="call-1",
        content="path must be relative",
        is_error=True,
    )


async def test_dispatcher_converts_an_unknown_tool_to_an_error_result():
    dispatcher = ToolDispatcher(ToolRegistry())

    result = await dispatcher.dispatch(
        _call("call-1", "not_registered")
    )

    assert result == ToolResultPart(
        call_id="call-1",
        content="unknown tool: not_registered",
        is_error=True,
    )


async def test_dispatcher_converts_an_unexpected_exception_to_an_error_result():
    dispatcher = ToolDispatcher(ToolRegistry([ExplodingTool()]))

    result = await dispatcher.dispatch(
        _call("call-1", "exploding_tool")
    )

    assert result == ToolResultPart(
        call_id="call-1",
        content="tool failed unexpectedly: exploding_tool",
        is_error=True,
    )


async def test_dispatch_all_returns_results_in_requested_order():
    first = ResultTool("first", ToolExecutionResult(content="first result"))
    second = ResultTool("second", ToolExecutionResult(content="second result"))
    dispatcher = ToolDispatcher(ToolRegistry([first, second]))

    results = await dispatcher.dispatch_all(
        [
            _call("call-1", "first"),
            _call("call-2", "second"),
        ]
    )

    assert results == [
        ToolResultPart(call_id="call-1", content="first result"),
        ToolResultPart(call_id="call-2", content="second result"),
    ]


class CountingTool(ResultTool):
    def __init__(self, name: str) -> None:
        super().__init__(name, ToolExecutionResult(content="ran"))
        self.calls = 0

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        self.calls += 1
        return await super().execute(arguments)


class ApproveAll:
    async def approve(self, action, decision) -> bool:
        return True


async def test_dispatcher_denies_a_readonly_workspace_write_without_running_it():
    tool = CountingTool("write_file")
    dispatcher = ToolDispatcher(
        ToolRegistry([tool]),
        permissions=PermissionPolicy(default_mode="readonly"),
    )

    result = await dispatcher.dispatch(
        _call("call-1", "write_file", {"path": "notes.txt"})
    )

    assert result.is_error is True
    assert "permission denied" in result.content
    assert tool.calls == 0


async def test_dispatcher_requires_approval_without_running_an_ask_action():
    tool = CountingTool("edit_file")
    dispatcher = ToolDispatcher(
        ToolRegistry([tool]),
        permissions=PermissionPolicy(default_mode="ask"),
    )

    result = await dispatcher.dispatch(
        _call("call-1", "edit_file", {"path": "notes.txt"})
    )

    assert result.is_error is True
    assert "permission required" in result.content
    assert tool.calls == 0


async def test_dispatcher_runs_an_ask_action_only_after_human_approval():
    tool = CountingTool("edit_file")
    dispatcher = ToolDispatcher(
        ToolRegistry([tool]),
        permissions=PermissionPolicy(default_mode="ask"),
        approval_handler=ApproveAll(),
    )

    result = await dispatcher.dispatch(
        _call("call-1", "edit_file", {"path": "notes.txt"})
    )

    assert result.is_error is False
    assert result.content == "ran"
    assert tool.calls == 1


async def test_dispatcher_blocks_untrusted_influenced_actions_without_running_them():
    tool = CountingTool("write_file")
    dispatcher = ToolDispatcher(
        ToolRegistry([tool]),
        permissions=PermissionPolicy(default_mode="full"),
    )

    result = await dispatcher.dispatch(
        _call("call-1", "write_file", {"path": "notes.txt"}),
        contains_untrusted_content=True,
    )

    assert result.is_error is True
    assert "untrusted content" in result.content
    assert tool.calls == 0
