from typing import Any

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
) -> ToolUsePart:
    return ToolUsePart(
        call_id=call_id,
        name=name,
        arguments={},
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