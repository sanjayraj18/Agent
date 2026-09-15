from typing import Any

from agent.providers.base import ToolSpec
from agent.tools.base import Tool, ToolExecutionResult


class AddNumbersTool(Tool):
    name = "add_numbers"
    description = "Add two whole numbers together."

    input_schema = {
        "type": "object",
        "properties": {
            "left": {"type": "integer"},
            "right": {"type": "integer"},
        },
        "required": ["left", "right"],
        "additionalProperties": False,
    }

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        left = arguments.get("left")
        right = arguments.get("right")

        if not isinstance(left, int) or not isinstance(right, int):
            return ToolExecutionResult(
                content="left and right must both be integers",
                is_error=True,
            )

        return ToolExecutionResult(content=str(left + right))


def test_tool_spec_exposes_tool_metadata_to_the_provider():
    tool = AddNumbersTool()

    assert tool.spec() == ToolSpec(
        name="add_numbers",
        description="Add two whole numbers together.",
        input_schema={
            "type": "object",
            "properties": {
                "left": {"type": "integer"},
                "right": {"type": "integer"},
            },
            "required": ["left", "right"],
            "additionalProperties": False,
        },
    )


async def test_tool_returns_success_and_recoverable_error_results():
    tool = AddNumbersTool()

    assert await tool.execute({"left": 12, "right": 7}) == ToolExecutionResult(
        content="19"
    )

    assert await tool.execute({"left": "twelve", "right": 7}) == ToolExecutionResult(
        content="left and right must both be integers",
        is_error=True,
    )