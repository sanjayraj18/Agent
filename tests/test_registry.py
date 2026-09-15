from typing import Any

import pytest

from agent.tools.base import Tool, ToolExecutionResult
from agent.tools.registry import ToolRegistry, UnknownToolError


class FakeTool(Tool):
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"Test tool named {name}."
        self.input_schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        return ToolExecutionResult(content="ok")


def test_registry_finds_tools_and_preserves_registration_order():
    first = FakeTool("first_tool")
    second = FakeTool("second_tool")

    registry = ToolRegistry([first, second])

    assert "first_tool" in registry
    assert registry.get("first_tool") is first
    assert registry.specs() == [first.spec(), second.spec()]


def test_registry_rejects_duplicate_tool_names():
    registry = ToolRegistry()
    registry.register(FakeTool("duplicate"))

    with pytest.raises(ValueError, match="tool already registered: duplicate"):
        registry.register(FakeTool("duplicate"))


def test_registry_reports_an_unknown_tool_precisely():
    registry = ToolRegistry()

    with pytest.raises(UnknownToolError, match="unknown tool: missing_tool"):
        registry.get("missing_tool")