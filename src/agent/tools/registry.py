from __future__ import annotations

from collections.abc import Iterable

from agent.providers.base import ToolSpec
from agent.tools.base import Tool


class UnknownToolError(LookupError):
    def __init__(self, name: str) -> None:
        super().__init__(f"unknown tool: {name}")
        self.name = name


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        name = tool.name.strip()

        if not name:
            raise ValueError("tool name must not be empty")

        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")

        self._tools[name] = tool


    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise UnknownToolError(name) from exc


    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]


    def __contains__(self, name: str) -> bool:
        return name in self._tools