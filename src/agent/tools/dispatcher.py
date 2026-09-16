

import asyncio
import logging
from typing import Sequence

from agent.providers.base import ToolResultPart, ToolUsePart
from agent.tools.registry import ToolRegistry, UnknownToolError

logger = logging.getLogger(__name__)


class ToolDispatcher:
    def __init__(self, registry : ToolRegistry) -> None:
        self._registry = registry


    async def dispatch(self,call : ToolUsePart):
        try:
            tool = self._registry.get(call.name)
            result = await tool.execute(call.arguments)

        except UnknownToolError:
            return ToolResultPart(
                call_id=call.call_id,
                content=f"unknown tool: {call.name}",
                is_error=True,
            )

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
        )


    async def dispatch_all(self, calls :Sequence[ToolUsePart]) -> list[ToolResultPart]:
        
        return list(
            await asyncio.gather(
                *(self.dispatch(call) for call in calls)
            )
        )
