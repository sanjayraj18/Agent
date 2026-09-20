#every tool should return the same standard structure

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from agent.providers.base import ToolSpec
from agent.tools.changes import FileChange


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    content : str
    is_error : bool = False
    # Local UI metadata. It is not added to the LLM conversation.
    file_change: FileChange | None = None


class Tool(ABC):
    name : str
    description : str
    input_schema : dict[str, Any]

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
        )

    @abstractmethod
    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult:
        raise NotImplementedError

    
