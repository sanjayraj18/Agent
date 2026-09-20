from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FileChange(BaseModel):
    """A confirmed text-file change retained for the local TUI only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    before: str
    after: str
    operation: Literal["created", "updated"]

    @field_validator("path")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)

        if path.is_absolute() or ".." in path.parts:
            raise ValueError("path must be workspace-relative")

        return value

    def event_payload(self) -> dict[str, str]:
        """Return JSON-safe metadata for a ToolResult event."""

        return {
            "path": self.path,
            "before": self.before,
            "after": self.after,
            "operation": self.operation,
        }
