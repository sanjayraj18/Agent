from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


TaskComplexity = Literal["simple","moderate","complex"]


class RoutingSignals(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_complexity : TaskComplexity
    turn_number: int = Field(ge=1)

    tool_error_count: int = Field(default=0, ge=0)
    verification_failure_count: int = Field(default=0, ge=0)

    @property
    def has_failure_evidence(self) -> bool:
        """Return whether tool or verification evidence suggests escalation."""

        return (self.tool_error_count > 0 or self.verification_failure_count > 0)

