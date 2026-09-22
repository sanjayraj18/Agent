from __future__ import annotations
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent.core.capabilities import require_capabilities
from agent.providers.base import Effort
from agent.providers.profiles import ProviderId, resolve_provider


RouteTier = Literal["economy", "strong"]


class RouteProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra = "forbid")

    route_id: str = Field(min_length=1,pattern=r"^[a-z][a-z0-9-]*$")
    tier: RouteTier
    provider: ProviderId
    model: str = Field(min_length=1)
    effort: Effort | None = None
    description: str = Field(min_length=1)

    @field_validator("model", "description")
    @classmethod
    def reject_blank_text(cls, value : str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_model_contract(self) -> Self:
        
        resolve_provider(self.provider, self.model)
        require_capabilities(self.model)
        return self



    