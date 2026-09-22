from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.routing.models import RouteProfile, RouteTier


class RouteNotFoundError(ValueError):
    """Raised when code requests a route that is not in the catalog."""


class RouteCatalog(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    routes : tuple[RouteProfile,...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_route_ids(self) -> RouteCatalog:
        route_ids = [route.route_id for route in self.routes]

        if len(route_ids) != len(set(route_ids)):
            raise ValueError("route_id values must be unique")

        return self

    @property
    def route_ids(self) -> tuple[str, ...]:
        return tuple(route.route_id for route in self.routes)

    def require(self, route_id: str) -> RouteProfile:
        for route in self.routes:
            if route.route_id == route_id:
                return route

        raise RouteNotFoundError(f"unknown route_id: {route_id!r}")

    def routes_for_tier(self, tier: RouteTier) -> tuple[RouteProfile, ...]:
        return tuple(route for route in self.routes if route.tier == tier)


DEFAULT_ROUTE_CATALOG = RouteCatalog(
    routes=(
        RouteProfile(
            route_id="economy-luna-medium",
            tier="economy",
            provider="openai",
            model="gpt-5.6-luna",
            effort="medium",
            description=(
                "Lower-cost route for straightforward, low-risk work."
            ),
        ),
        RouteProfile(
            route_id="strong-terra-high",
            tier="strong",
            provider="openai",
            model="gpt-5.6-terra",
            effort="high",
            description=(
                "Higher-capability route for difficult tasks or escalation."
            ),
        ),
    ),
)