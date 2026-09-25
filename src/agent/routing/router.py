from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from agent.routing.catalog import DEFAULT_ROUTE_CATALOG, RouteCatalog
from agent.routing.models import RouteProfile, RouteTier
from agent.routing.signals import RoutingSignals


class RouterConfigurationError(ValueError):
    """Raised when the route catalog cannot support this router policy."""


class RoutingDecision(BaseModel):
    """
    One explainable route recommendation for an LLM turn.

    This is only a recommendation in Phase 4. It does not call a provider and
    does not change AgentLoop behavior yet.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    route: RouteProfile
    signals: RoutingSignals
    reasons: tuple[str, ...] = Field(min_length=1)


class RuleBasedRouter:
    """
    A deliberately conservative and deterministic routing policy.

    Policy:
    - Any tool or verification failure escalates to the strong route.
    - Complex tasks start on the strong route.
    - Moderate tasks start on the strong route.
    - Only simple tasks without failure evidence may use the economy route.
    - A simple task that takes too many turns escalates rather than wandering.
    """

    def __init__(
        self,
        catalog: RouteCatalog = DEFAULT_ROUTE_CATALOG,
        economy_turn_limit: int = 2,
    ) -> None:
        if economy_turn_limit < 1:
            raise ValueError(
                "economy_turn_limit must be at least 1"
            )

        self._catalog = catalog
        self._economy_turn_limit = economy_turn_limit
        self._economy_route = self._require_first_route("economy")
        self._strong_route = self._require_first_route("strong")

    def decide(self, signals: RoutingSignals) -> RoutingDecision:
        """Return an approved route and the reasons for choosing it."""

        if signals.verification_failure_count > 0:
            return self._decision(
                route=self._strong_route,
                signals=signals,
                reasons=(
                    "verification failed, so the task requires escalation",
                ),
            )

        if signals.tool_error_count > 0:
            return self._decision(
                route=self._strong_route,
                signals=signals,
                reasons=(
                    "a tool failed, so the task requires escalation",
                ),
            )

        if signals.task_complexity == "complex":
            return self._decision(
                route=self._strong_route,
                signals=signals,
                reasons=(
                    "complex tasks start on the strong route",
                ),
            )

        if signals.task_complexity == "moderate":
            return self._decision(
                route=self._strong_route,
                signals=signals,
                reasons=(
                    "moderate tasks use the strong route conservatively",
                ),
            )

        if signals.turn_number > self._economy_turn_limit:
            return self._decision(
                route=self._strong_route,
                signals=signals,
                reasons=(
                    "the simple task exceeded the economy turn limit",
                ),
            )

        return self._decision(
            route=self._economy_route,
            signals=signals,
            reasons=(
                "simple task with no failure evidence",
            ),
        )

    def _require_first_route(self, tier: RouteTier) -> RouteProfile:
        routes = self._catalog.routes_for_tier(tier)

        if not routes:
            raise RouterConfigurationError(
                f"catalog must contain at least one {tier!r} route"
            )

        return routes[0]

    @staticmethod
    def _decision(
        *,
        route: RouteProfile,
        signals: RoutingSignals,
        reasons: tuple[str, ...],
    ) -> RoutingDecision:
        return RoutingDecision(
            route=route,
            signals=signals,
            reasons=reasons,
        )