from __future__ import annotations

from dataclasses import dataclass

from agent.core.capabilities import (
    ModelCapabilities,
    capabilities_for_model,
)
from agent.providers.base import ProviderRequest
from agent.providers.profiles import ProviderId
from agent.routing.models import RouteProfile
from agent.routing.router import RoutingDecision, RuleBasedRouter
from agent.routing.signals import RoutingSignals


@dataclass(frozen=True, slots=True)
class LiveRouteResult:
    """The outcome of attempting to apply one route to a provider request.

    ``decision`` is always recorded, even when the result falls back to the
    configured request. ``applied`` is the boundary between a recommendation
    and a request that is safe to send to the current provider.
    """

    decision: RoutingDecision
    request: ProviderRequest
    capabilities: ModelCapabilities | None
    applied: bool
    fallback_reason: str | None = None

    @property
    def route(self) -> RouteProfile | None:
        """Return the actual route only when it was safely applied."""

        return self.decision.route if self.applied else None


class LiveRouteController:
    """Apply approved same-provider routes without changing provider identity.

    Phase 6 deliberately does not switch between provider accounts or wire
    adapters. It changes only the model and effort of the provider that is
    already authenticated for this session. A route with incompatible
    provider or model features falls back to the configured request.
    """

    def __init__(
        self,
        *,
        active_provider: ProviderId,
        router: RuleBasedRouter | None = None,
    ) -> None:
        self._active_provider = active_provider
        self._router = router or RuleBasedRouter()

    def select(
        self,
        signals: RoutingSignals,
        request: ProviderRequest,
    ) -> LiveRouteResult:
        """Return a safe request for this turn and the decision behind it."""

        decision = self._router.decide(signals)
        route = decision.route

        if route.provider != self._active_provider:
            return self._fallback(
                decision,
                request,
                reason=(
                    "route provider does not match the authenticated "
                    "session provider"
                ),
            )

        capabilities = capabilities_for_model(route.model)

        if capabilities is None:
            return self._fallback(
                decision,
                request,
                reason="route model has no registered capabilities",
            )

        if request.tools and not capabilities.supports_tools:
            return self._fallback(
                decision,
                request,
                reason="route model does not support the active tools",
            )

        if request.thinking and not capabilities.supports_thinking:
            return self._fallback(
                decision,
                request,
                reason="route model does not support the active thinking mode",
            )

        return LiveRouteResult(
            decision=decision,
            request=request.model_copy(
                update={
                    "model": route.model,
                    "effort": route.effort,
                    "max_tokens": min(
                        request.max_tokens,
                        capabilities.max_output_tokens,
                    ),
                }
            ),
            capabilities=capabilities,
            applied=True,
        )

    @staticmethod
    def _fallback(
        decision: RoutingDecision,
        request: ProviderRequest,
        *,
        reason: str,
    ) -> LiveRouteResult:
        return LiveRouteResult(
            decision=decision,
            request=request,
            capabilities=capabilities_for_model(request.model),
            applied=False,
            fallback_reason=reason,
        )
