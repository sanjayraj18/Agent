from agent.providers.base import ProviderRequest
from agent.routing.catalog import RouteCatalog
from agent.routing.live import LiveRouteController
from agent.routing.models import RouteProfile
from agent.routing.router import RuleBasedRouter
from agent.routing.signals import RoutingSignals


def _request() -> ProviderRequest:
    return ProviderRequest(
        model="gpt-5.6-terra",
        max_tokens=16_000,
        effort="high",
        messages=[],
    )


def _signals(**overrides: object) -> RoutingSignals:
    values: dict[str, object] = {
        "task_complexity": "simple",
        "turn_number": 1,
    }
    values.update(overrides)
    return RoutingSignals.model_validate(values)


def test_applies_an_approved_same_provider_route_to_the_request():
    result = LiveRouteController(
        active_provider="openai",
    ).select(_signals(), _request())

    assert result.applied is True
    assert result.fallback_reason is None
    assert result.route is not None
    assert result.route.route_id == "economy-luna-medium"
    assert result.request.model == "gpt-5.6-luna"
    assert result.request.effort == "medium"
    assert result.capabilities is not None
    assert result.capabilities.model == "gpt-5.6-luna"


def test_falls_back_without_switching_provider_accounts():
    catalog = RouteCatalog(
        routes=(
            RouteProfile(
                route_id="economy-haiku-medium",
                tier="economy",
                provider="anthropic",
                model="claude-haiku-4-5",
                effort="medium",
                description="Economy route on another provider.",
            ),
            RouteProfile(
                route_id="strong-terra-high",
                tier="strong",
                provider="openai",
                model="gpt-5.6-terra",
                effort="high",
                description="Strong OpenAI route.",
            ),
        )
    )
    controller = LiveRouteController(
        active_provider="openai",
        router=RuleBasedRouter(catalog=catalog),
    )

    request = _request()
    result = controller.select(_signals(), request)

    assert result.applied is False
    assert result.route is None
    assert result.request is request
    assert result.fallback_reason == (
        "route provider does not match the authenticated session provider"
    )
