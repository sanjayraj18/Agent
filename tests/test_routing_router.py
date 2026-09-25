import pytest

from agent.routing.catalog import RouteCatalog
from agent.routing.models import RouteProfile
from agent.routing.router import (
    RouterConfigurationError,
    RuleBasedRouter,
)
from agent.routing.signals import RoutingSignals


def _signals(**overrides: object) -> RoutingSignals:
    data: dict[str, object] = {
        "task_complexity": "simple",
        "turn_number": 1,
        "tool_error_count": 0,
        "verification_failure_count": 0,
    }
    data.update(overrides)

    return RoutingSignals.model_validate(data)


def test_routes_a_simple_clean_task_to_the_economy_route():
    decision = RuleBasedRouter().decide(_signals())

    assert decision.route.route_id == "economy-luna-medium"
    assert decision.route.tier == "economy"
    assert decision.reasons == (
        "simple task with no failure evidence",
    )


def test_routes_a_moderate_task_to_the_strong_route():
    decision = RuleBasedRouter().decide(
        _signals(task_complexity="moderate")
    )

    assert decision.route.route_id == "strong-terra-high"
    assert decision.route.tier == "strong"


def test_routes_a_complex_task_to_the_strong_route():
    decision = RuleBasedRouter().decide(
        _signals(task_complexity="complex")
    )

    assert decision.route.route_id == "strong-terra-high"
    assert decision.reasons == (
        "complex tasks start on the strong route",
    )


def test_tool_failure_escalates_a_simple_task_to_the_strong_route():
    decision = RuleBasedRouter().decide(
        _signals(tool_error_count=1)
    )

    assert decision.route.route_id == "strong-terra-high"
    assert decision.reasons == (
        "a tool failed, so the task requires escalation",
    )


def test_verification_failure_escalates_to_the_strong_route():
    decision = RuleBasedRouter().decide(
        _signals(verification_failure_count=1)
    )

    assert decision.route.route_id == "strong-terra-high"
    assert decision.reasons == (
        "verification failed, so the task requires escalation",
    )


def test_simple_task_escalates_after_the_economy_turn_limit():
    router = RuleBasedRouter(economy_turn_limit=2)

    decision = router.decide(
        _signals(turn_number=3)
    )

    assert decision.route.route_id == "strong-terra-high"
    assert decision.reasons == (
        "the simple task exceeded the economy turn limit",
    )


def test_rejects_an_invalid_economy_turn_limit():
    with pytest.raises(
        ValueError,
        match="economy_turn_limit must be at least 1",
    ):
        RuleBasedRouter(economy_turn_limit=0)


def test_requires_at_least_one_route_for_each_tier():
    economy_only_catalog = RouteCatalog(
        routes=(
            RouteProfile(
                route_id="economy-luna-medium",
                tier="economy",
                provider="openai",
                model="gpt-5.6-luna",
                effort="medium",
                description="Lower-cost route.",
            ),
        ),
    )

    with pytest.raises(
        RouterConfigurationError,
        match="catalog must contain at least one 'strong' route",
    ):
        RuleBasedRouter(catalog=economy_only_catalog)