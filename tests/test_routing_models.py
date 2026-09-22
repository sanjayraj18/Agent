import pytest

from agent.routing.models import RouteProfile


def _route(**overrides: object) -> RouteProfile:
    data: dict[str, object] = {
        "route_id": "economy-luna-medium",
        "tier": "economy",
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "effort": "medium",
        "description": "Lower-cost route for straightforward work.",
    }
    data.update(overrides)

    return RouteProfile.model_validate(data)


def test_creates_a_valid_route_profile():
    route = _route()

    assert route.route_id == "economy-luna-medium"
    assert route.tier == "economy"
    assert route.provider == "openai"
    assert route.model == "gpt-5.6-luna"
    assert route.effort == "medium"


def test_strips_model_and_description_whitespace():
    route = _route(
        model="  gpt-5.6-luna  ",
        description="  Lower-cost route.  ",
    )

    assert route.model == "gpt-5.6-luna"
    assert route.description == "Lower-cost route."


def test_rejects_an_invalid_route_id():
    with pytest.raises(ValueError, match="pattern"):
        _route(route_id="Economy Route")


def test_rejects_a_blank_model_name():
    with pytest.raises(ValueError, match="value must not be blank"):
        _route(model="   ")


def test_rejects_a_known_provider_model_mismatch():
    with pytest.raises(ValueError, match="belongs to openai"):
        _route(
            provider="anthropic",
            model="gpt-5.6-luna",
        )


def test_rejects_a_model_without_known_capabilities():
    with pytest.raises(ValueError, match="unknown model capabilities"):
        _route(model="gpt-unknown-model")