import pytest

from agent.routing.catalog import (
    DEFAULT_ROUTE_CATALOG,
    RouteCatalog,
    RouteNotFoundError,
)
from agent.routing.models import RouteProfile


def _route(
    route_id: str,
    *,
    tier: str = "economy",
    model: str = "gpt-5.6-luna",
    effort: str = "medium",
) -> RouteProfile:
    return RouteProfile(
        route_id=route_id,
        tier=tier,
        provider="openai",
        model=model,
        effort=effort,
        description=f"Route named {route_id}.",
    )


def test_default_catalog_preserves_declared_route_order():
    assert DEFAULT_ROUTE_CATALOG.route_ids == (
        "economy-luna-medium",
        "strong-terra-high",
    )


def test_catalog_returns_a_route_by_its_id():
    route = DEFAULT_ROUTE_CATALOG.require("strong-terra-high")

    assert route.tier == "strong"
    assert route.provider == "openai"
    assert route.model == "gpt-5.6-terra"
    assert route.effort == "high"


def test_catalog_rejects_an_unknown_route_id():
    with pytest.raises(RouteNotFoundError, match="unknown route_id"):
        DEFAULT_ROUTE_CATALOG.require("not-a-real-route")


def test_catalog_rejects_duplicate_route_ids():
    with pytest.raises(ValueError, match="route_id values must be unique"):
        RouteCatalog(
            routes=(
                _route("same-route"),
                _route("same-route"),
            ),
        )


def test_catalog_returns_all_routes_for_a_tier():
    routes = DEFAULT_ROUTE_CATALOG.routes_for_tier("economy")

    assert tuple(route.route_id for route in routes) == (
        "economy-luna-medium",
    )


def test_catalog_rejects_an_empty_route_collection():
    with pytest.raises(ValueError, match="at least 1"):
        RouteCatalog(routes=())