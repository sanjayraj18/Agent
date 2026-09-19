from decimal import Decimal

import pytest

from agent.core.costs import (
    MODEL_PRICING,
    ModelPricing,
    calculate_known_model_cost,
    calculate_turn_cost,
    pricing_for_model,
)
from agent.events import Usage


def test_calculates_each_cost_component_for_one_turn():
    usage = Usage(
        input_tokens=1_200,
        output_tokens=400,
        cache_read_input_tokens=9_000,
        cache_creation_input_tokens=200,
    )

    cost = calculate_turn_cost(
        usage,
        MODEL_PRICING["claude-sonnet-5"],
    )

    assert cost.input_cost_usd == Decimal("0.0036")
    assert cost.output_cost_usd == Decimal("0.006")
    assert cost.cache_read_cost_usd == Decimal("0.0027")
    assert cost.cache_creation_cost_usd == Decimal("0.00075")
    assert cost.total_usd == Decimal("0.01305")


def test_cache_reads_are_cheaper_than_normal_input_tokens():
    pricing = MODEL_PRICING["claude-sonnet-5"]

    normal_input = calculate_turn_cost(
        Usage(input_tokens=1_000),
        pricing,
    )
    cached_input = calculate_turn_cost(
        Usage(cache_read_input_tokens=1_000),
        pricing,
    )

    assert cached_input.total_usd < normal_input.total_usd
    assert cached_input.total_usd == normal_input.total_usd / Decimal("10")


def test_cache_creation_costs_more_than_normal_input():
    pricing = MODEL_PRICING["claude-sonnet-5"]

    normal_input = calculate_turn_cost(
        Usage(input_tokens=1_000),
        pricing,
    )
    cache_creation = calculate_turn_cost(
        Usage(cache_creation_input_tokens=1_000),
        pricing,
    )

    assert cache_creation.total_usd > normal_input.total_usd
    assert cache_creation.total_usd == (
        normal_input.total_usd * Decimal("1.25")
    )


def test_returns_pricing_for_a_known_model():
    pricing = pricing_for_model("claude-opus-5")

    assert pricing == MODEL_PRICING["claude-opus-5"]


def test_returns_none_when_model_pricing_is_unknown():
    cost = calculate_known_model_cost(
        "unknown-model",
        Usage(input_tokens=100),
    )

    assert cost is None


def test_uses_a_custom_pricing_catalog():
    custom_pricing = ModelPricing(
        input_per_million=Decimal("2"),
        output_per_million=Decimal("8"),
        cache_read_per_million=Decimal("0.2"),
        cache_creation_per_million=Decimal("2.5"),
    )

    cost = calculate_known_model_cost(
        "local-test-model",
        Usage(input_tokens=500_000),
        pricing_catalog={
            "local-test-model": custom_pricing,
        },
    )

    assert cost is not None
    assert cost.total_usd == Decimal("1")


def test_rejects_negative_prices():
    with pytest.raises(ValueError, match="token prices must not be negative"):
        ModelPricing(
            input_per_million=Decimal("-1"),
            output_per_million=Decimal("1"),
            cache_read_per_million=Decimal("0.1"),
            cache_creation_per_million=Decimal("1.25"),
        )


def test_rejects_negative_token_counts():
    with pytest.raises(ValueError, match="token counts must not be negative"):
        calculate_turn_cost(
            Usage(input_tokens=-1),
            MODEL_PRICING["claude-haiku-4-5"],
        )