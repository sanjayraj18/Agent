from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from agent.events import Usage


MILLION_TOKENS = Decimal(1_000_000)


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """USD prices per one million tokens for one model."""

    input_per_million: Decimal
    output_per_million: Decimal
    cache_read_per_million: Decimal
    cache_creation_per_million: Decimal

    def __post_init__(self) -> None:
        prices = (
            self.input_per_million,
            self.output_per_million,
            self.cache_read_per_million,
            self.cache_creation_per_million,
        )

        if any(price < 0 for price in prices):
            raise ValueError("token prices must not be negative")


@dataclass(frozen=True, slots=True)
class TurnCost:
    """The cost breakdown for one completed LLM turn."""

    input_cost_usd: Decimal
    output_cost_usd: Decimal
    cache_read_cost_usd: Decimal
    cache_creation_cost_usd: Decimal

    @property
    def total_usd(self) -> Decimal:
        return (
            self.input_cost_usd
            + self.output_cost_usd
            + self.cache_read_cost_usd
            + self.cache_creation_cost_usd
        )


MODEL_PRICING: dict[str, ModelPricing] = {
    "claude-opus-5": ModelPricing(
        input_per_million=Decimal("5.00"),
        output_per_million=Decimal("25.00"),
        cache_read_per_million=Decimal("0.50"),
        cache_creation_per_million=Decimal("6.25"),
    ),
    "claude-sonnet-5": ModelPricing(
        input_per_million=Decimal("3.00"),
        output_per_million=Decimal("15.00"),
        cache_read_per_million=Decimal("0.30"),
        cache_creation_per_million=Decimal("3.75"),
    ),
    "claude-haiku-4-5": ModelPricing(
        input_per_million=Decimal("1.00"),
        output_per_million=Decimal("5.00"),
        cache_read_per_million=Decimal("0.10"),
        cache_creation_per_million=Decimal("1.25"),
    ),
}


def pricing_for_model(model: str) -> ModelPricing | None:
    """Return pricing when this agent knows the model; otherwise return None."""
    return MODEL_PRICING.get(model)


def calculate_turn_cost(
    usage: Usage,
    pricing: ModelPricing,
) -> TurnCost:
    """Calculate exact USD cost from one provider usage report."""
    token_counts = (
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_input_tokens,
        usage.cache_creation_input_tokens,
    )

    if any(count < 0 for count in token_counts):
        raise ValueError("token counts must not be negative")

    return TurnCost(
        input_cost_usd=_token_cost(
            usage.input_tokens,
            pricing.input_per_million,
        ),
        output_cost_usd=_token_cost(
            usage.output_tokens,
            pricing.output_per_million,
        ),
        cache_read_cost_usd=_token_cost(
            usage.cache_read_input_tokens,
            pricing.cache_read_per_million,
        ),
        cache_creation_cost_usd=_token_cost(
            usage.cache_creation_input_tokens,
            pricing.cache_creation_per_million,
        ),
    )


def calculate_known_model_cost(
    model: str,
    usage: Usage,
    pricing_catalog: Mapping[str, ModelPricing] = MODEL_PRICING,
) -> TurnCost | None:
    """
    Calculate cost for a known model.

    Returning None is deliberate: unknown pricing must never be displayed as
    a misleading $0 cost.
    """
    pricing = pricing_catalog.get(model)

    if pricing is None:
        return None

    return calculate_turn_cost(usage, pricing)


def _token_cost(
    token_count: int,
    price_per_million: Decimal,
) -> Decimal:
    return Decimal(token_count) * price_per_million / MILLION_TOKENS