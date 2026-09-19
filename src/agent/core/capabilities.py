from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping


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
class ModelCapabilities:
    """
    Facts the agent needs before it sends a request to a model.

    Keep these facts as data rather than scattering model-specific `if`
    statements across the loop, provider, and cost calculator.
    """

    model: str
    context_window_tokens: int
    max_output_tokens: int
    pricing: ModelPricing
    supports_prompt_caching: bool
    supports_thinking: bool
    supports_tools: bool

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")

        if self.context_window_tokens < 1:
            raise ValueError("context_window_tokens must be at least 1")

        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")

        if self.max_output_tokens > self.context_window_tokens:
            raise ValueError(
                "max_output_tokens cannot exceed context_window_tokens"
            )


MODEL_CAPABILITIES: Mapping[str, ModelCapabilities] = MappingProxyType(
    {
        "claude-opus-5": ModelCapabilities(
            model="claude-opus-5",
            context_window_tokens=200_000,
            max_output_tokens=64_000,
            pricing=ModelPricing(
                input_per_million=Decimal("5.00"),
                output_per_million=Decimal("25.00"),
                cache_read_per_million=Decimal("0.50"),
                cache_creation_per_million=Decimal("6.25"),
            ),
            supports_prompt_caching=True,
            supports_thinking=True,
            supports_tools=True,
        ),
        "claude-sonnet-5": ModelCapabilities(
            model="claude-sonnet-5",
            context_window_tokens=200_000,
            max_output_tokens=64_000,
            pricing=ModelPricing(
                input_per_million=Decimal("3.00"),
                output_per_million=Decimal("15.00"),
                cache_read_per_million=Decimal("0.30"),
                cache_creation_per_million=Decimal("3.75"),
            ),
            supports_prompt_caching=True,
            supports_thinking=True,
            supports_tools=True,
        ),
        "claude-haiku-4-5": ModelCapabilities(
            model="claude-haiku-4-5",
            context_window_tokens=200_000,
            max_output_tokens=64_000,
            pricing=ModelPricing(
                input_per_million=Decimal("1.00"),
                output_per_million=Decimal("5.00"),
                cache_read_per_million=Decimal("0.10"),
                cache_creation_per_million=Decimal("1.25"),
            ),
            supports_prompt_caching=True,
            supports_thinking=True,
            supports_tools=True,
        ),
    }
)


def capabilities_for_model(model: str) -> ModelCapabilities | None:
    return MODEL_CAPABILITIES.get(model)

def require_capabilities(model: str) -> ModelCapabilities:
    capabilities = capabilities_for_model(model)

    if capabilities is None:
        raise ValueError(f"unknown model capabilities: {model!r}")

    return capabilities