from decimal import Decimal

import pytest

from agent.core.capabilities import (
    MODEL_CAPABILITIES,
    ModelCapabilities,
    ModelPricing,
    capabilities_for_model,
    require_capabilities,
)


def _pricing() -> ModelPricing:
    return ModelPricing(
        input_per_million=Decimal("2"),
        output_per_million=Decimal("10"),
        cache_read_per_million=Decimal("0.2"),
        cache_creation_per_million=Decimal("2.5"),
    )


def test_finds_known_model_capabilities():
    capabilities = capabilities_for_model("claude-sonnet-5")

    assert capabilities is not None
    assert capabilities.model == "claude-sonnet-5"
    assert capabilities.context_window_tokens == 200_000
    assert capabilities.max_output_tokens == 64_000
    assert capabilities.supports_prompt_caching is True
    assert capabilities.supports_thinking is True
    assert capabilities.supports_tools is True


def test_catalog_key_and_model_name_are_consistent():
    for model, capabilities in MODEL_CAPABILITIES.items():
        assert capabilities.model == model


def test_model_capabilities_include_token_pricing():
    capabilities = require_capabilities("claude-haiku-4-5")

    assert capabilities.pricing.input_per_million == Decimal("1.00")
    assert capabilities.pricing.output_per_million == Decimal("5.00")
    assert capabilities.pricing.cache_read_per_million == Decimal("0.10")
    assert capabilities.pricing.cache_creation_per_million == Decimal("1.25")


def test_returns_none_for_unknown_model():
    assert capabilities_for_model("unknown-model") is None


def test_require_capabilities_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown model capabilities"):
        require_capabilities("unknown-model")


def test_rejects_an_empty_model_name():
    with pytest.raises(ValueError, match="model must not be empty"):
        ModelCapabilities(
            model="   ",
            context_window_tokens=200_000,
            max_output_tokens=64_000,
            pricing=_pricing(),
            supports_prompt_caching=True,
            supports_thinking=True,
            supports_tools=True,
        )


def test_rejects_an_output_limit_larger_than_context_window():
    with pytest.raises(
        ValueError,
        match="max_output_tokens cannot exceed context_window_tokens",
    ):
        ModelCapabilities(
            model="test-model",
            context_window_tokens=1_000,
            max_output_tokens=1_001,
            pricing=_pricing(),
            supports_prompt_caching=False,
            supports_thinking=False,
            supports_tools=False,
        )