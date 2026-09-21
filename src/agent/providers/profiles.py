from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


ProviderId = Literal["anthropic", "openai"]
ConfiguredProviderId = ProviderId | Literal["auto"]


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    """Static connection facts for one provider family."""

    provider_id: ProviderId
    default_base_url: str
    model_prefixes: tuple[str, ...]
    supports_streaming: bool = True
    supports_tools: bool = True


PROVIDER_PROFILES: Mapping[ProviderId, ProviderProfile] = MappingProxyType(
    {
        "anthropic": ProviderProfile(
            provider_id="anthropic",
            default_base_url="https://api.anthropic.com",
            model_prefixes=("claude-",),
        ),
        "openai": ProviderProfile(
            provider_id="openai",
            default_base_url="https://api.openai.com",
            model_prefixes=("gpt-", "o1", "o3", "o4"),
        ),
    }
)


def provider_for_model(model: str) -> ProviderId | None:
    """Infer a provider only for familiar model-name families."""

    normalized = model.strip().lower()

    for provider, profile in PROVIDER_PROFILES.items():
        if normalized.startswith(profile.model_prefixes):
            return provider
    return None


def resolve_provider(
    configured_provider: str,
    model: str,
) -> ProviderId:
    """Resolve explicit configuration, rejecting known unsafe mismatches."""

    if configured_provider == "auto":
        inferred = provider_for_model(model)
        if inferred is None:
            raise ValueError(
                "could not infer a provider for model "
                f"{model!r}; set provider explicitly"
            )
        return inferred

    if configured_provider not in PROVIDER_PROFILES:
        raise ValueError(
            "provider must be one of: anthropic, openai, auto; got "
            f"{configured_provider!r}"
        )

    if configured_provider == "anthropic":
        provider: ProviderId = "anthropic"
    elif configured_provider == "openai":
        provider = "openai"
    else:  # Membership check above keeps this branch defensive.
        raise ValueError(f"unsupported provider {configured_provider!r}")
    inferred = provider_for_model(model)
    if inferred is not None and inferred != provider:
        raise ValueError(
            f"model {model!r} belongs to {inferred}, not {provider}"
        )

    return provider
