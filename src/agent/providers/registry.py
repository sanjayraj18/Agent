from __future__ import annotations

from agent.auth.credentials import ApiKey, Credential, OAuthToken, OpenAIApiKey
from agent.providers.anthropic_raw import AnthropicRawProvider
from agent.providers.base import Provider
from agent.providers.openai_responses import OpenAIResponsesProvider
from agent.providers.profiles import PROVIDER_PROFILES, ProviderId


class ProviderRegistryError(ValueError):
    """Raised for unknown providers or credentials for the wrong provider."""


def create_provider(
    provider_id: ProviderId,
    credential: Credential,
    *,
    base_url: str | None = None,
) -> Provider:
    """Construct exactly one adapter from provider-neutral configuration."""

    profile = PROVIDER_PROFILES[provider_id]
    endpoint = base_url or profile.default_base_url

    if provider_id == "anthropic":
        if not isinstance(credential, (ApiKey, OAuthToken)):
            raise ProviderRegistryError(
                "Anthropic requires an Anthropic API key or OAuth token"
            )
        return AnthropicRawProvider(credential, base_url=endpoint)

    if provider_id == "openai":
        if not isinstance(credential, OpenAIApiKey):
            raise ProviderRegistryError("OpenAI requires an OpenAI API key")
        return OpenAIResponsesProvider(credential, base_url=endpoint)

    raise ProviderRegistryError(f"unsupported provider: {provider_id!r}")
