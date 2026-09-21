"""Resolve credentials without ever mixing provider authentication schemes."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Mapping

from pydantic import SecretStr

from agent.auth.credentials import ApiKey, Credential, OAuthToken, OpenAIApiKey
from agent.providers.profiles import ProviderId

ANTHROPIC_API_KEY_VAR = "ANTHROPIC_API_KEY"
ANTHROPIC_AUTH_TOKEN_VAR = "ANTHROPIC_AUTH_TOKEN"
OPENAI_API_KEY_VAR = "OPENAI_API_KEY"

# Compatibility names for code that predates provider selection.
API_KEY_VAR = ANTHROPIC_API_KEY_VAR
AUTH_TOKEN_VAR = ANTHROPIC_AUTH_TOKEN_VAR


class CredentialError(Exception):
    """No usable credential, or a credential that is present but unusable."""


@dataclass(frozen=True)
class ResolvedCredential:
    credential: Credential
    source: str  # flag | env | store
    origin: str  # --api-key | provider environment variable | credential store

    def describe(self) -> str:
        return f"{self.credential.describe()}  [{self.source}] {self.origin}"


def resolve(
    *,
    api_key: str | None = None,
    env: Mapping[str, str] | None = None,
    store: Callable[[], Credential | None] | None = None,
) -> ResolvedCredential:
    """Resolve Anthropic credentials (the original public API)."""
    return resolve_for_provider(
        "anthropic",
        api_key=api_key,
        env=env,
        store=store,
    )


def resolve_for_provider(
    provider: ProviderId,
    *,
    api_key: str | None = None,
    env: Mapping[str, str] | None = None,
    store: Callable[[], Credential | None] | None = None,
) -> ResolvedCredential:
    """Apply the normal precedence chain for one selected provider.

    Precedence is explicit command-line key, environment, then the local
    credential store. The selected provider decides what a key means.
    """
    environment = os.environ if env is None else env
    if provider == "anthropic":
        return _resolve_anthropic(api_key=api_key, env=environment, store=store)
    if provider == "openai":
        return _resolve_openai(api_key=api_key, env=environment, store=store)
    raise CredentialError(f"unsupported provider {provider!r}")


def _resolve_anthropic(
    *,
    api_key: str | None,
    env: Mapping[str, str],
    store: Callable[[], Credential | None] | None,
) -> ResolvedCredential:
    if api_key is not None:
        return ResolvedCredential(
            ApiKey(value=SecretStr(_require_nonempty(api_key, "--api-key"))),
            "flag",
            "--api-key",
        )

    if ANTHROPIC_API_KEY_VAR in env and ANTHROPIC_AUTH_TOKEN_VAR in env:
        raise CredentialError(
            f"both {ANTHROPIC_API_KEY_VAR} and {ANTHROPIC_AUTH_TOKEN_VAR} are set. "
            "Unset one — they are different authentication schemes."
        )

    if ANTHROPIC_API_KEY_VAR in env:
        value = _require_nonempty(env[ANTHROPIC_API_KEY_VAR], ANTHROPIC_API_KEY_VAR)
        return ResolvedCredential(ApiKey(value=SecretStr(value)), "env", ANTHROPIC_API_KEY_VAR)

    if ANTHROPIC_AUTH_TOKEN_VAR in env:
        value = _require_nonempty(
            env[ANTHROPIC_AUTH_TOKEN_VAR],
            ANTHROPIC_AUTH_TOKEN_VAR,
        )
        return ResolvedCredential(
            OAuthToken(access_token=SecretStr(value)),
            "env",
            ANTHROPIC_AUTH_TOKEN_VAR,
        )

    if store is not None and (credential := store()) is not None:
        if isinstance(credential, (ApiKey, OAuthToken)):
            return ResolvedCredential(credential, "store", "credential store")
        raise CredentialError("credential store contains an OpenAI key, not an Anthropic credential")

    raise CredentialError("no credential found. Set ANTHROPIC_API_KEY, or pass --api-key.")


def _resolve_openai(
    *,
    api_key: str | None,
    env: Mapping[str, str],
    store: Callable[[], Credential | None] | None,
) -> ResolvedCredential:
    if api_key is not None:
        return ResolvedCredential(
            OpenAIApiKey(value=SecretStr(_require_nonempty(api_key, "--api-key"))),
            "flag",
            "--api-key",
        )

    if OPENAI_API_KEY_VAR in env:
        value = _require_nonempty(env[OPENAI_API_KEY_VAR], OPENAI_API_KEY_VAR)
        return ResolvedCredential(
            OpenAIApiKey(value=SecretStr(value)),
            "env",
            OPENAI_API_KEY_VAR,
        )

    if store is not None and (credential := store()) is not None:
        if isinstance(credential, OpenAIApiKey):
            return ResolvedCredential(credential, "store", "credential store")
        raise CredentialError("credential store contains an Anthropic credential, not an OpenAI key")

    raise CredentialError("no credential found. Set OPENAI_API_KEY, or pass --api-key.")


def _require_nonempty(raw: str, origin: str) -> str:
    value = raw.strip()
    if not value:
        raise CredentialError(
            f"{origin} is set but empty. Unset it (`unset {origin}`) or give it a "
            "value — an empty credential still takes precedence over every "
            "source below it."
        )
    return value
