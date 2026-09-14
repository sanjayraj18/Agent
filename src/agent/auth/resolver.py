

from dataclasses import dataclass
import os
from typing import Callable, Mapping

from pydantic import SecretStr

from agent.auth.credentials import ApiKey, Credential, OAuthToken

API_KEY_VAR = "ANTHROPIC_API_KEY"
AUTH_TOKEN_VAR = "ANTHROPIC_AUTH_TOKEN"

class CredentialError(Exception):
    """No usable credential, or a credential that is present but unusable."""


@dataclass(frozen=True)
class ResolvedCredential:
    credential: Credential
    source: str   # flag | env | store
    origin: str   # --api-key | ANTHROPIC_API_KEY | credential store

    def describe(self) -> str:
        return f"{self.credential.describe()}  [{self.source}] {self.origin}"

def resolve(
    *,
    api_key: str | None = None,
    env: Mapping[str, str] | None = None,
    store: Callable[[], Credential | None] | None = None,
) -> ResolvedCredential:
    env = os.environ if env is None else env

    if api_key is not None:
        return ResolvedCredential(
            ApiKey(value=SecretStr(_require_nonempty(api_key, "--api-key"))), "flag", "--api-key"
        )

    if API_KEY_VAR in env and AUTH_TOKEN_VAR in env:
        # The API rejects a request carrying both auth schemes. We only ever
        # send one, but the ambiguity means someone's intent is being ignored.
        raise CredentialError(
            f"both {API_KEY_VAR} and {AUTH_TOKEN_VAR} are set. "
            f"Unset one — they are different authentication schemes."
        )

    if API_KEY_VAR in env:
        value = _require_nonempty(env[API_KEY_VAR], API_KEY_VAR)
        return ResolvedCredential(ApiKey(value=SecretStr(value)), "env", API_KEY_VAR)

    if AUTH_TOKEN_VAR in env:
        value = _require_nonempty(env[AUTH_TOKEN_VAR], AUTH_TOKEN_VAR)
        return ResolvedCredential(OAuthToken(access_token=SecretStr(value)), "env", AUTH_TOKEN_VAR)

    if store is not None and (cred := store()) is not None:
        return ResolvedCredential(cred, "store", "credential store")

    raise CredentialError(
        "no credential found. Set ANTHROPIC_API_KEY, or pass --api-key."
    )


def _require_nonempty(raw: str, origin: str) -> str:
    value = raw.strip()
    if not value:
        raise CredentialError(
            f"{origin} is set but empty. Unset it (`unset {origin}`) or give it a "
            f"value — an empty credential still takes precedence over every "
            f"source below it."
        )
    return value