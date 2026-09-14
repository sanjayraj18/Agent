from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    TypeAdapter,
    field_validator,
)

DEFAULT_SKEW_SECONDS=60.0
OAUTH_BETA = "oauth-2025-04-20"


def _now() -> datetime:
    return datetime.now(timezone.utc)

def _fingerprint(secret : str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()[:8]


class _CredentialBase(BaseModel):
    model_config = ConfigDict(frozen = True)

    def headers(self) -> dict[str,str]:
        raise NotImplementedError

    def is_expired(self, *, now: datetime | None = None, skew: float = DEFAULT_SKEW_SECONDS) -> bool:
        raise NotImplementedError

    @property
    def fingerprint(self) -> str:
        raise NotImplementedError

    def to_storage(self) -> dict[str, Any]:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError


class ApiKey(_CredentialBase):
    type : Literal["api_key"] = "api_key"
    value : SecretStr

    def headers(self) -> dict[str, str]:
        return {"x-api-key" : self.value.get_secret_value()}

    def is_expired(self, *, now: datetime | None = None, skew: float = DEFAULT_SKEW_SECONDS) -> bool:
        return False

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.value.get_secret_value())

    def to_storage(self) -> dict[str, Any]:
        return {"type" : "api_key" , "value" : self.value.get_secret_value()}

    def describe(self) -> str:
        return f"api_key (fp {self.fingerprint})"


class OAuthToken(_CredentialBase):
    type: Literal["oauth"] = "oauth"
    access_token: SecretStr
    refresh_token: SecretStr | None = None
    expires_at: datetime | None = None
    scopes: tuple[str, ...] = ()

    @field_validator("expires_at")
    @classmethod
    def _force_utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token.get_secret_value()}",
            "anthropic-beta": OAUTH_BETA,
        }

    def is_expired(self, *, now: datetime | None = None, skew: float = DEFAULT_SKEW_SECONDS) -> bool:
        if self.expires_at is None:
            return False
        return (now or _now()) >= self.expires_at - timedelta(seconds=skew)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.access_token.get_secret_value())

    @property
    def can_refresh(self) -> bool:
        return self.refresh_token is not None

    def to_storage(self) -> dict[str, Any]:
        return {
            "type": "oauth",
            "access_token": self.access_token.get_secret_value(),
            "refresh_token": (
                self.refresh_token.get_secret_value() if self.refresh_token else None
            ),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "scopes": list(self.scopes),
        }

    def describe(self) -> str:
        when = self.expires_at.isoformat(timespec="seconds") if self.expires_at else "never"
        return f"oauth (fp {self.fingerprint}, expires {when})"


def from_storage(data: dict[str, Any]) -> Credential:
    return CredentialAdapter.validate_python(data)


Credential = Annotated[Union[ApiKey, OAuthToken], Field(discriminator="type")]
CredentialAdapter: TypeAdapter[Credential] = TypeAdapter(Credential)