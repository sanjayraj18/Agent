"""A private, atomic credential store with one entry per provider."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, cast

from agent.auth.credentials import (
    ApiKey,
    Credential,
    OAuthToken,
    OpenAIApiKey,
    from_storage,
)
from agent.providers.profiles import ProviderId

FORMAT_VERSION = 2
_LEGACY_FORMAT_VERSION = 1


class StoreError(Exception):
    """The credential file exists but cannot be used."""


def default_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "agent" / "credentials.json"


class FileStore:
    """Provider-keyed credentials in a 0600 file.

    Version 1 held a single Anthropic credential. Version 2 keeps separate
    entries, which prevents saving an OpenAI key from replacing that credential.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or default_path()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> Credential | None:
        """Compatibility helper for the original Anthropic-only caller."""
        return self.load_for_provider("anthropic")

    def load_for_provider(self, provider: ProviderId) -> Credential | None:
        return self.load_all().get(provider)

    def load_all(self) -> dict[ProviderId, Credential]:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except PermissionError as exc:
            raise StoreError(f"{self._path}: cannot read ({exc})") from exc

        self._require_private()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StoreError(
                f"{self._path}: corrupt credential file ({exc}). "
                "Run `agent auth logout` then `agent auth login`."
            ) from exc

        if not isinstance(data, dict):
            raise StoreError(f"{self._path}: corrupt credential file (root must be an object)")

        version = data.get("version")
        if version == _LEGACY_FORMAT_VERSION:
            credentials_data: dict[str, Any] = {"anthropic": data.get("credential")}
        elif version == FORMAT_VERSION:
            raw_credentials = data.get("credentials")
            if not isinstance(raw_credentials, dict):
                raise StoreError(f"{self._path}: unreadable credential (credentials must be an object)")
            credentials_data = raw_credentials
        else:
            raise StoreError(f"{self._path}: unsupported format version {version!r}")

        credentials: dict[ProviderId, Credential] = {}
        for provider_name, encoded in credentials_data.items():
            if provider_name not in {"anthropic", "openai"}:
                raise StoreError(f"{self._path}: unknown credential provider {provider_name!r}")
            if not isinstance(encoded, dict):
                raise StoreError(f"{self._path}: unreadable credential for {provider_name}")
            provider = cast(ProviderId, provider_name)
            try:
                credential = from_storage(encoded)
            except Exception as exc:
                raise StoreError(
                    f"{self._path}: unreadable credential ({exc})"
                ) from exc
            _validate_provider_credential(provider, credential)
            credentials[provider] = credential
        return credentials

    def save(self, credential: Credential, *, provider: ProviderId | None = None) -> None:
        provider_id = provider or _provider_for_credential(credential)
        _validate_provider_credential(provider_id, credential)
        credentials = self.load_all()
        credentials[provider_id] = credential
        self._write(credentials)

    def delete(self) -> bool:
        """Remove every stored credential (legacy logout behavior)."""
        try:
            self._path.unlink()
            return True
        except FileNotFoundError:
            return False

    def delete_for_provider(self, provider: ProviderId) -> bool:
        credentials = self.load_all()
        if provider not in credentials:
            return False
        del credentials[provider]
        if credentials:
            self._write(credentials)
        else:
            self.delete()
        return True

    def _write(self, credentials: dict[ProviderId, Credential]) -> None:
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            os.chmod(parent, 0o700)

        payload = json.dumps(
            {
                "version": FORMAT_VERSION,
                "credentials": {
                    provider: credential.to_storage()
                    for provider, credential in credentials.items()
                },
            },
            indent=2,
        )
        fd, tmp_name = tempfile.mkstemp(dir=parent, prefix=".credentials-", suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                fd = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self._path)
            tmp = None
        finally:
            if fd >= 0:
                os.close(fd)
            if tmp is not None and tmp.exists():
                tmp.unlink()

    def _require_private(self) -> None:
        if os.name == "nt":
            return
        mode = stat.S_IMODE(self._path.stat().st_mode)
        if mode & 0o077:
            raise StoreError(
                f"{self._path} is readable by other users (mode {mode:o}). "
                f"Fix with: chmod 600 {self._path}"
            )


def _provider_for_credential(credential: Credential) -> ProviderId:
    if isinstance(credential, OpenAIApiKey):
        return "openai"
    return "anthropic"


def _validate_provider_credential(provider: ProviderId, credential: Credential) -> None:
    if provider == "openai" and not isinstance(credential, OpenAIApiKey):
        raise StoreError("OpenAI credentials must use the openai_api_key type")
    if provider == "anthropic" and not isinstance(credential, (ApiKey, OAuthToken)):
        raise StoreError("Anthropic credentials must use api_key or oauth type")
