from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

from agent.auth.credentials import Credential, from_storage

FORMAT_VERSION = 1


class StoreError(Exception):
    """The credential file exists but cannot be used."""


def default_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "agent" / "credentials.json"


class FileStore:
    """A single credential in a 0600 file. Injectable path so tests never
    touch the real one."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or default_path()

    @property
    def path(self) -> Path:
        return self._path


    def load(self) -> Credential | None:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except PermissionError as exc:
            raise StoreError(f"{self._path}: cannot read ({exc})") from exc

        self._require_private()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StoreError(
                f"{self._path}: corrupt credential file ({exc}). "
                f"Run `agent auth logout` then `agent auth login`."
            ) from exc

        if data.get("version") != FORMAT_VERSION:
            raise StoreError(
                f"{self._path}: unsupported format version {data.get('version')!r}"
            )

        try:
            return from_storage(data["credential"])
        except Exception as exc:
            raise StoreError(f"{self._path}: unreadable credential ({exc})") from exc

   

    def save(self, credential: Credential) -> None:
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            os.chmod(parent, 0o700)

        payload = json.dumps(
            {"version": FORMAT_VERSION, "credential": credential.to_storage()},
            indent=2,
        )

   
        fd, tmp_name = tempfile.mkstemp(
            dir=parent, prefix=".credentials-", suffix=".tmp"
        )
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

  

    def delete(self) -> bool:
        """Remove the stored credential. Returns False if there wasn't one.

        This does not attempt to scrub the bytes from the disk -- on a
        copy-on-write or wear-levelling filesystem, overwriting in place does
        not reliably erase anything. Rotate the credential at the provider if
        it may have been exposed.
        """
        try:
            self._path.unlink()
            return True
        except FileNotFoundError:
            return False



    def _require_private(self) -> None:
        """Refuse to read a credential that other users can read.

        Same stance as ssh with private keys: loose permissions mean the file
        should be treated as compromised, not quietly used.
        """
        if os.name == "nt":
            return  
        mode = stat.S_IMODE(self._path.stat().st_mode)
        if mode & 0o077:
            raise StoreError(
                f"{self._path} is readable by other users (mode {mode:o}). "
                f"Fix with: chmod 600 {self._path}"
            )