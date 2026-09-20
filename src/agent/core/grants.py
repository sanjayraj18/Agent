from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.core.permission import ToolAction
from agent.tools.shell_syntax import analyze_shell


GRANTS_VERSION = 1
GRANTS_DIRECTORY = ".agent"
GRANTS_FILENAME = "permissions.json"


class GrantStoreError(ValueError):
    """Raised when persisted project permission grants are unsafe or invalid."""


@dataclass(frozen=True, slots=True)
class CommandGrant:
    """
    Permission for one safe Bash command pattern.

    Example:
        argv_prefix=("uv", "run", "pytest")
        allow_extra_args=True

    matches:
        uv run pytest
        uv run pytest -q

    It does not match:
        uv run pytest && rm -rf .
        uv run pytest > result.txt
        FOO=bar uv run pytest
    """

    argv_prefix: tuple[str, ...]
    allow_extra_args: bool = False

    def __post_init__(self) -> None:
        if not self.argv_prefix:
            raise GrantStoreError(
                "command grant argv_prefix must not be empty"
            )

        for argument in self.argv_prefix:
            if not argument:
                raise GrantStoreError(
                    "command grant arguments must not be empty"
                )

            if "\x00" in argument:
                raise GrantStoreError(
                    "command grant arguments must not contain null bytes"
                )

    def allows(self, action: ToolAction) -> bool:
        """Return true only for one matching, simple foreground Bash command."""
        if action.tool_name != "bash":
            return False

        bash_action = action.arguments.get("action", "run")

        if bash_action != "run":
            return False

        if action.arguments.get("background", False) is not False:
            return False

        raw_command = action.arguments.get("command")

        if not isinstance(raw_command, str):
            return False

        analysis = analyze_shell(raw_command)

        if (
            not analysis.is_safe_for_allowlist
            or analysis.command is None
            or analysis.command.environment
        ):
            return False

        argv = analysis.command.argv

        if argv[: len(self.argv_prefix)] != self.argv_prefix:
            return False

        return self.allow_extra_args or len(argv) == len(
            self.argv_prefix
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "argv_prefix": list(self.argv_prefix),
            "allow_extra_args": self.allow_extra_args,
        }

    @classmethod
    def from_json(cls, value: object) -> CommandGrant:
        if not isinstance(value, dict):
            raise GrantStoreError(
                "command grant must be a JSON object"
            )

        if set(value) != {"argv_prefix", "allow_extra_args"}:
            raise GrantStoreError(
                "command grant must contain exactly argv_prefix and "
                "allow_extra_args"
            )

        raw_prefix = value["argv_prefix"]
        allow_extra_args = value["allow_extra_args"]

        if (
            not isinstance(raw_prefix, list)
            or not all(isinstance(argument, str) for argument in raw_prefix)
        ):
            raise GrantStoreError(
                "command grant argv_prefix must be a list of strings"
            )

        if not isinstance(allow_extra_args, bool):
            raise GrantStoreError(
                "command grant allow_extra_args must be a boolean"
            )

        return cls(
            argv_prefix=tuple(raw_prefix),
            allow_extra_args=allow_extra_args,
        )


class ProjectGrantStore:
    """
    Project-local store for approved Bash command patterns.

    The store implements `GrantChecker`, so PermissionPolicy can call
    `store.allows(action)` without needing to know how JSON persistence works.
    """

    def __init__(
        self,
        workspace_root: Path,
        grants: tuple[CommandGrant, ...] = (),
    ) -> None:
        root = workspace_root.resolve()

        if not root.is_dir():
            raise GrantStoreError(
                f"workspace root is not a directory: {root}"
            )

        self._workspace_root = root
        self._grants = grants

    @property
    def path(self) -> Path:
        return (
            self._workspace_root
            / GRANTS_DIRECTORY
            / GRANTS_FILENAME
        )

    @property
    def grants(self) -> tuple[CommandGrant, ...]:
        return self._grants

    @classmethod
    def load(cls, workspace_root: Path) -> ProjectGrantStore:
        """Load grants if present; a missing grants file means no grants."""
        store = cls(workspace_root)
        path = store.path

        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return store
        except OSError as exc:
            raise GrantStoreError(
                f"could not read project grants: {exc}"
            ) from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GrantStoreError(
                f"{path}: invalid JSON ({exc})"
            ) from exc

        store._grants = _parse_grants_payload(payload)
        return store

    def allows(self, action: ToolAction) -> bool:
        """Return true when any persisted grant matches this exact action."""
        return any(grant.allows(action) for grant in self._grants)

    def add(self, grant: CommandGrant) -> None:
        """
        Save one new project grant.

        Adding the same grant twice is harmless and does not create duplicate
        permission records.
        """
        if grant in self._grants:
            return

        self._grants = (*self._grants, grant)
        self.save()

    def save(self) -> None:
        """Atomically write private project-local permission data."""
        directory = self.path.parent

        if directory.exists() and directory.is_symlink():
            raise GrantStoreError(
                f"grants directory must not be a symlink: {directory}"
            )

        directory.mkdir(mode=0o700, parents=True, exist_ok=True)

        if self.path.exists() and self.path.is_symlink():
            raise GrantStoreError(
                f"grants file must not be a symlink: {self.path}"
            )

        payload = {
            "version": GRANTS_VERSION,
            "bash_command_grants": [
                grant.to_json()
                for grant in self._grants
            ],
        }
        encoded = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".permissions-",
            dir=directory,
        )
        temporary_path = Path(temporary_name)

        try:
            os.fchmod(descriptor, 0o600)

            with os.fdopen(descriptor, "wb") as file:
                file.write(encoded)

            os.replace(temporary_path, self.path)

        except OSError as exc:
            raise GrantStoreError(
                f"could not write project grants: {exc}"
            ) from exc

        finally:
            temporary_path.unlink(missing_ok=True)


def _parse_grants_payload(
    payload: object,
) -> tuple[CommandGrant, ...]:
    if not isinstance(payload, dict):
        raise GrantStoreError(
            "project grants must be a JSON object"
        )

    if set(payload) != {"version", "bash_command_grants"}:
        raise GrantStoreError(
            "project grants must contain exactly version and "
            "bash_command_grants"
        )

    if payload["version"] != GRANTS_VERSION:
        raise GrantStoreError(
            f"unsupported project grants version: {payload['version']!r}"
        )

    raw_grants = payload["bash_command_grants"]

    if not isinstance(raw_grants, list):
        raise GrantStoreError(
            "bash_command_grants must be a JSON list"
        )

    grants = tuple(
        CommandGrant.from_json(raw_grant)
        for raw_grant in raw_grants
    )

    if len(set(grants)) != len(grants):
        raise GrantStoreError(
            "project grants must not contain duplicates"
        )

    return grants