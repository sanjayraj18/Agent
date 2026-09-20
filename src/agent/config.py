from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_PERMISSION_MODES = {
    "readonly",
    "ask",
    "auto",
    "full",
}

_SANDBOX_MODES = {
    "enforced",
    "disabled",
    "container",
}


DEFAULTS: dict[str, Any] = {
    "model": "claude-opus-5",
    "effort": "high",
    "max_tokens": 16_000,
    "permission_mode": "ask",
    "tool_permission_modes": {},
    "sandbox_mode": "enforced",
    "sandbox_network_allowed": False,
    # Kept relative to the workspace by default. This makes a project’s
    # durable conversations easy to back up, while an organization can point
    # it at a managed location with AGENT_SESSION_DATABASE_PATH.
    "session_database_path": ".agent/sessions.sqlite3",
    "log_level": "info",
}


ENV_MAP: dict[str, str] = {
    "AGENT_MODEL": "model",
    "AGENT_EFFORT": "effort",
    "AGENT_MAX_TOKENS": "max_tokens",
    "AGENT_PERMISSION_MODE": "permission_mode",
    "AGENT_TOOL_PERMISSION_MODES": "tool_permission_modes",
    "AGENT_LOG_LEVEL": "log_level",
    "AGENT_SANDBOX_MODE": "sandbox_mode",
    "AGENT_SANDBOX_NETWORK_ALLOWED": (
        "sandbox_network_allowed"
    ),
    "AGENT_SESSION_DATABASE_PATH": "session_database_path",
}


SYSTEM_CONFIG = Path("/etc/agent/config.json")
USER_CONFIG = Path.home() / ".config" / "agent" / "config.json"
PROJECT_CONFIG = Path(".agent/config.json")


class ConfigError(Exception):
    """Raised for malformed, unknown, or unsafe configuration."""


@dataclass(frozen=True)
class Resolved:
    value: Any
    layer: str
    origin: str


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
    ):
        return {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{path}: invalid JSON ({exc})"
        ) from exc

    if not isinstance(data, dict):
        raise ConfigError(
            f"{path}: expected a JSON object, got "
            f"{type(data).__name__}"
        )

    return data


def _coerce(key: str, raw: str) -> Any:
    prototype = DEFAULTS[key]

    if isinstance(prototype, bool):
        return raw.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    if isinstance(prototype, int):
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(
                f"{key}: expected an integer, got {raw!r}"
            ) from exc

    if isinstance(prototype, dict):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"{key}: expected a JSON object, got {raw!r}"
            ) from exc

        if not isinstance(value, dict):
            raise ConfigError(
                f"{key}: expected a JSON object, got "
                f"{type(value).__name__}"
            )

        return value

    return raw


def load(
    cwd: Path,
    flags: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Resolved]:
    env = os.environ if env is None else env

    resolved = {
        key: Resolved(value, "default", "built-in")
        for key, value in DEFAULTS.items()
    }

    for layer, path in (
        ("system", SYSTEM_CONFIG),
        ("user", USER_CONFIG),
        ("project", cwd / PROJECT_CONFIG),
    ):
        for key, value in _read_json(path).items():
            if key not in DEFAULTS:
                raise ConfigError(
                    f"{path}: unknown setting {key!r}"
                )

            resolved[key] = Resolved(value, layer, str(path))

    for variable, key in ENV_MAP.items():
        if variable in env:
            resolved[key] = Resolved(
                _coerce(key, env[variable]),
                "env",
                variable,
            )

    for key, value in (flags or {}).items():
        if value is None:
            continue

        if key not in DEFAULTS:
            raise ConfigError(
                f"unknown setting {key!r} passed as a flag"
            )

        resolved[key] = Resolved(
            value,
            "flag",
            f"--{key.replace('_', '-')}",
        )

    _validate_permission_settings(resolved)
    _validate_sandbox_settings(resolved)

    return resolved


def values(resolved: Mapping[str, Resolved]) -> dict[str, Any]:
    return {
        key: resolved_value.value
        for key, resolved_value in resolved.items()
    }


def render(resolved: Mapping[str, Resolved]) -> str:
    key_width = max(len(key) for key in resolved)
    value_width = max(
        len(repr(item.value))
        for item in resolved.values()
    )
    layer_width = max(
        len(item.layer)
        for item in resolved.values()
    )

    lines = [
        f"{'SETTING':<{key_width}}  "
        f"{'VALUE':<{value_width}}  "
        f"{'LAYER':<{layer_width}}  ORIGIN"
    ]

    for key in sorted(resolved):
        item = resolved[key]
        lines.append(
            f"{key:<{key_width}}  "
            f"{item.value!r:<{value_width}}  "
            f"{item.layer:<{layer_width}}  "
            f"{item.origin}"
        )

    return "\n".join(lines)


def _validate_sandbox_settings( resolved: Mapping[str, Resolved]) -> None:
    mode = resolved["sandbox_mode"].value

    if not isinstance(mode, str) or mode not in _SANDBOX_MODES:
        choices = ", ".join(sorted(_SANDBOX_MODES))
        raise ConfigError(
            "sandbox_mode must be one of: "
            f"{choices}; got {mode!r}"
        )

    network_allowed = resolved[
        "sandbox_network_allowed"
    ].value

    if not isinstance(network_allowed, bool):
        raise ConfigError(
            "sandbox_network_allowed must be a boolean"
        )

    
def _validate_permission_settings(
    resolved: Mapping[str, Resolved],
) -> None:
    default_mode = resolved["permission_mode"].value

    _validate_permission_mode(
        default_mode,
        setting="permission_mode",
    )

    tool_modes = resolved["tool_permission_modes"].value

    if not isinstance(tool_modes, dict):
        raise ConfigError(
            "tool_permission_modes must be a JSON object"
        )

    for tool_name, mode in tool_modes.items():
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ConfigError(
                "tool_permission_modes keys must be non-empty strings"
            )

        _validate_permission_mode(
            mode,
            setting=f"tool_permission_modes[{tool_name!r}]",
        )


def _validate_permission_mode(
    value: object,
    *,
    setting: str,
) -> None:
    if not isinstance(value, str) or value not in _PERMISSION_MODES:
        choices = ", ".join(sorted(_PERMISSION_MODES))
        raise ConfigError(
            f"{setting} must be one of: {choices}; got {value!r}"
        )
