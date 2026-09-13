import json
import os
from typing import Any, Mapping
from pathlib import Path
from dataclasses import dataclass

DEFAULTS : dict[str, Any] = { 
    "model": "claude-opus-5",
    "effort": "high",
    "max_tokens": 16000,
    "permission_mode": "ask",
    "log_level": "info",
}

ENV_MAP : dict[str, str] = {
    "AGENT_MODEL": "model",
    "AGENT_EFFORT": "effort",
    "AGENT_MAX_TOKENS": "max_tokens",
    "AGENT_PERMISSION_MODE": "permission_mode",
    "AGENT_LOG_LEVEL": "log_level",
}


SYSTEM_CONFIG = Path("/etc/agent/config.json")
USER_CONFIG= Path.home() / ".config" / "agent" / "config.json"
PROJECT_CONFIG = Path(".agent/config.json")


class ConfigError(Exception):
    """Raised for malformed or unknown configuration."""

@dataclass(frozen=True)
class Resolved:
    value : Any
    layer : str # which layer won: default | system | user | project | env | flag
    origin : str # the concrete source: a file path, an env var name, or a flag


def _read_json(path : Path) -> dict[str, Any]:
    try:
        raw = path.read_text()
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path}: invalid JSON ({e})") from e

    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a JSON object, got {type(data).__name__}")
    return data


def _coerce(key: str, raw: str) -> Any:
    proto = DEFAULTS[key]
    if isinstance(proto, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(proto, int):
        try:
            return int(raw)
        except ValueError as e:
            raise ConfigError(f"{key}: expected an integer, got {raw!r}") from e
    return raw



def load(cwd : Path, flags: Mapping[str, Any] | None = None, env : Mapping[str, str] | None = None):
    env = os.environ if env is None else env

    resolved = {k : Resolved(v, "default","built-in") for k,v in DEFAULTS.items()}

    for layer,path in (
        ("system", SYSTEM_CONFIG),
        ("user",    USER_CONFIG),
        ("project", cwd / PROJECT_CONFIG)
    ):
        for key, value in _read_json(path).items():
            if key not in DEFAULTS:
                raise ConfigError(f"{path}: unknown setting {key!r}")
            resolved[key] = Resolved(value, layer, str(path))

    for var, key in ENV_MAP.items():
        if var in env:
            resolved[key] = Resolved(_coerce(key, env[var]), "env", var)

    for key, value in (flags or {}).items():
        if value is None:
            continue  # argparse gives None for unset flags — not an override
        if key not in DEFAULTS:
            raise ConfigError(f"unknown setting {key!r} passed as a flag")
        resolved[key] = Resolved(value, "flag", f"--{key.replace('_', '-')}")

    return resolved



def values(resolved: dict[str, Resolved]) -> dict[str, Any]:
    return {k: r.value for k, r in resolved.items()}



def render(resolved: dict[str, Resolved]) -> str:
    kw = max(len(k) for k in resolved)
    vw = max(len(repr(r.value)) for r in resolved.values())
    lw = max(len(r.layer) for r in resolved.values())
    lines = [f"{'SETTING':<{kw}}  {'VALUE':<{vw}}  {'LAYER':<{lw}}  ORIGIN"]
    for key in sorted(resolved):
        r = resolved[key]
        lines.append(f"{key:<{kw}}  {r.value!r:<{vw}}  {r.layer:<{lw}}  {r.origin}")
    return "\n".join(lines)
