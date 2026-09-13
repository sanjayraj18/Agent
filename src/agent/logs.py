from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

def _keep_prefix(prefix_len: int):
    """Show enough of a credential to identify it, never enough to use it."""
    def repl(m: re.Match[str]) -> str:
        s = m.group(0)
        return f"{s[:prefix_len]}***"
    return repl

_PATTERNS: list[tuple[re.Pattern[str], object]] = [
    # Anthropic — API keys and OAuth access tokens
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"), _keep_prefix(7)),
    # OpenAI and lookalikes
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), _keep_prefix(3)),
    # GitHub
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}"), _keep_prefix(4)),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), _keep_prefix(11)),
    # AWS access key id
    (re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), _keep_prefix(4)),
    # Slack
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"), _keep_prefix(5)),
    # JWTs
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
     _keep_prefix(6)),
    # Authorization headers, any scheme
    (re.compile(r"(?i)\b(authorization\s*[:=]\s*)(?:bearer\s+)?\S+"), r"\1***"),
    (re.compile(r"(?i)\b(x-api-key\s*[:=]\s*)\S+"), r"\1***"),
    # Generic key=value / "key": "value" shapes, whatever the value looks like
     (re.compile(
        r"(?i)([\"']?\b(?:api[_-]?key|secret|password|passwd|token|"
        r"refresh[_-]?token|access[_-]?token|client[_-]?secret)\b[\"']?"
        r"\s*[:=]\s*)([\"']?)(?![^\s,;}\"']*\*\*\*)([^\s,;}\"']+)\2"),
     r"\1\2***\2"),
]

def redact(text: str) -> str:
    """Scrub credential-shaped substrings. The only exit for log output."""
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)  # type: ignore[arg-type]
    return text


_STANDARD_ATTRS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
})

def _extras(record: logging.LogRecord) -> dict[str, object]:
    return {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(_extras(record))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload, default=str))


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{record.levelname.lower():<7} {record.name}  {record.getMessage()}"
        extras = _extras(record)
        if extras:
            base += "  " + " ".join(f"{k}={v!r}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return redact(base)

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

def setup(level: str = "info", log_file: Path | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(_LEVELS.get(level, logging.INFO))

    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(ConsoleFormatter())
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"agent.{name}")