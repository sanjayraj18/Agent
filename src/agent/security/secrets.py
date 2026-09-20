from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Pattern

#Input:
#OPENAI_API_KEY=sk-proj-very-secret-value

#Output:
#OPENAI_API_KEY=[REDACTED_SECRET]


class SecretKind(StrEnum):
    PRIVATE_KEY = "private_key"
    ANTHROPIC_KEY = "anthropic_key"
    OPENAI_KEY = "openai_key"
    GITHUB_TOKEN = "github_token"
    AWS_KEY = "aws_key"
    SLACK_TOKEN = "slack_token"
    JWT = "jwt"
    AUTHORIZATION_HEADER = "authorization_header"
    API_KEY_HEADER = "api_key_header"
    NAMED_SECRET = "named_secret"


@dataclass(frozen=True, slots=True)
class SecretMatch:
    """A secret-shaped span found in the original text."""

    start: int
    end: int
    kinds: tuple[SecretKind, ...]


@dataclass(frozen=True, slots=True)
class SecretScan:
    """The safe text and metadata produced by one scan."""

    redacted_text: str
    matches: tuple[SecretMatch, ...]

    @property
    def was_redacted(self) -> bool:
        return bool(self.matches)

    @property
    def redaction_count(self) -> int:
        return len(self.matches)


@dataclass(frozen=True, slots=True)
class _SecretPattern:
    kind: SecretKind
    expression: Pattern[str]


_PATTERNS = (
    _SecretPattern(
        SecretKind.PRIVATE_KEY,
        re.compile(
            r"(?P<secret>"
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"
            r"[\s\S]*?"
            r"-----END [A-Z0-9 ]*PRIVATE KEY-----"
            r")"
        ),
    ),
    _SecretPattern(
        SecretKind.ANTHROPIC_KEY,
        re.compile(
            r"(?P<secret>\bsk-ant-[A-Za-z0-9_-]{8,})"
        ),
    ),
    _SecretPattern(
        SecretKind.OPENAI_KEY,
        re.compile(
            r"(?P<secret>\bsk-(?:proj-)?[A-Za-z0-9_-]{16,})"
        ),
    ),
    _SecretPattern(
        SecretKind.GITHUB_TOKEN,
        re.compile(
            r"(?P<secret>"
            r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}"
            r"|\bgithub_pat_[A-Za-z0-9_]{20,}"
            r")"
        ),
    ),
    _SecretPattern(
        SecretKind.AWS_KEY,
        re.compile(
            r"(?P<secret>\b(?:AKIA|ASIA)[A-Z0-9]{16}\b)"
        ),
    ),
    _SecretPattern(
        SecretKind.SLACK_TOKEN,
        re.compile(
            r"(?P<secret>\bxox[baprs]-[A-Za-z0-9-]{10,})"
        ),
    ),
    _SecretPattern(
        SecretKind.JWT,
        re.compile(
            r"(?P<secret>"
            r"\beyJ[A-Za-z0-9_-]{10,}"
            r"\.[A-Za-z0-9_-]+"
            r"\.[A-Za-z0-9_-]+"
            r")"
        ),
    ),
    _SecretPattern(
        SecretKind.AUTHORIZATION_HEADER,
        re.compile(
            r"(?im)"
            r"^\s*authorization\s*:\s*"
            r"(?:bearer|basic|token)\s+"
            r"(?P<secret>\S+)"
        ),
    ),
    _SecretPattern(
        SecretKind.API_KEY_HEADER,
        re.compile(
            r"(?im)"
            r"^\s*x-api-key\s*:\s*"
            r"(?P<secret>\S+)"
        ),
    ),
    _SecretPattern(
        SecretKind.NAMED_SECRET,
        re.compile(
            r"""
            (?:
                ^ | [\s,{?&]
            )
            (?:export\s+)?
            (?:
                [A-Za-z_][A-Za-z0-9_]*
                (?:
                    api[_-]?key
                    | token
                    | secret
                    | password
                    | passwd
                    | private[_-]?key
                    | client[_-]?secret
                    | database[_-]?url
                )
                | api[_-]?key
                | secret
                | password
                | passwd
                | token
            )
            \s*[:=]\s*
            (?P<secret>
                "(?:[^"\\\r\n]|\\.)*"
                | '(?:[^'\\\r\n]|\\.)*'
                | [^\s,;}\]]+
            )
            """,
            re.IGNORECASE | re.MULTILINE | re.VERBOSE,
        ),
    ),
)


class SecretScanner:
    """Detect and redact secret-shaped text before it reaches the model."""

    def __init__(
        self,
        replacement: str = "[REDACTED_SECRET]",
    ) -> None:
        if not replacement:
            raise ValueError("replacement must not be empty")

        if "\n" in replacement or "\r" in replacement:
            raise ValueError(
                "replacement must not contain a newline"
            )

        self._replacement = replacement

    def scan(self, text: str) -> SecretScan:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        matches = _find_matches(text)
        redacted_text = _redact(
            text,
            matches,
            replacement=self._replacement,
        )

        return SecretScan(
            redacted_text=redacted_text,
            matches=matches,
        )

    def redact(self, text: str) -> str:
        """Return safe text when match details are unnecessary."""

        return self.scan(text).redacted_text


def _find_matches(text: str) -> tuple[SecretMatch, ...]:
    candidates: list[
        tuple[int, int, SecretKind]
    ] = []

    for pattern in _PATTERNS:
        for match in pattern.expression.finditer(text):
            start, end = match.span("secret")

            if start != end:
                candidates.append(
                    (start, end, pattern.kind)
                )

    return _merge_matches(candidates)


def _merge_matches(
    candidates: list[tuple[int, int, SecretKind]],
) -> tuple[SecretMatch, ...]:
    if not candidates:
        return ()

    candidates.sort(key=lambda item: (item[0], item[1]))

    merged: list[SecretMatch] = []

    for start, end, kind in candidates:
        if not merged or start > merged[-1].end:
            merged.append(
                SecretMatch(
                    start=start,
                    end=end,
                    kinds=(kind,),
                )
            )
            continue

        previous = merged[-1]
        kinds = previous.kinds

        if kind not in kinds:
            kinds = (*kinds, kind)

        merged[-1] = SecretMatch(
            start=previous.start,
            end=max(previous.end, end),
            kinds=kinds,
        )

    return tuple(merged)


def _redact(
    text: str,
    matches: tuple[SecretMatch, ...],
    *,
    replacement: str,
) -> str:
    if not matches:
        return text

    parts: list[str] = []
    position = 0

    for match in matches:
        parts.append(text[position : match.start])
        secret_text = text[match.start : match.end]
        parts.append(replacement + ("\n" * secret_text.count("\n")))
        position = match.end

    parts.append(text[position:])
    return "".join(parts)
