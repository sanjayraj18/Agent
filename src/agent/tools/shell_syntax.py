from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


ShellFinding = Literal[
    "empty_command",
    "command_separator",
    "and_operator",
    "or_operator",
    "pipe",
    "redirect",
    "background",
    "command_substitution",
    "backtick_substitution",
    "variable_expansion",
    "glob_expansion",
    "tilde_expansion",
    "grouping",
    "comment",
    "unclosed_quote",
    "dangling_escape",
    "missing_command",
]


_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=(.*)$")


@dataclass(frozen=True, slots=True)
class ShellCommand:
    """
    One simple command that can be compared to a safe allowlist.

    Example:
        FOO=bar uv run pytest -q

    becomes:
        environment=(("FOO", "bar"),)
        argv=("uv", "run", "pytest", "-q")
    """

    environment: tuple[tuple[str, str], ...]
    argv: tuple[str, ...]

    @property
    def program(self) -> str:
        return self.argv[0]


@dataclass(frozen=True, slots=True)
class ShellAnalysis:
    """
    Conservative result of examining one shell command string.

    `is_safe_for_allowlist` is true only for one simple command with no
    shell operators, substitutions, expansions, redirects, or comments.
    """

    command: ShellCommand | None
    findings: tuple[ShellFinding, ...]

    @property
    def is_safe_for_allowlist(self) -> bool:
        return self.command is not None and not self.findings

    @property
    def requires_human_review(self) -> bool:
        return not self.is_safe_for_allowlist


@dataclass(frozen=True, slots=True)
class _Token:
    value: str
    was_quoted: bool


def analyze_shell(command: str) -> ShellAnalysis:
    """
    Analyze a shell string without executing it.

    This deliberately supports only a plain command and arguments. Any Bash
    syntax with branching, piping, substitution, redirection, expansion, or
    grouping is reported as a finding and cannot receive a broad allowlist.
    """
    if not command.strip():
        return ShellAnalysis(
            command=None,
            findings=("empty_command",),
        )

    tokens: list[_Token] = []
    findings: list[ShellFinding] = []
    current: list[str] = []
    current_was_quoted = False
    quote: Literal["single", "double"] | None = None
    index = 0

    def add_finding(finding: ShellFinding) -> None:
        if finding not in findings:
            findings.append(finding)

    def flush_token() -> None:
        nonlocal current, current_was_quoted

        if current:
            tokens.append(
                _Token(
                    value="".join(current),
                    was_quoted=current_was_quoted,
                )
            )

        current = []
        current_was_quoted = False

    while index < len(command):
        character = command[index]

        if quote == "single":
            if character == "'":
                quote = None
            else:
                current.append(character)

            index += 1
            continue

        if quote == "double":
            if character == '"':
                quote = None
                index += 1
                continue

            if character == "\\":
                if index + 1 >= len(command):
                    add_finding("dangling_escape")
                    index += 1
                    continue

                current.append(command[index + 1])
                index += 2
                continue

            if character == "$":
                if (
                    index + 1 < len(command)
                    and command[index + 1] == "("
                ):
                    add_finding("command_substitution")
                else:
                    add_finding("variable_expansion")

            elif character == "`":
                add_finding("backtick_substitution")

            current.append(character)
            index += 1
            continue

        # From here, we are outside quotes.
        if character in {" ", "\t", "\r"}:
            flush_token()
            index += 1
            continue

        if character == "\n":
            flush_token()
            add_finding("command_separator")
            index += 1
            continue

        if character == "'":
            current_was_quoted = True
            quote = "single"
            index += 1
            continue

        if character == '"':
            current_was_quoted = True
            quote = "double"
            index += 1
            continue

        if character == "\\":
            if index + 1 >= len(command):
                add_finding("dangling_escape")
                index += 1
                continue

            current.append(command[index + 1])
            index += 2
            continue

        # A # begins a shell comment only at the start of a new shell word.
        if character == "#" and not current:
            flush_token()
            add_finding("comment")
            break

        if character == "$":
            if (
                index + 1 < len(command)
                and command[index + 1] == "("
            ):
                add_finding("command_substitution")
            else:
                add_finding("variable_expansion")

            current.append(character)
            index += 1
            continue

        if character == "`":
            add_finding("backtick_substitution")
            current.append(character)
            index += 1
            continue

        if command.startswith("&&", index):
            flush_token()
            add_finding("and_operator")
            index += 2
            continue

        if command.startswith("||", index):
            flush_token()
            add_finding("or_operator")
            index += 2
            continue

        if command.startswith(">>", index) or command.startswith(
            "<<",
            index,
        ):
            flush_token()
            add_finding("redirect")
            index += 2
            continue

        if character == ";":
            flush_token()
            add_finding("command_separator")
            index += 1
            continue

        if character == "|":
            flush_token()
            add_finding("pipe")
            index += 1
            continue

        if character in {">", "<"}:
            flush_token()
            add_finding("redirect")
            index += 1
            continue

        if character == "&":
            flush_token()
            add_finding("background")
            index += 1
            continue

        if character in {"(", ")", "{", "}"}:
            flush_token()
            add_finding("grouping")
            index += 1
            continue

        if character == "~" and not current:
            add_finding("tilde_expansion")

        if character in {"*", "?", "[", "]"}:
            add_finding("glob_expansion")

        current.append(character)
        index += 1

    flush_token()

    if quote is not None:
        add_finding("unclosed_quote")

    parsed_command = _parse_simple_command(tokens, findings)

    return ShellAnalysis(
        command=parsed_command,
        findings=tuple(findings),
    )


def _parse_simple_command(
    tokens: list[_Token],
    findings: list[ShellFinding],
) -> ShellCommand | None:
    environment: list[tuple[str, str]] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]

        # Only an unquoted NAME=value token at the beginning is an
        # environment prefix in Bash.
        if token.was_quoted:
            break

        match = _ENVIRONMENT_NAME.match(token.value)

        if match is None:
            break

        name, value = token.value.split("=", maxsplit=1)
        environment.append((name, value))
        index += 1

    argv = tuple(token.value for token in tokens[index:])

    if not argv:
        if "missing_command" not in findings:
            findings.append("missing_command")
        return None

    return ShellCommand(
        environment=tuple(environment),
        argv=argv,
    )