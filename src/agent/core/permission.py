from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping, Protocol


PermissionMode = Literal["readonly", "ask", "auto", "full"]

PermissionVerdict = Literal["allow", "ask", "deny"]

ActionRisk = Literal[
    "read",
    "workspace_write",
    "shell_execute",
    "process_control",
    "unknown",
]


_VALID_MODES: set[str] = {
    "readonly",
    "ask",
    "auto",
    "full",
}


_READ_ONLY_TOOLS = {
    "read_file",
    "glob",
    "grep",
}


_WORKSPACE_WRITE_TOOLS = {
    "write_file",
    "edit_file",
}


class PermissionError(ValueError):
    """Raised for malformed permission-policy input."""


@dataclass(frozen=True, slots=True)
class ToolAction:
    """
    One tool call proposed by the model.

    `contains_untrusted_content` becomes important when a model has just read
    text from a file, command output, or a remote source. We should not let
    such text silently cause a risky action.
    """

    tool_name: str
    arguments: Mapping[str, object] = field(default_factory=dict)
    contains_untrusted_content: bool = False

    def __post_init__(self) -> None:
        if not self.tool_name.strip():
            raise PermissionError("tool_name must not be empty")

        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(dict(self.arguments)),
        )


@dataclass(frozen=True, slots=True)
class PermissionResult:
    """The policy's answer before a tool is allowed to execute."""

    verdict: PermissionVerdict
    reason: str
    risk: ActionRisk

    def __post_init__(self) -> None:
        if self.verdict not in {"allow", "ask", "deny"}:
            raise PermissionError(
                f"invalid permission verdict: {self.verdict!r}"
            )

        if not self.reason.strip():
            raise PermissionError("permission reason must not be empty")


class GrantChecker(Protocol):
    """Implemented later by the per-project persisted grant store."""

    def allows(self, action: ToolAction) -> bool:
        ...


@dataclass(frozen=True, slots=True)
class PermissionPolicy:
    """
    Decide whether a proposed tool call may run.

    Policy is intentionally separate from execution:
    - this class decides allow / ask / deny;
    - ToolDispatcher enforces the decision;
    - BashTool only executes a command that passed the dispatcher gate.
    """

    default_mode: PermissionMode = "ask"
    tool_modes: Mapping[str, PermissionMode] = field(
        default_factory=dict
    )
    grants: GrantChecker | None = None

    def __post_init__(self) -> None:
        _validate_mode(self.default_mode, setting="default_mode")

        normalized_tool_modes: dict[str, PermissionMode] = {}

        for tool_name, mode in self.tool_modes.items():
            if not tool_name.strip():
                raise PermissionError(
                    "tool_modes cannot contain an empty tool name"
                )

            _validate_mode(mode, setting=f"tool_modes[{tool_name!r}]")
            normalized_tool_modes[tool_name] = mode

        object.__setattr__(
            self,
            "tool_modes",
            MappingProxyType(normalized_tool_modes),
        )

    def mode_for(self, action: ToolAction) -> PermissionMode:
        """Return a per-tool override, or the project's default mode."""
        return self.tool_modes.get(
            action.tool_name,
            self.default_mode,
        )

    def decide(self, action: ToolAction) -> PermissionResult:
        """
        Return a decision without performing any side effect.

        A persistent grant can approve a specific action in `ask` and `auto`
        modes. It never overrides `readonly`, because that mode is an
        explicit temporary safety boundary.
        """
        mode = self.mode_for(action)
        risk = classify_action_risk(action)

        if risk == "unknown":
            return PermissionResult(
                verdict="deny",
                risk=risk,
                reason=(
                    "the tool is unknown to the permission policy"
                ),
            )

        if mode == "readonly":
            return self._readonly_decision(risk)

        # Untrusted text may be hostile prompt injection. Reading it is safe;
        # acting on it must require a human decision, even in full mode.
        if action.contains_untrusted_content and risk != "read":
            return PermissionResult(
                verdict="ask",
                risk=risk,
                reason=(
                    "risky action may have been influenced by "
                    "untrusted content"
                ),
            )

        if mode == "full":
            return PermissionResult(
                verdict="allow",
                risk=risk,
                reason="full mode allows known tools",
            )

        if risk == "read":
            return PermissionResult(
                verdict="allow",
                risk=risk,
                reason="read-only action is safe to run automatically",
            )

        if self.grants is not None and self.grants.allows(action):
            return PermissionResult(
                verdict="allow",
                risk=risk,
                reason="action matches a persisted project grant",
            )

        if mode == "auto" and risk == "workspace_write":
            return PermissionResult(
                verdict="allow",
                risk=risk,
                reason=(
                    "auto mode allows workspace-scoped file changes"
                ),
            )

        if risk == "shell_execute":
            reason = (
                "shell execution requires an approved command allowlist"
            )
        else:
            reason = "risky action requires user approval"

        return PermissionResult(
            verdict="ask",
            risk=risk,
            reason=reason,
        )

    @staticmethod
    def _readonly_decision(
        risk: ActionRisk,
    ) -> PermissionResult:
        if risk == "read":
            return PermissionResult(
                verdict="allow",
                risk=risk,
                reason="readonly mode allows inspection tools",
            )

        return PermissionResult(
            verdict="deny",
            risk=risk,
            reason="readonly mode blocks actions that change state",
        )


def classify_action_risk(action: ToolAction) -> ActionRisk:
    """
    Classify a tool call before deeper Bash parsing is added.

    `shell_syntax.py` will later inspect a Bash `run` command in more detail.
    For now, every Bash run is correctly treated as potentially risky.
    """
    if action.tool_name in _READ_ONLY_TOOLS:
        return "read"

    if action.tool_name in _WORKSPACE_WRITE_TOOLS:
        return "workspace_write"

    if action.tool_name != "bash":
        return "unknown"

    bash_action = action.arguments.get("action", "run")

    if bash_action == "status":
        return "read"

    if bash_action == "stop":
        return "process_control"

    if bash_action == "run":
        return "shell_execute"

    return "unknown"


def _validate_mode(
    mode: str,
    *,
    setting: str,
) -> None:
    if mode not in _VALID_MODES:
        expected = ", ".join(sorted(_VALID_MODES))
        raise PermissionError(
            f"{setting} must be one of: {expected}; got {mode!r}"
        )