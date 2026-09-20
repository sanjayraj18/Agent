import pytest

from agent.core.permission import (
    PermissionError,
    PermissionPolicy,
    ToolAction,
    classify_action_risk,
)


class AlwaysGrant:
    def allows(self, action: ToolAction) -> bool:
        return action.tool_name == "bash"


def action(
    tool_name: str,
    arguments: dict[str, object] | None = None,
    *,
    untrusted: bool = False,
) -> ToolAction:
    return ToolAction(
        tool_name=tool_name,
        arguments=arguments or {},
        contains_untrusted_content=untrusted,
    )


@pytest.mark.parametrize(
    ("proposed", "expected_risk"),
    [
        (action("read_file"), "read"),
        (action("glob"), "read"),
        (action("grep"), "read"),
        (action("write_file"), "workspace_write"),
        (action("edit_file"), "workspace_write"),
        (action("bash", {"action": "status"}), "read"),
        (action("bash", {"action": "stop"}), "process_control"),
        (action("bash", {"action": "run"}), "shell_execute"),
        (action("unknown_tool"), "unknown"),
    ],
)
def test_classifies_tool_risk(
    proposed: ToolAction,
    expected_risk: str,
):
    assert classify_action_risk(proposed) == expected_risk


def test_readonly_allows_reads_and_denies_state_changes():
    policy = PermissionPolicy(default_mode="readonly")

    assert policy.decide(action("read_file")).verdict == "allow"
    assert policy.decide(action("write_file")).verdict == "deny"
    assert policy.decide(action("bash", {"action": "run"})).verdict == "deny"


def test_ask_mode_allows_reads_but_requires_approval_for_writes():
    policy = PermissionPolicy(default_mode="ask")

    assert policy.decide(action("grep")).verdict == "allow"
    assert policy.decide(action("edit_file")).verdict == "ask"
    assert policy.decide(action("bash", {"action": "run"})).verdict == "ask"


def test_auto_mode_allows_workspace_writes_but_not_ungranted_shell():
    policy = PermissionPolicy(default_mode="auto")

    assert policy.decide(action("write_file")).verdict == "allow"
    assert policy.decide(action("bash", {"action": "run"})).verdict == "ask"


def test_full_mode_allows_known_tools_but_denies_unknown_tools():
    policy = PermissionPolicy(default_mode="full")

    assert policy.decide(action("edit_file")).verdict == "allow"
    assert policy.decide(action("bash", {"action": "stop"})).verdict == "allow"
    assert policy.decide(action("made_up_tool")).verdict == "deny"


def test_persisted_grant_allows_shell_execution_in_ask_mode():
    policy = PermissionPolicy(
        default_mode="ask",
        grants=AlwaysGrant(),
    )

    result = policy.decide(
        action("bash", {"action": "run", "command": "uv run pytest"})
    )

    assert result.verdict == "allow"
    assert result.reason == "action matches a persisted project grant"


def test_per_tool_mode_overrides_the_default_mode():
    policy = PermissionPolicy(
        default_mode="full",
        tool_modes={"bash": "ask"},
    )

    assert policy.decide(action("edit_file")).verdict == "allow"
    assert policy.decide(action("bash", {"action": "run"})).verdict == "ask"


def test_untrusted_content_cannot_silently_trigger_a_risky_action():
    policy = PermissionPolicy(default_mode="full")

    result = policy.decide(action("write_file", untrusted=True))

    assert result.verdict == "ask"
    assert "untrusted content" in result.reason


def test_untrusted_content_does_not_block_read_only_inspection():
    policy = PermissionPolicy(default_mode="full")

    assert policy.decide(action("read_file", untrusted=True)).verdict == "allow"


def test_rejects_invalid_default_and_per_tool_modes():
    with pytest.raises(PermissionError, match="default_mode must be one of"):
        PermissionPolicy(default_mode="unsafe")  # type: ignore[arg-type]

    with pytest.raises(PermissionError, match=r"tool_modes\['bash'\]"):
        PermissionPolicy(tool_modes={"bash": "unsafe"})  # type: ignore[dict-item]


def test_tool_action_copies_its_arguments():
    arguments: dict[str, object] = {"path": "notes.txt"}
    proposed = action("read_file", arguments)
    arguments["path"] = "changed.txt"

    assert proposed.arguments["path"] == "notes.txt"
