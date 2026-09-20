import json

import pytest

from agent.core.grants import (
    CommandGrant,
    GrantStoreError,
    ProjectGrantStore,
)
from agent.core.permission import ToolAction


def bash_action(
    command: str,
    *,
    background: bool = False,
) -> ToolAction:
    return ToolAction(
        tool_name="bash",
        arguments={
            "action": "run",
            "command": command,
            "background": background,
        },
    )


def test_exact_command_grant_matches_only_its_exact_argv():
    grant = CommandGrant(("uv", "run", "pytest"))

    assert grant.allows(bash_action("uv run pytest")) is True
    assert grant.allows(bash_action("uv run pytest -q")) is False
    assert grant.allows(bash_action("uv run ruff")) is False


def test_prefix_grant_can_allow_extra_arguments():
    grant = CommandGrant(
        ("uv", "run", "pytest"),
        allow_extra_args=True,
    )

    assert grant.allows(bash_action("uv run pytest -q")) is True


@pytest.mark.parametrize(
    "command",
    [
        "uv run pytest; rm -rf .",
        "uv run pytest && git status",
        "uv run pytest | cat",
        "uv run pytest > result.txt",
        "uv run pytest $(whoami)",
        "CI=1 uv run pytest",
    ],
)
def test_grant_refuses_shell_syntax_that_cannot_be_safely_allowlisted(
    command: str,
):
    grant = CommandGrant(
        ("uv", "run", "pytest"),
        allow_extra_args=True,
    )

    assert grant.allows(bash_action(command)) is False


def test_grant_refuses_background_commands_and_other_tools():
    grant = CommandGrant(("uv", "run", "pytest"))

    assert grant.allows(
        bash_action("uv run pytest", background=True)
    ) is False
    assert grant.allows(ToolAction(tool_name="write_file")) is False


def test_store_round_trips_grants_as_project_local_json(tmp_path):
    store = ProjectGrantStore(tmp_path)
    grant = CommandGrant(
        ("uv", "run", "pytest"),
        allow_extra_args=True,
    )

    store.add(grant)
    loaded = ProjectGrantStore.load(tmp_path)

    assert loaded.grants == (grant,)
    assert loaded.allows(bash_action("uv run pytest -q")) is True

    payload = json.loads(loaded.path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["bash_command_grants"] == [grant.to_json()]


def test_adding_the_same_grant_twice_does_not_duplicate_it(tmp_path):
    store = ProjectGrantStore(tmp_path)
    grant = CommandGrant(("git", "status"))

    store.add(grant)
    store.add(grant)

    assert ProjectGrantStore.load(tmp_path).grants == (grant,)


def test_missing_grants_file_means_no_approval(tmp_path):
    store = ProjectGrantStore.load(tmp_path)

    assert store.grants == ()
    assert store.allows(bash_action("uv run pytest")) is False


def test_rejects_malformed_or_duplicate_persisted_grants(tmp_path):
    grants_path = tmp_path / ".agent" / "permissions.json"
    grants_path.parent.mkdir()
    grants_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(GrantStoreError, match="invalid JSON"):
        ProjectGrantStore.load(tmp_path)

    grants_path.write_text(
        json.dumps(
            {
                "version": 1,
                "bash_command_grants": [
                    {
                        "argv_prefix": ["git", "status"],
                        "allow_extra_args": False,
                    },
                    {
                        "argv_prefix": ["git", "status"],
                        "allow_extra_args": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GrantStoreError, match="must not contain duplicates"):
        ProjectGrantStore.load(tmp_path)


def test_rejects_invalid_grant_schema():
    with pytest.raises(GrantStoreError, match="argv_prefix"):
        CommandGrant.from_json(
            {
                "argv_prefix": "uv run pytest",
                "allow_extra_args": False,
            }
        )
