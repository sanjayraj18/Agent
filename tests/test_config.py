import json

import pytest

from agent import config
from agent.config import ConfigError


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point every config layer at tmp_path so the host machine can't leak in."""
    monkeypatch.setattr(config, "SYSTEM_CONFIG", tmp_path / "system.json")
    monkeypatch.setattr(config, "USER_CONFIG", tmp_path / "user.json")
    return tmp_path


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_defaults_when_nothing_is_set(isolated):
    r = config.load(isolated, env={})
    assert r["model"].value == "claude-opus-5"
    assert r["model"].layer == "default"
    assert r["permission_mode"].value == "ask"
    assert r["tool_permission_modes"].value == {}


def test_project_beats_user(isolated):
    write(isolated / "user.json", {"model": "user-model"})
    write(isolated / ".agent/config.json", {"model": "project-model"})
    r = config.load(isolated, env={})
    assert r["model"].value == "project-model"
    assert r["model"].layer == "project"


def test_env_beats_project(isolated):
    write(isolated / ".agent/config.json", {"model": "project-model"})
    r = config.load(isolated, env={"AGENT_MODEL": "env-model"})
    assert r["model"].layer == "env"
    assert r["model"].origin == "AGENT_MODEL"


def test_flag_beats_env(isolated):
    r = config.load(isolated, flags={"model": "flag-model"},
                    env={"AGENT_MODEL": "env-model"})
    assert r["model"].value == "flag-model"
    assert r["model"].layer == "flag"


def test_unset_flags_do_not_override(isolated):
    """argparse hands us None for every flag the user didn't pass."""
    r = config.load(isolated, flags={"model": None}, env={"AGENT_MODEL": "env-model"})
    assert r["model"].value == "env-model"


def test_env_ints_are_coerced(isolated):
    r = config.load(isolated, env={"AGENT_MAX_TOKENS": "64000"})
    assert r["max_tokens"].value == 64000


def test_bad_int_in_env_is_rejected(isolated):
    with pytest.raises(ConfigError, match="expected an integer"):
        config.load(isolated, env={"AGENT_MAX_TOKENS": "lots"})


def test_unknown_key_in_a_file_is_rejected(isolated):
    write(isolated / ".agent/config.json", {"modle": "typo"})
    with pytest.raises(ConfigError, match="unknown setting"):
        config.load(isolated, env={})


def test_malformed_json_is_rejected(isolated):
    p = isolated / ".agent/config.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json")
    with pytest.raises(ConfigError, match="invalid JSON"):
        config.load(isolated, env={})


def test_project_can_set_default_and_per_tool_permission_modes(isolated):
    write(
        isolated / ".agent/config.json",
        {
            "permission_mode": "auto",
            "tool_permission_modes": {
                "bash": "ask",
                "edit_file": "auto",
            },
        },
    )

    settings = config.values(config.load(isolated, env={}))

    assert settings["permission_mode"] == "auto"
    assert settings["tool_permission_modes"] == {
        "bash": "ask",
        "edit_file": "auto",
    }


def test_invalid_permission_modes_are_rejected_at_configuration_time(isolated):
    write(
        isolated / ".agent/config.json",
        {"permission_mode": "dangerously_fast"},
    )

    with pytest.raises(ConfigError, match="permission_mode must be one of"):
        config.load(isolated, env={})


def test_invalid_per_tool_permission_mode_is_rejected(isolated):
    write(
        isolated / ".agent/config.json",
        {"tool_permission_modes": {"bash": "dangerously_fast"}},
    )

    with pytest.raises(
        ConfigError,
        match=r"tool_permission_modes\['bash'\]",
    ):
        config.load(isolated, env={})


def test_permission_mode_and_tool_overrides_can_come_from_the_environment(isolated):
    resolved = config.load(
        isolated,
        env={
            "AGENT_PERMISSION_MODE": "readonly",
            "AGENT_TOOL_PERMISSION_MODES": '{"bash": "ask"}',
        },
    )

    assert resolved["permission_mode"].value == "readonly"
    assert resolved["tool_permission_modes"].value == {"bash": "ask"}


def test_non_object_tool_permission_environment_value_is_rejected(isolated):
    with pytest.raises(ConfigError, match="expected a JSON object"):
        config.load(
            isolated,
            env={"AGENT_TOOL_PERMISSION_MODES": "[]"},
        )

def test_routing_mode_defaults_to_off(isolated):
    resolved = config.load(isolated, env={})

    assert resolved["routing_mode"].value == "off"
    assert resolved["routing_mode"].layer == "default"


def test_routing_mode_can_be_enabled_for_shadow_observation(isolated):
    resolved = config.load(
        isolated,
        env={"AGENT_ROUTING_MODE": "shadow"},
    )

    assert resolved["routing_mode"].value == "shadow"
    assert resolved["routing_mode"].layer == "env"
    assert resolved["routing_mode"].origin == "AGENT_ROUTING_MODE"


def test_routing_mode_can_be_enabled_for_live_execution(isolated):
    resolved = config.load(
        isolated,
        env={"AGENT_ROUTING_MODE": "live"},
    )

    assert resolved["routing_mode"].value == "live"


def test_rejects_an_unknown_routing_mode(isolated):
    with pytest.raises(
        ConfigError,
        match="routing_mode must be one of: live, off, shadow",
    ):
        config.load(
            isolated,
            env={"AGENT_ROUTING_MODE": "adaptive"},
        )
