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
