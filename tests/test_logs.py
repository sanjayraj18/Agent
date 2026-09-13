import json
import logging

import pytest

from agent import logs


@pytest.fixture(autouse=True)
def reset_logging():
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)


ANTHROPIC_KEY = "sk-ant-api03-" + "x" * 40
OAUTH_TOKEN = "sk-ant-oat01-" + "y" * 40
GITHUB_TOKEN = "ghp_" + "z" * 36


def test_redacts_anthropic_key():
    out = logs.redact(f"using key {ANTHROPIC_KEY} for request")
    assert ANTHROPIC_KEY not in out
    assert "sk-ant-***" in out


def test_redacts_inside_a_url():
    """Shape-based patterns must fire without surrounding structure."""
    out = logs.redact(f"GET https://api.example.com/v1?token={GITHUB_TOKEN}&x=1")
    assert GITHUB_TOKEN not in out
    assert "ghp_***" in out


def test_redacts_authorization_header():
    out = logs.redact(f"Authorization: Bearer {OAUTH_TOKEN}")
    assert OAUTH_TOKEN not in out


def test_redacts_shapeless_value_by_key_name():
    out = logs.redact('{"password": "hunter2", "user": "alice"}')
    assert "hunter2" not in out
    assert "alice" in out  # non-secrets survive


def test_secret_cannot_escape_through_a_handler(capsys):
    logs.setup(level="info")
    logs.get("provider").info("auth failed with %s", ANTHROPIC_KEY)
    err = capsys.readouterr().err
    assert ANTHROPIC_KEY not in err
    assert "sk-ant-***" in err


def test_secret_cannot_escape_through_a_traceback(capsys):
    logs.setup(level="info")
    log = logs.get("provider")
    try:
        raise ValueError(f"bad key: {ANTHROPIC_KEY}")
    except ValueError:
        log.exception("request failed")
    err = capsys.readouterr().err
    assert ANTHROPIC_KEY not in err


def test_third_party_loggers_are_also_redacted(capsys):
    """httpx and the anthropic SDK propagate to root — they must be covered too."""
    logs.setup(level="info")
    logging.getLogger("httpx").info("request header x-api-key: %s", ANTHROPIC_KEY)
    err = capsys.readouterr().err
    assert ANTHROPIC_KEY not in err


def test_json_file_output_is_structured_and_redacted(tmp_path):
    path = tmp_path / "agent.jsonl"
    logs.setup(level="info", log_file=path)
    logs.get("provider").info(
        "request complete",
        extra={"model": "claude-opus-5", "cache_read": 9120, "key": ANTHROPIC_KEY},
    )
    logging.shutdown()

    record = json.loads(path.read_text().strip())
    assert record["msg"] == "request complete"
    assert record["model"] == "claude-opus-5"
    assert record["cache_read"] == 9120
    assert ANTHROPIC_KEY not in path.read_text()