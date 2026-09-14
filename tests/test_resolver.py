import pytest

from agent.auth.credentials import ApiKey, OAuthToken
from agent.auth.resolver import CredentialError, resolve

KEY = "sk-ant-api03-" + "x" * 40
TOKEN = "sk-ant-oat01-" + "y" * 40


def test_flag_wins_over_env():
    r = resolve(api_key=KEY, env={"ANTHROPIC_API_KEY": "other"})
    assert r.source == "flag"
    assert r.credential.value.get_secret_value() == KEY


def test_env_api_key():
    r = resolve(env={"ANTHROPIC_API_KEY": KEY})
    assert isinstance(r.credential, ApiKey)
    assert r.origin == "ANTHROPIC_API_KEY"
    assert r.credential.headers() == {"x-api-key": KEY}


def test_env_auth_token_is_oauth():
    r = resolve(env={"ANTHROPIC_AUTH_TOKEN": TOKEN})
    assert isinstance(r.credential, OAuthToken)
    h = r.credential.headers()
    assert h["Authorization"] == f"Bearer {TOKEN}"
    assert h["anthropic-beta"] == "oauth-2025-04-20"


def test_both_env_vars_is_an_error():
    with pytest.raises(CredentialError, match="both"):
        resolve(env={"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_AUTH_TOKEN": TOKEN})


def test_empty_env_var_is_an_error_not_a_fallthrough():
    """The stale-.zshrc trap: empty still wins its slot."""
    with pytest.raises(CredentialError, match="set but empty"):
        resolve(env={"ANTHROPIC_API_KEY": ""}, store=lambda: ApiKey(value=KEY))


def test_whitespace_only_counts_as_empty():
    with pytest.raises(CredentialError, match="set but empty"):
        resolve(env={"ANTHROPIC_API_KEY": "   "})


def test_env_values_are_stripped():
    r = resolve(env={"ANTHROPIC_API_KEY": f"  {KEY}\n"})
    assert r.credential.value.get_secret_value() == KEY


def test_store_is_used_when_env_is_absent():
    r = resolve(env={}, store=lambda: ApiKey(value=KEY))
    assert r.source == "store"


def test_env_beats_store():
    r = resolve(env={"ANTHROPIC_API_KEY": KEY}, store=lambda: ApiKey(value="stored"))
    assert r.source == "env"


def test_no_credential_gives_an_actionable_message():
    with pytest.raises(CredentialError, match="agent auth login|ANTHROPIC_API_KEY"):
        resolve(env={}, store=lambda: None)


def test_describe_never_reveals_the_secret():
    d = resolve(env={"ANTHROPIC_API_KEY": KEY}).describe()
    assert KEY not in d
    assert "ANTHROPIC_API_KEY" in d