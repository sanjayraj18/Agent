import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from agent import logs
from agent.auth.credentials import (
    OAUTH_BETA, ApiKey, OAuthToken, from_storage,
)

REAL_KEY = "sk-ant-api03-" + "x" * 40
REAL_TOKEN = "sk-ant-oat01-" + "y" * 40


@pytest.fixture(autouse=True)
def reset_logging():
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)


# ------------------------------------------------------------ the wall holds


@pytest.mark.parametrize(
    "cred,secret",
    [
        (ApiKey(value=REAL_KEY), REAL_KEY),
        (OAuthToken(access_token=REAL_TOKEN), REAL_TOKEN),
    ],
    ids=["api_key", "oauth"],
)
def test_secret_escapes_through_no_ordinary_path(cred, secret):
    assert secret not in repr(cred)
    assert secret not in str(cred)
    assert secret not in f"{cred}"
    assert secret not in cred.model_dump_json()
    assert secret not in str(cred.model_dump())
    assert secret not in cred.describe()


def test_secret_does_not_escape_through_a_traceback(capsys):
    logs.setup(level="info")
    cred = ApiKey(value=REAL_KEY)
    try:
        raise RuntimeError(f"auth failed for {cred!r}")
    except RuntimeError:
        logs.get("auth").exception("boom")
    assert REAL_KEY not in capsys.readouterr().err


def test_headers_leak_is_covered_by_the_scrubber(capsys):
    """headers() necessarily contains the real secret — the net catches it."""
    logs.setup(level="info")
    cred = ApiKey(value=REAL_KEY)
    assert cred.headers()["x-api-key"] == REAL_KEY      # the door exists
    logs.get("provider").info("sending %s", cred.headers())
    err = capsys.readouterr().err
    assert REAL_KEY not in err                           # the net holds
    assert "sk-ant-***" in err


# ---------------------------------------------------------------- behaviour


def test_api_key_header_shape():
    assert ApiKey(value=REAL_KEY).headers() == {"x-api-key": REAL_KEY}


def test_oauth_header_shape_includes_the_beta():
    h = OAuthToken(access_token=REAL_TOKEN).headers()
    assert h["Authorization"] == f"Bearer {REAL_TOKEN}"
    assert h[OAUTH_BETA.split("-")[0] and "anthropic-beta"] == OAUTH_BETA
    assert "x-api-key" not in h


def test_api_keys_never_expire():
    assert ApiKey(value=REAL_KEY).is_expired() is False


def test_token_without_expiry_never_expires():
    assert OAuthToken(access_token=REAL_TOKEN).is_expired() is False


def test_skew_expires_a_token_early():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    tok = OAuthToken(access_token=REAL_TOKEN, expires_at=now + timedelta(seconds=30))
    assert tok.is_expired(now=now, skew=60.0) is True     # inside the window
    assert tok.is_expired(now=now, skew=10.0) is False    # outside it


def test_naive_expiry_is_coerced_to_utc():
    """A naive datetime from storage would raise TypeError on comparison."""
    tok = OAuthToken(access_token=REAL_TOKEN, expires_at=datetime(2030, 1, 1, 0, 0, 0))
    assert tok.expires_at.tzinfo is timezone.utc
    tok.is_expired()  # must not raise


def test_can_refresh_reflects_the_refresh_token():
    assert OAuthToken(access_token=REAL_TOKEN).can_refresh is False
    assert OAuthToken(access_token=REAL_TOKEN, refresh_token="r").can_refresh is True


# ------------------------------------------------------------------ storage


def test_storage_roundtrip_preserves_the_secret():
    original = ApiKey(value=REAL_KEY)
    restored = from_storage(original.to_storage())
    assert isinstance(restored, ApiKey)
    assert restored.value.get_secret_value() == REAL_KEY


def test_oauth_storage_roundtrip_preserves_every_field():
    expires = datetime(2030, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    original = OAuthToken(
        access_token=REAL_TOKEN,
        refresh_token="refresh-me",
        expires_at=expires,
        scopes=("user:inference",),
    )
    restored = from_storage(json.loads(json.dumps(original.to_storage())))
    assert isinstance(restored, OAuthToken)
    assert restored.access_token.get_secret_value() == REAL_TOKEN
    assert restored.refresh_token.get_secret_value() == "refresh-me"
    assert restored.expires_at == expires
    assert restored.scopes == ("user:inference",)


def test_storage_is_the_only_form_that_contains_the_secret():
    cred = ApiKey(value=REAL_KEY)
    assert REAL_KEY in json.dumps(cred.to_storage())      # on purpose
    assert REAL_KEY not in cred.model_dump_json()         # everywhere else


# -------------------------------------------------------------- fingerprint


def test_fingerprint_is_stable_and_reveals_nothing():
    a, b = ApiKey(value=REAL_KEY), ApiKey(value=REAL_KEY)
    assert a.fingerprint == b.fingerprint
    assert len(a.fingerprint) == 8
    assert a.fingerprint not in REAL_KEY


def test_different_credentials_fingerprint_differently():
    assert ApiKey(value=REAL_KEY).fingerprint != ApiKey(value="sk-ant-other").fingerprint