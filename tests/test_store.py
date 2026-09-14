import json
import os
import stat

import pytest
from pydantic import SecretStr

from agent.auth.credentials import ApiKey, OAuthToken
from agent.auth.store import FileStore, StoreError

KEY = "sk-ant-api03-" + "x" * 40


@pytest.fixture
def store(tmp_path):
    return FileStore(tmp_path / "cfg" / "credentials.json")


def test_missing_file_loads_as_none(store):
    assert store.load() is None


def test_roundtrip(store):
    store.save(ApiKey(value=SecretStr(KEY)))
    loaded = store.load()
    assert isinstance(loaded, ApiKey)
    assert loaded.value.get_secret_value() == KEY


def test_oauth_roundtrip_keeps_refresh_token(store):
    store.save(OAuthToken(access_token=SecretStr("a"), refresh_token=SecretStr("r")))
    loaded = store.load()
    assert isinstance(loaded, OAuthToken)
    assert loaded.refresh_token.get_secret_value() == "r"


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_file_is_private_from_creation(store):
    store.save(ApiKey(value=SecretStr(KEY)))
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_loose_permissions_are_refused(store):
    store.save(ApiKey(value=SecretStr(KEY)))
    store.path.chmod(0o644)
    with pytest.raises(StoreError, match="readable by other users"):
        store.load()


def test_corrupt_file_names_the_fix(store):
    store.path.parent.mkdir(parents=True)
    store.path.write_text("{not json")
    store.path.chmod(0o600)
    with pytest.raises(StoreError, match="agent auth logout"):
        store.load()


def test_unknown_version_is_rejected(store):
    store.save(ApiKey(value=SecretStr(KEY)))
    data = json.loads(store.path.read_text())
    data["version"] = 99
    store.path.write_text(json.dumps(data))
    with pytest.raises(StoreError, match="unsupported format version"):
        store.load()


def test_overwrite_leaves_no_temp_files(store):
    for _ in range(3):
        store.save(ApiKey(value=SecretStr(KEY)))
    leftovers = [p.name for p in store.path.parent.iterdir() if p.name != "credentials.json"]
    assert leftovers == []


def test_delete_reports_whether_anything_was_there(store):
    store.save(ApiKey(value=SecretStr(KEY)))
    assert store.delete() is True
    assert store.delete() is False


def test_resolver_falls_back_to_the_store(store):
    from agent.auth.resolver import resolve

    store.save(ApiKey(value=SecretStr(KEY)))
    resolved = resolve(env={}, store=store.load)
    assert resolved.source == "store"
    assert KEY not in resolved.describe()