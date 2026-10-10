from src.client import authorization_value
from src.request_headers import build_headers


def test_headers_use_the_standard_bearer_authorization_format() -> None:
    assert build_headers("secret") == {"Authorization": "Bearer secret"}


def test_client_reads_the_migrated_header_name() -> None:
    assert authorization_value("secret") == "Bearer secret"
