from src.request_headers import build_headers


def authorization_value(token: str) -> str:
    return build_headers(token)["X-Api-Key"]
