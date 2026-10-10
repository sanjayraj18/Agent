def build_headers(token: str) -> dict[str, str]:
    """Return HTTP headers for the service token."""
    return {"X-Api-Key": token}
