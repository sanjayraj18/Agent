def format_user_name(first_name: str, last_name: str) -> str:
    """Format a user-facing full name."""
    return f"{first_name.strip()} {last_name.strip()}"
