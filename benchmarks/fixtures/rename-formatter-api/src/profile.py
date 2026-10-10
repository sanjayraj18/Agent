from src.formatting import format_user_name


def render_profile(first_name: str, last_name: str) -> str:
    return f"Profile: {format_user_name(first_name, last_name)}"
