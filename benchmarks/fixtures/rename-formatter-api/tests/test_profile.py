from src.formatting import format_display_name
from src.profile import render_profile


def test_new_public_formatter_name_is_available() -> None:
    assert format_display_name(" Ada ", " Lovelace ") == "Ada Lovelace"


def test_profile_uses_the_renamed_formatter() -> None:
    assert render_profile("Ada", "Lovelace") == "Profile: Ada Lovelace"
