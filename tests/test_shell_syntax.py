import pytest

from agent.tools.shell_syntax import analyze_shell


def test_parses_one_simple_command_for_an_allowlist():
    analysis = analyze_shell("uv run pytest -q")

    assert analysis.is_safe_for_allowlist is True
    assert analysis.command is not None
    assert analysis.command.environment == ()
    assert analysis.command.argv == ("uv", "run", "pytest", "-q")


def test_preserves_quoted_arguments_without_treating_operators_as_syntax():
    analysis = analyze_shell('printf "%s" "a && b"')

    assert analysis.is_safe_for_allowlist is True
    assert analysis.command is not None
    assert analysis.command.argv == ("printf", "%s", "a && b")


def test_parses_a_leading_environment_prefix():
    analysis = analyze_shell("CI=1 uv run pytest")

    assert analysis.is_safe_for_allowlist is True
    assert analysis.command is not None
    assert analysis.command.environment == (("CI", "1"),)
    assert analysis.command.argv == ("uv", "run", "pytest")


@pytest.mark.parametrize(
    ("command", "finding"),
    [
        ("uv run pytest; rm -rf .", "command_separator"),
        ("uv run pytest && git status", "and_operator"),
        ("uv run pytest || git status", "or_operator"),
        ("uv run pytest | cat", "pipe"),
        ("uv run pytest > result.txt", "redirect"),
        ("uv run pytest &", "background"),
        ("echo $(whoami)", "command_substitution"),
        ("echo `whoami`", "backtick_substitution"),
        ("echo $HOME", "variable_expansion"),
        ("echo *.py", "glob_expansion"),
        ("echo ~", "tilde_expansion"),
        ("(echo hello)", "grouping"),
        ("echo hello # comment", "comment"),
        ("echo 'unfinished", "unclosed_quote"),
    ],
)
def test_marks_risky_shell_syntax_for_human_review(
    command: str,
    finding: str,
):
    analysis = analyze_shell(command)

    assert analysis.requires_human_review is True
    assert finding in analysis.findings


def test_rejects_an_empty_command_and_environment_only_command():
    assert analyze_shell("   ").findings == ("empty_command",)

    environment_only = analyze_shell("CI=1")
    assert environment_only.command is None
    assert "missing_command" in environment_only.findings


def test_detects_a_dangling_escape():
    analysis = analyze_shell("echo hello\\")

    assert "dangling_escape" in analysis.findings
    assert analysis.requires_human_review is True
