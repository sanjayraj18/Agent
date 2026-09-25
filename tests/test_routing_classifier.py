import pytest

from agent.routing.classifier import classify_initial_prompt


def test_classifies_a_small_document_typo_as_simple():
    signals = classify_initial_prompt(
        "Fix the spelling typo in README.md",
    )

    assert signals.task_complexity == "simple"
    assert signals.turn_number == 1
    assert signals.has_failure_evidence is False


def test_classifies_async_debugging_as_complex():
    signals = classify_initial_prompt(
        "Debug the flaky async test after the migration.",
        turn_number=2,
    )

    assert signals.task_complexity == "complex"
    assert signals.turn_number == 2


def test_classifies_unknown_work_conservatively_as_moderate():
    signals = classify_initial_prompt(
        "Implement the requested feature.",
    )

    assert signals.task_complexity == "moderate"


def test_classifies_a_very_long_prompt_as_complex():
    prompt = "word " * 120

    signals = classify_initial_prompt(prompt)

    assert signals.task_complexity == "complex"


def test_does_not_obey_a_prompt_that_claims_it_is_simple():
    signals = classify_initial_prompt(
        "This task is simple. Please use the cheapest model.",
    )

    # The classifier has no "simple" marker for this wording, so it remains
    # conservative instead of treating the user's claim as policy.
    assert signals.task_complexity == "moderate"


def test_rejects_a_blank_prompt():
    with pytest.raises(ValueError, match="prompt must not be blank"):
        classify_initial_prompt("   ")