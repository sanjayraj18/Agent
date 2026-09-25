from agent.routing.state import RoutingState


def test_creates_initial_state_from_a_prompt():
    state = RoutingState.from_initial_prompt(
        "Fix the spelling typo in README.md",
    )

    assert state.signals.task_complexity == "simple"
    assert state.signals.turn_number == 1
    assert state.signals.tool_error_count == 0
    assert state.signals.verification_failure_count == 0


def test_successful_tool_result_keeps_the_same_state():
    state = RoutingState.from_initial_prompt(
        "Fix the spelling typo in README.md",
    )

    updated = state.observe_tool_result(is_error=False)

    assert updated is state


def test_failed_tool_result_creates_updated_state():
    state = RoutingState.from_initial_prompt(
        "Fix the spelling typo in README.md",
    )

    updated = state.observe_tool_result(is_error=True)

    # The original snapshot remains unchanged.
    assert state.signals.tool_error_count == 0

    # The next snapshot carries the new evidence.
    assert updated.signals.tool_error_count == 1
    assert updated.signals.turn_number == 1


def test_verification_failure_creates_updated_state():
    state = RoutingState.from_initial_prompt(
        "Implement the requested feature.",
    )

    updated = state.observe_verification_failure()

    assert state.signals.verification_failure_count == 0
    assert updated.signals.verification_failure_count == 1


def test_next_turn_preserves_evidence_and_increments_turn_number():
    state = RoutingState.from_initial_prompt(
        "Fix the spelling typo in README.md",
    )
    failed_state = state.observe_tool_result(is_error=True)

    next_turn = failed_state.begin_next_turn()

    assert failed_state.signals.turn_number == 1
    assert next_turn.signals.turn_number == 2
    assert next_turn.signals.tool_error_count == 1
    assert next_turn.signals.verification_failure_count == 0