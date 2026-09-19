import pytest

from agent.core.prompt_cache import (
    PromptCachePlan,
    PromptCachePlanError,
    PromptSection,
    stable_prefix_fingerprint,
)


def _plan(dynamic_content: str) -> PromptCachePlan:
    return PromptCachePlan(
        sections=(
            PromptSection(
                kind="system",
                content="You are a careful coding agent.",
            ),
            PromptSection(
                kind="tools",
                content='[{"name":"read_file"},{"name":"grep"}]',
            ),
            PromptSection(
                kind="stable_context",
                content="Workspace paths must stay relative.",
            ),
            PromptSection(
                kind="dynamic",
                content=dynamic_content,
            ),
        )
    )


def test_accepts_the_correct_cache_section_order():
    plan = _plan("User: inspect the failing test")

    assert [section.kind for section in plan.stable_sections] == [
        "system",
        "tools",
        "stable_context",
    ]
    assert [section.kind for section in plan.dynamic_sections] == [
        "dynamic",
    ]


def test_dynamic_content_does_not_change_the_stable_fingerprint():
    first_turn = _plan("User: inspect the failing test")
    second_turn = _plan(
        "User: inspect the failing test\n"
        "Tool result: tests/test_example.py"
    )

    assert stable_prefix_fingerprint(first_turn) == (
        stable_prefix_fingerprint(second_turn)
    )


def test_changing_a_tool_definition_changes_the_stable_fingerprint():
    first_plan = _plan("User: inspect the failing test")

    changed_tools_plan = PromptCachePlan(
        sections=(
            PromptSection(
                kind="system",
                content="You are a careful coding agent.",
            ),
            PromptSection(
                kind="tools",
                content=(
                    '[{"name":"read_file"},'
                    '{"name":"grep"},'
                    '{"name":"bash"}]'
                ),
            ),
            PromptSection(
                kind="stable_context",
                content="Workspace paths must stay relative.",
            ),
            PromptSection(
                kind="dynamic",
                content="User: inspect the failing test",
            ),
        )
    )

    assert first_plan.stable_fingerprint != (
        changed_tools_plan.stable_fingerprint
    )


def test_rejects_cacheable_content_after_dynamic_content():
    with pytest.raises(
        PromptCachePlanError,
        match="prompt sections must be ordered",
    ):
        PromptCachePlan(
            sections=(
                PromptSection(
                    kind="system",
                    content="You are an agent.",
                ),
                PromptSection(
                    kind="dynamic",
                    content="User: fix the test.",
                ),
                PromptSection(
                    kind="tools",
                    content='[{"name":"read_file"}]',
                ),
            )
        )


def test_rejects_duplicate_cacheable_sections():
    with pytest.raises(
        PromptCachePlanError,
        match="cacheable section appears more than once: tools",
    ):
        PromptCachePlan(
            sections=(
                PromptSection(
                    kind="system",
                    content="You are an agent.",
                ),
                PromptSection(
                    kind="tools",
                    content='[{"name":"read_file"}]',
                ),
                PromptSection(
                    kind="tools",
                    content='[{"name":"grep"}]',
                ),
                PromptSection(
                    kind="dynamic",
                    content="User: search for TODO comments.",
                ),
            )
        )


def test_rejects_a_plan_without_cacheable_content():
    with pytest.raises(
        PromptCachePlanError,
        match="needs at least one cacheable section",
    ):
        PromptCachePlan(
            sections=(
                PromptSection(
                    kind="dynamic",
                    content="User: hello",
                ),
            )
        )


def test_rejects_an_empty_plan():
    with pytest.raises(
        PromptCachePlanError,
        match="must contain at least one section",
    ):
        PromptCachePlan(sections=())