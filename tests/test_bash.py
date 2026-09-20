import asyncio
import re
import shlex
import sys
from pathlib import Path

import pytest

from agent.core.prompt_cache import PromptCachePlanError
from agent.providers.base import Message, ProviderRequest, TextPart, ToolSpec
from agent.tools.bash import BashTool
from agent.tools.processes import ProcessRegistry
from agent.tools.workspace import Workspace


def _python_command(script: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"


def _tool(tmp_path: Path) -> BashTool:
    return BashTool(Workspace(tmp_path), registry=ProcessRegistry())


async def test_runs_a_foreground_command_in_the_workspace(tmp_path: Path):
    result = await _tool(tmp_path).execute(
        {"command": _python_command("print('hello from bash')")}
    )

    assert result.is_error is False
    assert result.content == "hello from bash"


async def test_reports_foreground_failure_and_timeout(tmp_path: Path):
    tool = _tool(tmp_path)

    failed = await tool.execute({"command": _python_command("raise SystemExit(3)")})
    timed_out = await tool.execute(
        {
            "command": _python_command("import time; time.sleep(30)"),
            "timeout_seconds": 0.1,
        }
    )

    assert failed.is_error is True
    assert failed.content.startswith("command exited with status 3")
    assert timed_out.is_error is True
    assert timed_out.content.startswith("command timed out after 0.1 seconds")


async def test_starts_then_inspects_a_background_command(tmp_path: Path):
    tool = _tool(tmp_path)
    started = await tool.execute(
        {
            "command": _python_command(
                "import time; print('ready', flush=True); time.sleep(0.05)"
            ),
            "background": True,
        }
    )
    match = re.search(r"job ([0-9a-f]+)", started.content)

    assert started.is_error is False
    assert match is not None

    await asyncio.sleep(0.1)
    status = await tool.execute({"action": "status", "job_id": match.group(1)})

    assert status.is_error is False
    assert "completed" in status.content
    assert "ready" in status.content


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"command": ""}, "command must be a non-empty string"),
        ({"action": "delete"}, "action must be one of: run, status, stop"),
        ({"command": "echo hi", "background": "yes"}, "background must be a boolean"),
        ({"command": "echo hi", "timeout_seconds": True}, "timeout_seconds must be a number"),
        ({"action": "status", "job_id": "missing"}, "unknown background job: missing"),
        ({"command": "echo hi", "unknown": True}, "unexpected arguments: unknown"),
    ],
)
async def test_rejects_invalid_bash_requests(
    tmp_path: Path,
    arguments: dict,
    message: str,
):
    result = await _tool(tmp_path).execute(arguments)

    assert result.is_error is True
    assert result.content == message

def _tool_spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} tool",
        input_schema={"type": "object"},
    )


def _user_message(text: str) -> Message:
    return Message(
        role="user",
        content=[TextPart(text=text)],
    )


def test_request_has_no_cache_plan_when_caching_is_disabled():
    request = ProviderRequest(
        model="claude-sonnet-5",
        max_tokens=1_024,
        system="You are a coding agent.",
        messages=[_user_message("Inspect the test failure.")],
    )

    assert request.cache_plan is None


def test_request_builds_a_cache_plan_in_stable_then_dynamic_order():
    request = ProviderRequest(
        model="claude-sonnet-5",
        max_tokens=1_024,
        system="You are a coding agent.",
        stable_context="Only modify files inside the workspace.",
        tools=[_tool_spec("read_file"), _tool_spec("grep")],
        messages=[_user_message("Inspect the test failure.")],
        cache_stable_prefix=True,
    )

    plan = request.cache_plan

    assert plan is not None
    assert [section.kind for section in plan.sections] == [
        "system",
        "tools",
        "stable_context",
        "dynamic",
    ]
    assert request.system_prompt == (
        "You are a coding agent.\n\n"
        "Only modify files inside the workspace."
    )


def test_changing_messages_does_not_change_request_cache_fingerprint():
    common_fields = {
        "model": "claude-sonnet-5",
        "max_tokens": 1_024,
        "system": "You are a coding agent.",
        "tools": [_tool_spec("read_file")],
        "cache_stable_prefix": True,
    }

    first_request = ProviderRequest(
        **common_fields,
        messages=[_user_message("Read src/agent/main.py")],
    )
    second_request = ProviderRequest(
        **common_fields,
        messages=[
            _user_message(
                "Tool result: file contents\n"
                "Now explain the failing test."
            )
        ],
    )

    assert first_request.cache_plan is not None
    assert second_request.cache_plan is not None
    assert first_request.cache_plan.stable_fingerprint == (
        second_request.cache_plan.stable_fingerprint
    )


def test_changing_tool_registration_order_changes_the_fingerprint():
    common_fields = {
        "model": "claude-sonnet-5",
        "max_tokens": 1_024,
        "system": "You are a coding agent.",
        "messages": [_user_message("Inspect the project.")],
        "cache_stable_prefix": True,
    }

    first_request = ProviderRequest(
        **common_fields,
        tools=[_tool_spec("read_file"), _tool_spec("grep")],
    )
    second_request = ProviderRequest(
        **common_fields,
        tools=[_tool_spec("grep"), _tool_spec("read_file")],
    )

    assert first_request.cache_plan is not None
    assert second_request.cache_plan is not None
    assert first_request.cache_plan.stable_fingerprint != (
        second_request.cache_plan.stable_fingerprint
    )


def test_enabled_caching_requires_some_stable_content():
    request = ProviderRequest(
        model="claude-sonnet-5",
        max_tokens=1_024,
        messages=[_user_message("Hello")],
        cache_stable_prefix=True,
    )

    with pytest.raises(
        PromptCachePlanError,
        match="needs at least one cacheable section",
    ):
        _ = request.cache_plan
