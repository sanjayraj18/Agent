"""Reusable Textual widgets used by the agent terminal UI."""

from agent.tui.widgets.approval_dialog import ApprovalDialog
from agent.tui.widgets.diff_viewer import DiffDocument, DiffViewer
from agent.tui.widgets.prompt_input import PromptInput
from agent.tui.widgets.status_bar import StatusBar
from agent.tui.widgets.tool_activity import ToolActivityView
from agent.tui.widgets.transcript import TranscriptView

__all__ = [
    "ApprovalDialog",
    "DiffDocument",
    "DiffViewer",
    "PromptInput",
    "StatusBar",
    "ToolActivityView",
    "TranscriptView",
]
