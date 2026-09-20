from __future__ import annotations

from dataclasses import dataclass
from difflib import unified_diff

from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static


@dataclass(frozen=True, slots=True)
class DiffDocument:
    """The before/after text needed to render one local file change."""

    path: str
    before: str
    after: str


class DiffViewer(ModalScreen[None]):
    """Read-only modal preview of a single proposed or completed edit."""

    BINDINGS = [
        Binding(
            "escape",
            "close",
            "Close",
            show=False,
        ),
    ]

    DEFAULT_CSS = """
    DiffViewer {
        align: center middle;
    }

    #diff-panel {
        width: 96%;
        max-width: 120;
        height: 88%;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }

    #diff-title {
        height: 1;
        margin-bottom: 1;
    }

    #diff-content {
        height: 1fr;
        padding: 0 1;
        border: round $secondary;
    }

    #diff-close {
        width: 14;
        margin-top: 1;
        dock: right;
    }
    """

    def __init__(self, document: DiffDocument) -> None:
        super().__init__()
        self._document = document

    def compose(self) -> ComposeResult:
        diff = build_unified_diff(self._document)

        with Vertical(id="diff-panel"):
            yield Static(
                f"Diff preview — {self._document.path}",
                id="diff-title",
                markup=False,
            )
            with VerticalScroll(id="diff-content"):
                yield Static(
                    Syntax(
                        diff,
                        "diff",
                        theme="monokai",
                        word_wrap=True,
                    ),
                    markup=False,
                )
            yield Button("Close", id="diff-close")

    def action_close(self) -> None:
        self.dismiss()

    def on_button_pressed(
        self,
        event: Button.Pressed,
    ) -> None:
        if event.button.id == "diff-close":
            self.dismiss()


def build_unified_diff(document: DiffDocument) -> str:
    """Return a standard unified diff suitable for terminal display."""

    lines = unified_diff(
        document.before.splitlines(keepends=True),
        document.after.splitlines(keepends=True),
        fromfile=f"a/{document.path}",
        tofile=f"b/{document.path}",
        lineterm="",
    )
    rendered = "\n".join(lines)
    return rendered or "No textual changes."
