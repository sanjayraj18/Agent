from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ApprovalDialog(ModalScreen[bool]):
    """A fail-closed human decision dialog for a proposed tool action."""

    BINDINGS = [
        Binding(
            "escape",
            "deny",
            "Deny",
            show=False,
        ),
    ]

    DEFAULT_CSS = """
    ApprovalDialog {
        align: center middle;
    }

    #approval-panel {
        width: 92%;
        max-width: 88;
        height: auto;
        max-height: 85%;
        padding: 1 2;
        border: thick $warning;
        background: $surface;
    }

    #approval-title {
        margin-bottom: 1;
    }

    #approval-details {
        height: auto;
        max-height: 16;
        overflow: auto;
        padding: 1;
        border: round $primary;
    }

    #approval-actions {
        margin-top: 1;
        height: auto;
        align: right middle;
    }

    #approval-actions Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        *,
        action: str,
        details: str,
    ) -> None:
        super().__init__()
        self._action = action
        self._details = details

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-panel"):
            yield Static(
                f"Allow this action?  {self._action}",
                id="approval-title",
                markup=False,
            )
            yield Static(
                self._details,
                id="approval-details",
                markup=False,
            )
            with Horizontal(id="approval-actions"):
                yield Button(
                    "Deny",
                    id="deny",
                    variant="error",
                )
                yield Button(
                    "Allow once",
                    id="allow",
                    variant="success",
                )

    def action_deny(self) -> None:
        """Escape always denies; approval must be an explicit action."""

        self.dismiss(False)

    def on_button_pressed(
        self,
        event: Button.Pressed,
    ) -> None:
        self.dismiss(event.button.id == "allow")
