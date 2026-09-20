from __future__ import annotations

from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea


class PromptInput(TextArea):
    """
    Multi-line prompt editor.

    Enter creates a new line. Ctrl+Enter or Ctrl+S submits the prompt.
    """

    BINDINGS = [
        Binding(
            "ctrl+enter",
            "submit_prompt",
            "Send",
            show=True,
        ),
        Binding(
            "ctrl+s",
            "submit_prompt",
            "Send",
            show=False,
        ),
        Binding(
            "escape",
            "clear_prompt",
            "Clear",
            show=False,
        ),
    ]

    class Submitted(Message):
        """Posted to the parent app after the user submits a prompt."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def __init__(
        self,
        *,
        widget_id: str = "prompt-input",
    ) -> None:
        super().__init__("", id=widget_id)
        self.soft_wrap = True
        self.show_line_numbers = False
        self.border_title = (
            "Prompt · Ctrl+Enter to send"
        )

    @property
    def has_prompt(self) -> bool:
        return bool(self.text.strip())

    def action_submit_prompt(self) -> None:
        """Send non-empty input to the parent app."""

        value = self.text.strip()

        if not value:
            self.app.bell()
            return

        self.post_message(self.Submitted(value))

    def action_clear_prompt(self) -> None:
        """Clear the editor without sending anything."""

        self.clear()