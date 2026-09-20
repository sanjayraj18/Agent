from __future__ import annotations

from rich.markdown import Markdown
from textual.containers import VerticalScroll
from textual.widgets import Static

from agent.tui.state import TuiState


class TranscriptView(VerticalScroll):
    """Scrollable Markdown transcript for user and assistant messages."""

    DEFAULT_CSS = """
    TranscriptView {
        height: 1fr;
        border: round $primary;
    }

    #transcript-content {
        padding: 1 2;
        width: 100%;
    }
    """

    def __init__(
        self,
        *,
        widget_id: str = "transcript",
    ) -> None:
        super().__init__(id=widget_id)
        self._content = Static(id="transcript-content")

    def compose(self):
        yield self._content

    def render_state(
        self,
        state: TuiState,
        *,
        show_thinking: bool = False,
    ) -> None:
        """Refresh the transcript from framework-independent TuiState."""

        document = build_transcript_markdown(
            state,
            show_thinking=show_thinking,
        )
        self._content.update(
            Markdown(
                document,
                code_theme="monokai",
            )
        )
        self.scroll_end(animate=False)


def build_transcript_markdown(
    state: TuiState,
    *,
    show_thinking: bool = False,
) -> str:
    """Build one Markdown document from the current transcript state."""

    if not state.transcript:
        return "_Waiting for a prompt..._"

    blocks: list[str] = []

    for message in state.transcript:
        title = (
            "You"
            if message.role == "user"
            else "Agent"
        )

        text = message.text or "_Working..._"
        blocks.append(f"## {title}\n\n{text}")

        if show_thinking and message.thinking:
            blocks.append(
                "### Thinking\n\n"
                f"> {message.thinking}"
            )

    return "\n\n---\n\n".join(blocks)