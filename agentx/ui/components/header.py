"""Compact, refined application header for AgentX Terminal UI."""

from rich.text import Text
from textual.widgets import Static


class HeaderBar(Static):
    """Understated header displaying application identity, state, and session."""

    def __init__(self, session_id: str, version: str = "v2.0", **kwargs):
        super().__init__(**kwargs)
        self.session_id = session_id
        self.version = version
        self.current_status = "Ready"
        self.status_color = "#5ea28e"  # Sage green

    def on_mount(self) -> None:
        self.refresh_display()

    def set_status(self, status: str, color: str = "#5ea28e") -> None:
        """Update the operational status displayed in the header."""
        self.current_status = status
        self.status_color = color
        self.refresh_display()

    def refresh_display(self) -> None:
        """Render the clean horizontal header without noisy borders or banners."""
        text = Text()
        text.append("AgentX", style="bold #f4f4f5")
        text.append(f" {self.version}", style="#71717a")
        text.append("  ·  ", style="#3f3f46")

        # Operational status in natural case
        text.append(self.current_status, style=f"bold {self.status_color}")
        text.append("  ·  ", style="#3f3f46")

        # Shortened session identifier
        text.append(f"{self.session_id}", style="#71717a")

        self.update(text)
