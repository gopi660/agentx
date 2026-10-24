"""Live Operational Activity & Telemetry Log component for AgentX UI."""

import datetime
from typing import Any, Dict
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static


class ActivityPanel(Static):
    """Panel displaying real-time agent execution telemetry."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_container: VerticalScroll = None

    def compose(self):
        yield Static("Recent Activity", id="activity-header")
        self.log_container = VerticalScroll(id="activity-log")
        yield self.log_container

    def on_mount(self) -> None:
        self.add_entry("AgentX telemetry ready", category="system")

    def add_entry(self, message: str, category: str = "info") -> None:
        """Append an operational event to the telemetry log."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        text = Text()
        text.append(f"{timestamp} ", style="dim #52525b")

        if category == "tool_success":
            text.append("✓ ", style="bold #5ea28e")
            text.append(message, style="#d4d4d8")
        elif category == "tool_no_results":
            text.append("○ ", style="bold #f59e0b")
            text.append(message, style="#fbbf24")
        elif category == "tool_error":
            text.append("✗ ", style="bold #f87171")
            text.append(message, style="#f87171")
        elif category == "fallback":
            text.append("⟳ ", style="bold #f59e0b")
            text.append(message, style="#fcd34d")
        elif category == "model":
            text.append("◈ ", style="bold #5ea28e")
            text.append(message, style="#a1a1aa")
        else:
            text.append("• ", style="#71717a")
            text.append(message, style="#71717a")

        entry_widget = Static(text)
        if self.log_container:
            self.log_container.mount(entry_widget)
            entry_widget.scroll_visible()

    def handle_agent_trace(self, trace: Dict[str, Any]) -> None:
        """Handle trace dictionary emitted by the AgentX backend."""
        event_type = trace.get("event")

        if event_type == "model_fallback":
            model = trace.get("model", "unknown")
            attempts = trace.get("attempts", 1)
            self.add_entry(f"Model failover engaged -> Switched to '{model}' (Attempt {attempts})", category="fallback")
            return

        tool_name = trace.get("tool", "unknown_tool")
        status = trace.get("status", "unknown")
        summary = trace.get("message") or trace.get("error") or ""

        if status == "success":
            cat = "tool_success"
        elif status == "no_results":
            cat = "tool_no_results"
        else:
            cat = "tool_error"

        # Truncate summary cleanly if long
        short_summary = summary if len(summary) <= 50 else summary[:47] + "..."
        self.add_entry(f"{tool_name} [{status}]: {short_summary}", category=cat)
