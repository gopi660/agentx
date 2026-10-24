"""Conversation History component for AgentX Professional Terminal UI.

Presents user inquiries, agent responses, inline tool execution traces,
and error notices with clear visual separation, natural sentence case,
understated metadata, and zero decorative box borders.
"""

from typing import Optional
from rich.markdown import Markdown
from rich.text import Text
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static


class UserMessageWidget(Static):
    """User inquiry block with comfortable padding and understated typography."""

    def __init__(self, prompt: str, **kwargs):
        super().__init__(classes="user-message-block", **kwargs)
        self.prompt = prompt
        self.styles.height = "auto"

    def on_mount(self) -> None:
        t = Text()
        t.append("You\n", style="bold #5ea28e")
        t.append(f"  {self.prompt}\n", style="#f4f4f5")
        self.update(t)


class InlineToolTraceWidget(Static):
    """Compact inline status event for tool executions."""

    def __init__(self, tool_name: str, status: str, details: str = "", **kwargs):
        super().__init__(classes="inline-tool-trace", **kwargs)
        self.tool_name = tool_name
        self.status = status
        self.details = details
        self.styles.height = "auto"

    def on_mount(self) -> None:
        t = Text()
        if self.status == "success":
            t.append("  ✓ ", style="bold #5ea28e")
        elif self.status == "no_results":
            t.append("  ○ ", style="bold #f59e0b")
        else:
            t.append("  ✗ ", style="bold #f87171")

        t.append(self.tool_name, style="bold #e4e4e7")
        if self.details:
            t.append(f" · {self.details}", style="#71717a")
        self.update(t)


class AgentMessageWidget(Vertical):
    """Agent response block with understated metadata and Rich Markdown body."""

    def __init__(
        self,
        answer: str,
        model_used: Optional[str] = None,
        fallback: bool = False,
        elapsed_s: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(classes="agent-message-block", **kwargs)
        self.answer = answer
        self.model_used = model_used or "openai/gpt-oss-120b"
        self.fallback = fallback
        self.elapsed_s = elapsed_s
        self.styles.height = "auto"

    def compose(self):
        # Understated metadata header
        header_text = Text()
        header_text.append("AgentX", style="bold #a1a1aa")
        header_text.append(f" · {self.model_used}", style="#71717a")
        if self.fallback:
            header_text.append(" (fallback)", style="italic #f59e0b")
        if self.elapsed_s is not None:
            header_text.append(f" · {self.elapsed_s:.2f}s", style="#71717a")

        header_widget = Static(header_text, classes="agent-message-header")
        header_widget.styles.height = "auto"
        yield header_widget

        body_text = self.answer.strip() if (self.answer and self.answer.strip()) else "_No response content produced._"
        # Render markdown content inside clean static body with auto height
        body_widget = Static(Markdown(body_text), classes="agent-message-body")
        body_widget.styles.height = "auto"
        yield body_widget


class SystemNoticeWidget(Static):
    """Notice card for system events, rephrasing, or startup info."""

    def __init__(self, notice: str, color: str = "#71717a", **kwargs):
        super().__init__(classes="system-notice-block", **kwargs)
        self.notice = notice
        self.color = color
        self.styles.height = "auto"

    def on_mount(self) -> None:
        t = Text(f"• {self.notice}", style=f"{self.color}")
        self.update(t)


class ErrorNoticeWidget(Static):
    """Notice card for agent or API failures."""

    def __init__(self, error: str, **kwargs):
        super().__init__(classes="error-notice-block", **kwargs)
        self.error = error
        self.styles.height = "auto"

    def on_mount(self) -> None:
        t = Text()
        t.append("✕ Error: ", style="bold #f87171")
        t.append(self.error, style="#f87171")
        self.update(t)


class ConversationHistory(VerticalScroll):
    """Scrollable container maintaining all conversation messages."""

    def add_user_message(self, prompt: str) -> None:
        """Add a user message and scroll to bottom."""
        widget = UserMessageWidget(prompt)
        self.mount(widget)
        self.call_after_refresh(self.scroll_end, animate=False)

    def add_inline_tool_trace(self, tool_name: str, status: str, details: str = "") -> None:
        """Add a compact inline tool trace event."""
        widget = InlineToolTraceWidget(tool_name=tool_name, status=status, details=details)
        self.mount(widget)
        self.call_after_refresh(self.scroll_end, animate=False)

    def add_agent_response(
        self,
        answer: str,
        model_used: Optional[str] = None,
        fallback: bool = False,
        elapsed_s: Optional[float] = None,
    ) -> None:
        """Add an agent response and scroll to bottom."""
        widget = AgentMessageWidget(
            answer=answer,
            model_used=model_used,
            fallback=fallback,
            elapsed_s=elapsed_s,
        )
        self.mount(widget)
        self.call_after_refresh(self.scroll_end, animate=False)

    def add_system_notice(self, notice: str, color: str = "#71717a") -> None:
        widget = SystemNoticeWidget(notice, color=color)
        self.mount(widget)
        self.call_after_refresh(self.scroll_end, animate=False)

    def add_error_notice(self, error: str) -> None:
        widget = ErrorNoticeWidget(error)
        self.mount(widget)
        self.call_after_refresh(self.scroll_end, animate=False)

    def clear_conversation(self) -> None:
        """Remove all message widgets."""
        for child in list(self.children):
            child.remove()
        self.mount(SystemNoticeWidget("Conversation cleared.", color="#71717a"))
