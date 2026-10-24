"""Main Textual Application for AgentX Professional Terminal UI.

Coordinates asynchronous backend execution, real-time agent telemetry,
clean inline tool event streams, and rock-solid input stability inside
a restrained developer command center.
"""

import argparse
import sys
import time
import uuid
from typing import Any, Dict, Optional

# Ensure clean UTF-8 rendering on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Input, Static

from agentx.agent import AgentX
from agentx.config import settings
from agentx.fallback import sanitize_message
from agentx.ui.components.activity_panel import ActivityPanel
from agentx.ui.components.conversation import ConversationHistory
from agentx.ui.components.header import HeaderBar
from agentx.ui.components.help_modal import HelpModal
from agentx.ui.components.startup_screen import StartupScreen
from agentx.ui.components.system_panel import SystemMetricsBar
from agentx.ui.components.wireframe_3d import Wireframe3DWidget
from agentx.ui.styles import AGENTX_TCSS


class AgentXTerminalApp(App):
    """Professional interactive terminal interface for AgentX."""

    CSS = AGENTX_TCSS
    TITLE = "AgentX"
    SUB_TITLE = "Tool-Using AI Agent"

    BINDINGS = [
        ("ctrl+q", "quit_app", "Quit"),
        ("ctrl+c", "quit_app", "Quit"),
        ("ctrl+n", "new_conversation", "New Chat"),
        ("ctrl+l", "clear_conversation", "Clear"),
        ("ctrl+t", "toggle_animation", "Wireframe"),
        ("ctrl+m", "show_models_info", "Models"),
        ("f1", "show_help", "Help"),
        ("question_mark", "show_help", "Help"),
    ]

    def __init__(
        self,
        agent: Optional[AgentX] = None,
        session_id: Optional[str] = None,
        skip_startup: bool = False,
        enable_animation: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self.skip_startup = skip_startup
        self.enable_animation = enable_animation
        self.is_processing = False

        # Lazy initialize backend agent if API key is configured
        self.agent = agent
        if self.agent is None and settings.is_api_key_configured():
            try:
                self.agent = AgentX()
            except Exception:
                self.agent = None

        # Widget references
        self.header_bar: HeaderBar = None
        self.conversation: ConversationHistory = None
        self.sidebar_container: Vertical = None
        self.activity_panel: ActivityPanel = None
        self.system_panel: SystemMetricsBar = None
        self.wireframe_widget: Wireframe3DWidget = None
        self.prompt_input: Input = None
        self.hints_bar: Static = None

    def compose(self) -> ComposeResult:
        """Compose the responsive, clean developer dashboard layout."""
        # 1. Compact header bar
        self.header_bar = HeaderBar(session_id=self.session_id, id="app-header")
        yield self.header_bar

        # 2. Middle main body
        with Horizontal(id="main-container"):
            # Left: Conversation stream (74% desktop, 100% on narrow)
            with Vertical(id="chat-container"):
                self.conversation = ConversationHistory(id="conversation-history")
                yield self.conversation

            # Right: Telemetry sidebar (26% desktop, hidden on narrow)
            with Vertical(id="sidebar-container") as sidebar:
                self.sidebar_container = sidebar
                # Faux-3D widget (disabled and hidden by default)
                self.wireframe_widget = Wireframe3DWidget(
                    enabled=self.enable_animation, id="wireframe-canvas"
                )
                yield self.wireframe_widget

                self.activity_panel = ActivityPanel(id="activity-panel")
                yield self.activity_panel

                self.system_panel = SystemMetricsBar(id="system-metrics")
                yield self.system_panel

        # 3. Dedicated persistent bottom composer (guaranteed 5 cells, cannot be overlapped)
        with Vertical(id="composer-container"):
            self.prompt_input = Input(
                placeholder="Send an inquiry... (Enter to submit, ? for help)",
                id="prompt-input",
            )
            yield self.prompt_input
            self.hints_bar = Static(
                "Enter Submit   Ctrl+L Clear   Ctrl+M Models   Ctrl+T Wireframe   F1 Help   Ctrl+C Quit",
                id="input-hints",
            )
            yield self.hints_bar

    def on_mount(self) -> None:
        """Present startup diagnostics or jump directly to dashboard."""
        if not self.skip_startup:
            self.push_screen(StartupScreen(auto_dismiss=True))

        # Initial clean system notice
        models_str = " → ".join(m.split("/")[-1] for m in settings.get_ordered_models())
        self.conversation.add_system_notice(
            f"AgentX initialized · Active models: {models_str}"
        )

        # Responsive check: hide sidebar if starting on a compact terminal
        if self.size.width < 96 and self.sidebar_container:
            self.sidebar_container.display = False

        self.prompt_input.focus()

    def on_resize(self, event: events.Resize) -> None:
        """Progressively collapse secondary telemetry on narrow terminals."""
        if self.sidebar_container:
            if event.size.width < 96:
                self.sidebar_container.display = False
            else:
                self.sidebar_container.display = True

        # Keep focus on prompt input during terminal resize
        if self.prompt_input and self.is_mounted:
            self.prompt_input.focus()

    def on_screen_resume(self) -> None:
        """Regain input focus automatically whenever any modal dialog is dismissed."""
        if self.prompt_input and self.is_mounted:
            self.prompt_input.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission asynchronously."""
        user_text = event.value.strip()
        if not user_text:
            return

        # Check for session control commands
        if user_text.lower() in ("/new", "/reset", "new session"):
            self.prompt_input.value = ""
            self.action_new_conversation()
            return

        if user_text.lower() in ("/clear", "clear"):
            self.prompt_input.value = ""
            self.action_clear_conversation()
            return

        if self.is_processing:
            self.conversation.add_system_notice(
                "Agent is currently processing a prior inquiry. Please wait.", color="#f59e0b"
            )
            return

        # Clear input field immediately
        self.prompt_input.value = ""

        # Post user inquiry to conversation view
        self.conversation.add_user_message(user_text)

        # Update state and header status
        self.is_processing = True
        self.header_bar.set_status("Thinking...", color="#f59e0b")
        self.activity_panel.add_entry(f"Dispatched: '{user_text}'", category="model")

        # Launch background worker thread
        self._execute_agent_query(user_text)

    @work(thread=True)
    def _execute_agent_query(self, prompt: str) -> None:
        """Worker thread executing the AgentX inquiry without blocking UI rendering."""
        start_time = time.time()
        try:
            if self.agent is None:
                if not settings.is_api_key_configured():
                    self.call_from_thread(
                        self.conversation.add_error_notice,
                        "GROQ_API_KEY is not configured. Set GROQ_API_KEY in your .env file to enable live model queries.",
                    )
                    self.call_from_thread(self.header_bar.set_status, "API Key Missing", "#f87171")
                    self.call_from_thread(self.prompt_input.focus)
                    return
                try:
                    self.agent = AgentX()
                except Exception as e:
                    clean_err = sanitize_message(str(e))
                    self.call_from_thread(
                        self.conversation.add_error_notice,
                        f"Failed to initialize AgentX: {clean_err}",
                    )
                    self.call_from_thread(self.header_bar.set_status, "Init Error", "#f87171")
                    self.call_from_thread(self.prompt_input.focus)
                    return

            def on_trace_callback(trace: Dict[str, Any]) -> None:
                self.call_from_thread(self.activity_panel.handle_agent_trace, trace)
                tool_name = trace.get("tool")
                if tool_name:
                    status = trace.get("status", "unknown")
                    summary = trace.get("message") or trace.get("error") or ""
                    self.call_from_thread(
                        self.conversation.add_inline_tool_trace,
                        tool_name=tool_name,
                        status=status,
                        details=summary,
                    )

            response = self.agent.run(
                user_prompt=prompt,
                session_id=self.session_id,
                trace_callback=on_trace_callback,
            )
            elapsed = time.time() - start_time

            # Post results safely on UI thread
            if response.error and not response.final_answer:
                self.call_from_thread(self.conversation.add_error_notice, response.error)
                self.call_from_thread(self.header_bar.set_status, "Error", "#f87171")
            else:
                self.call_from_thread(
                    self.conversation.add_agent_response,
                    answer=response.final_answer,
                    model_used=response.model_used,
                    fallback=response.fallback_occurred,
                    elapsed_s=elapsed,
                )
                self.call_from_thread(
                    self.system_panel.record_query,
                    tools_used=len(response.tool_traces),
                    fallback_engaged=response.fallback_occurred,
                    latency=elapsed,
                )
                self.call_from_thread(self.header_bar.set_status, "Ready", "#5ea28e")

            model_short = (response.model_used or "model").split("/")[-1]
            self.call_from_thread(
                self.activity_panel.add_entry,
                f"Completed in {elapsed:.2f}s ({model_short})",
                category="tool_success",
            )

        except Exception as e:
            elapsed = time.time() - start_time
            err_msg = sanitize_message(str(e))
            self.call_from_thread(
                self.conversation.add_error_notice,
                f"Execution failed: {err_msg}",
            )
            self.call_from_thread(self.header_bar.set_status, "Error", "#f87171")
            self.call_from_thread(
                self.activity_panel.add_entry,
                f"Aborted: {err_msg}",
                category="tool_error",
            )
        finally:
            self.is_processing = False
            self.call_from_thread(self.prompt_input.focus)

    def action_quit_app(self) -> None:
        """Cleanly exit application, stopping timers and background threads."""
        if self.wireframe_widget:
            self.wireframe_widget.stop_animation()
        try:
            from agentx.tools.browser import cleanup_visible_browser
            cleanup_visible_browser()
        except Exception:
            pass
        self.exit()

    def action_clear_conversation(self) -> None:
        """Clear conversation stream and active session checkpoint."""
        self.is_processing = False
        if self.header_bar:
            self.header_bar.set_status("Ready", "#5ea28e")
        if self.conversation:
            self.conversation.clear_conversation()
        if self.agent:
            try:
                self.agent.clear_session(self.session_id)
            except Exception:
                pass
        if self.activity_panel:
            self.activity_panel.add_entry(f"Cleared memory for session: {self.session_id}", category="system")
        if self.prompt_input:
            self.prompt_input.focus()

    def action_new_conversation(self) -> None:
        """Start a new conversation with a fresh session identifier and empty memory."""
        self.is_processing = False
        if self.agent:
            self.session_id = self.agent.start_new_session()
        else:
            self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        if self.header_bar:
            self.header_bar.session_id = self.session_id
            self.header_bar.set_status("Ready", "#5ea28e")
            self.header_bar.refresh()
        if self.conversation:
            self.conversation.clear_conversation()
        if self.activity_panel:
            self.activity_panel.add_entry(f"Started new session: {self.session_id}", category="system")
        if self.prompt_input:
            self.prompt_input.focus()

    def action_toggle_animation(self) -> None:
        """Toggle 3D wireframe secondary telemetry."""
        if self.wireframe_widget:
            is_active = self.wireframe_widget.toggle_animation()
            msg = "Wireframe view enabled" if is_active else "Wireframe view hidden"
            self.activity_panel.add_entry(msg, category="system")

    def action_show_models_info(self) -> None:
        """Display model chain in activity log."""
        models = settings.get_ordered_models()
        self.activity_panel.add_entry(
            f"Configured models: Primary='{models[0]}', Fallbacks={models[1:]}",
            category="model",
        )

    def action_show_help(self) -> None:
        """Display help dialog."""
        self.push_screen(HelpModal())


def run_ui(argv=None) -> int:
    """CLI launcher for AgentX Professional Terminal UI."""
    parser = argparse.ArgumentParser(description="AgentX Terminal UI")
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Session identifier",
    )
    parser.add_argument(
        "--skip-startup",
        action="store_true",
        help="Skip startup diagnostic sequence",
    )
    parser.add_argument(
        "--enable-animation",
        action="store_true",
        help="Start with secondary 3D wireframe animation visible",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Start with animation disabled (default)",
    )
    args = parser.parse_args(argv)

    enable_anim = args.enable_animation and not args.no_animation

    app = AgentXTerminalApp(
        session_id=args.session,
        skip_startup=args.skip_startup,
        enable_animation=enable_anim,
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(run_ui())
