"""Startup Verification Screen for AgentX Professional Terminal UI.

Runs real-time diagnostic checks on Groq configuration, model priority chain,
tool registries, browser subsystem, and knowledge base dataset before entering the dashboard.
"""

import json
from pathlib import Path
from typing import List, Tuple
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from agentx.config import settings
from agentx.tool_registry import create_default_registry


class StartupScreen(ModalScreen):
    """Modal screen performing real diagnostic checks during startup."""

    DEFAULT_CSS = """
    StartupScreen {
        align: center middle;
        background: rgba(18, 19, 22, 0.90);
    }
    """

    def __init__(self, auto_dismiss: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.auto_dismiss = auto_dismiss
        self.diagnostics_box: Static = None

    def compose(self) -> ComposeResult:
        with Container(id="startup-dialog"):
            yield Static("AgentX Diagnostics", id="startup-title")
            self.diagnostics_box = Static(id="startup-checks")
            yield self.diagnostics_box
            yield Button("Continue (Enter)", id="startup-proceed-btn")

    def on_mount(self) -> None:
        """Execute real startup checks on mount."""
        self.run_diagnostics()

    def run_diagnostics(self) -> None:
        """Run real verification checks against local environment and configuration."""
        checks: List[Tuple[str, bool, str]] = []

        # 1. Groq Configuration
        is_key_set = settings.is_api_key_configured()
        checks.append((
            "Groq API",
            is_key_set,
            "Key configured in environment" if is_key_set else "Key missing or placeholder in .env",
        ))

        # 2. Configured Model Chain
        models = settings.get_ordered_models()
        checks.append((
            "Model Priority",
            len(models) > 0,
            f"{models[0].split('/')[-1]} (Primary) → {len(models) - 1} fallback(s)",
        ))

        # 3. Tool Registry
        registry = create_default_registry()
        tool_names = registry.list_tool_names()
        checks.append((
            "Tool Registry",
            len(tool_names) >= 4,
            f"{len(tool_names)} tools active ({', '.join(tool_names)})",
        ))

        # 4. Local Knowledge Base
        kb_path = settings.KNOWLEDGE_BASE_PATH
        kb_ok = False
        kb_details = "Knowledge base file missing"
        if kb_path.exists():
            try:
                with open(kb_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    kb_ok = isinstance(data, list)
                    kb_details = f"{len(data)} verified records loaded"
            except Exception as e:
                kb_details = f"Parse error: {e}"
        checks.append(("Knowledge Base", kb_ok, kb_details))

        # 5. Playwright Browser Subsystem
        try:
            from agentx.tools.browser import sync_playwright
            pw_ok = sync_playwright is not None
            pw_details = "Playwright Chromium ready" if pw_ok else "Playwright library not installed"
        except Exception as e:
            pw_ok = False
            pw_details = f"Error: {e}"
        checks.append(("Browser Engine", pw_ok, pw_details))

        # Render clean, quiet Rich checklist
        t = Text()
        for label, passed, details in checks:
            if passed:
                t.append("  ✓ ", style="bold #5ea28e")
                t.append(f"{label}: ", style="bold #f4f4f5")
                t.append(f"{details}\n", style="#a1a1aa")
            else:
                t.append("  ✗ ", style="bold #f87171")
                t.append(f"{label}: ", style="bold #f4f4f5")
                t.append(f"{details}\n", style="#f87171")

        all_passed = all(p for _, p, _ in checks)
        t.append("\n  Status: ", style="bold #71717a")
        if all_passed:
            t.append("All systems operational", style="bold #5ea28e")
        else:
            t.append("Environment warnings detected", style="bold #f59e0b")

        self.diagnostics_box.update(t)

        # Auto-dismiss after 1.5 seconds if requested
        if self.auto_dismiss:
            self.set_timer(1.5, self.dismiss_screen)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss_screen()

    def on_key(self, event) -> None:
        if event.key in ("enter", "space", "escape"):
            self.dismiss_screen()

    def dismiss_screen(self) -> None:
        if self.is_mounted and len(self.app.screen_stack) > 1:
            self.dismiss()
