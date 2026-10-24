"""Help and Keybindings Modal component for AgentX Professional UI."""

from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class HelpModal(ModalScreen):
    """Modal displaying keyboard shortcuts and operational guidance."""

    DEFAULT_CSS = """
    HelpModal {
        align: center middle;
        background: rgba(18, 19, 22, 0.85);
    }
    """

    def compose(self) -> ComposeResult:
        with Container(id="help-dialog"):
            yield Static("Keyboard Shortcuts & Help", id="help-title")

            table = Table(box=None, show_header=True, header_style="bold #5ea28e")
            table.add_column("Shortcut", style="bold #f4f4f5", width=16)
            table.add_column("Action", style="#a1a1aa")

            table.add_row("Enter", "Submit inquiry in input composer")
            table.add_row("Ctrl+C / Ctrl+Q", "Safely exit application and close browser")
            table.add_row("Ctrl+N", "Start fresh conversation session (new memory)")
            table.add_row("Ctrl+L", "Clear conversation view and session memory")
            table.add_row("Ctrl+T", "Toggle secondary wireframe view")
            table.add_row("Ctrl+M", "Show configured model priority chain")
            table.add_row("F1 or ?", "Open this help dialog")
            table.add_row("Esc", "Dismiss active modal")

            with Container(id="help-content"):
                yield Static(table)
                yield Static(
                    "\n[bold #f4f4f5]Capabilities:[/] Math (AST calculator), Webpage extraction (read-only), "
                    "Visible browser (headed YouTube / web lookup), System time & timezone, Local knowledge base.\n",
                )

            yield Button("Close (Esc)", id="help-close-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(True)

    def on_key(self, event) -> None:
        if event.key in ("escape", "enter"):
            self.dismiss(True)
