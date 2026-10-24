"""System Metrics & Engine Status component for AgentX UI."""

from rich.text import Text
from textual.widgets import Static
from agentx.config import settings


class SystemMetricsBar(Static):
    """Panel displaying active models, tool availability, and live session stats."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.queries_count: int = 0
        self.tools_count: int = 0
        self.fallbacks_count: int = 0
        self.last_latency_s: float = 0.0

    def on_mount(self) -> None:
        self.refresh_metrics()

    def record_query(self, tools_used: int, fallback_engaged: bool, latency: float) -> None:
        """Update metrics from a completed inquiry."""
        self.queries_count += 1
        self.tools_count += tools_used
        if fallback_engaged:
            self.fallbacks_count += 1
        self.last_latency_s = latency
        self.refresh_metrics()

    def refresh_metrics(self) -> None:
        """Render understated metrics layout."""
        ordered_models = settings.get_ordered_models()
        primary = ordered_models[0].split("/")[-1] if ordered_models else "gpt-oss-120b"
        fallbacks = ", ".join(m.split("/")[-1] for m in ordered_models[1:]) if len(ordered_models) > 1 else "none"

        text = Text()
        text.append("Models: ", style="bold #a1a1aa")
        text.append(f"{primary} → {fallbacks}\n", style="#71717a")

        text.append("Tools: ", style="bold #a1a1aa")
        text.append("calc · browser · headed · time · kb\n", style="#71717a")

        text.append("Stats: ", style="bold #a1a1aa")
        text.append(
            f"{self.queries_count} queries · {self.tools_count} tools · "
            f"{self.fallbacks_count} failovers · {self.last_latency_s:.2f}s",
            style="#71717a",
        )

        self.update(text)
