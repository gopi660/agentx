"""Tool registry and validation engine for AgentX.

Manages tool registrations, generates Groq-compliant function schemas,
validates model-invoked tool calls against Pydantic models, and executes tools.
"""

from dataclasses import dataclass
import json
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel, ValidationError

from agentx.schemas import (
    BrowserArgs,
    CalculatorArgs,
    CurrentTimeArgs,
    KnowledgeBaseArgs,
    ToolResult,
    ToolStatus,
    VisibleBrowserArgs,
    WebSearchArgs,
)
from agentx.tools.browser import browse_page, open_visible_browser
from agentx.tools.calculator import safe_calculate
from agentx.tools.knowledge_base import search_knowledge_base
from agentx.tools.search import web_search
from agentx.tools.time_tool import get_current_time



@dataclass
class ToolDefinition:
    """Registered tool metadata and execution handler."""

    name: str
    description: str
    args_schema: Type[BaseModel]
    handler: Callable[..., ToolResult]

    def to_groq_schema(self) -> Dict[str, Any]:
        """Convert tool definition into Groq/OpenAI function calling schema."""
        # Extract json schema from Pydantic model
        schema = self.args_schema.model_json_schema()
        # Clean up internal pydantic titles if present
        parameters = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


class ToolRegistry:
    """Registry managing allowed agent tools and argument validation."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        args_schema: Type[BaseModel],
        handler: Callable[..., ToolResult],
    ) -> None:
        """Register a new tool.

        Args:
            name: Unique name of the tool.
            description: Clear instructions for the LLM explaining when to use this tool.
            args_schema: Pydantic model for parameter validation.
            handler: Callable function returning ToolResult.
        """
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            args_schema=args_schema,
            handler=handler,
        )

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Retrieve a tool definition by name."""
        return self._tools.get(name)

    def list_tool_names(self) -> List[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    def get_groq_tools(self) -> List[Dict[str, Any]]:
        """Return list of Groq-compatible tool schemas for registered tools."""
        return [tool.to_groq_schema() for tool in self._tools.values()]

    def execute(self, tool_name: str, raw_args: Any) -> ToolResult:
        """Validate and execute a tool call requested by the model.

        Args:
            tool_name: Requested tool name.
            raw_args: Dict or JSON string of arguments provided by the model.

        Returns:
            ToolResult containing execution status and output or validation errors.
        """
        tool = self.get_tool(tool_name)
        if not tool:
            allowed = ", ".join(self.list_tool_names())
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Unrecognized tool '{tool_name}'. Allowed tools are: [{allowed}].",
            )

        # Parse arguments if passed as JSON string
        args_dict: Dict[str, Any] = {}
        if isinstance(raw_args, str):
            try:
                args_dict = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError as e:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    error=f"Malformed JSON arguments for tool '{tool_name}': {str(e)}",
                )
        elif isinstance(raw_args, dict):
            args_dict = raw_args
        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                error=f"Arguments for tool '{tool_name}' must be a JSON object or dictionary.",
            )

        # Validate arguments against Pydantic schema
        try:
            validated_args = tool.args_schema.model_validate(args_dict)
        except ValidationError as e:
            # Format clean error message
            error_details = "; ".join(
                f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                for err in e.errors()
            )
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                error=f"Argument validation failed for tool '{tool_name}': {error_details}",
            )

        # Execute registered handler
        try:
            # Unpack validated fields into handler
            result = tool.handler(**validated_args.model_dump())
            if not isinstance(result, ToolResult):
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error=f"Tool handler '{tool_name}' did not return a valid ToolResult instance.",
                )
            return result
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Unhandled exception in tool '{tool_name}': {str(e)}",
            )


def create_default_registry() -> ToolRegistry:
    """Build and populate the standard AgentX ToolRegistry."""
    registry = ToolRegistry()

    # 1. Safe Calculator
    registry.register(
        name="calculator",
        description=(
            "Use this tool to evaluate mathematical and arithmetic calculations accurately. "
            "Supports addition (+), subtraction (-), multiplication (*), division (/), integer division (//), "
            "modulo (%), exponentiation (**), and parentheses. Input must be an arithmetic expression string."
        ),
        args_schema=CalculatorArgs,
        handler=safe_calculate,
    )

    # 2. Browser Automation (Read-Only Webpage Content Extraction)
    registry.register(
        name="browser",
        description=(
            "Use this tool to extract page title, main heading, and visible text content from a public HTTP or HTTPS webpage in headless read-only mode. "
            "Use this whenever the user asks for text or information from a specific webpage URL. "
            "Do not use for local or private network addresses."
        ),
        args_schema=BrowserArgs,
        handler=browse_page,
    )

    # 3. Visible Browser Window (Headed Navigation & YouTube)
    registry.register(
        name="open_visible_browser",
        description=(
            "Use this tool when the user explicitly requests to open, launch, or view a visible browser window on their screen "
            "(e.g., 'open a browser to...', 'open youtube in browser', 'search youtube for... in browser'). "
            "Opens a headed browser window to a public URL or YouTube search/video destination. "
            "Does NOT bypass CAPTCHA, authentication, paywalls, or browser security controls. "
            "Never purchases or submits sensitive forms."
        ),
        args_schema=VisibleBrowserArgs,
        handler=open_visible_browser,
    )

    # 4. Current System Time & Timezone
    registry.register(
        name="current_time",
        description=(
            "Use this tool to retrieve the current host computer system time, date, local timezone, and UTC time. "
            "Use this whenever the user asks for the current time, today's date, or local/UTC time comparison."
        ),
        args_schema=CurrentTimeArgs,
        handler=get_current_time,
    )

    # 5. Knowledge Base Search
    registry.register(
        name="knowledge_base_search",
        description=(
            "Use this tool to search internal records, company policies, release dates, and documentation "
            "in the local knowledge base. Returns matched record content and official source IDs (e.g. KB-001). "
            "Use this whenever answering questions regarding internal architecture, policies, or schedules."
        ),
        args_schema=KnowledgeBaseArgs,
        handler=search_knowledge_base,
    )

    # 6. General Web Search (Discovery of Public Sources, Prices, and Real-Time Facts)
    registry.register(
        name="web_search",
        description=(
            "Use this tool to search the live web for real-time information, product/commodity prices (e.g. RAM prices, gold rates, laptop prices), "
            "current news, or unfamiliar topics when no specific URL was provided. "
            "Returns verified source titles, URLs, and text snippets. "
            "Never guess or invent URLs; use this tool to discover them dynamically."
        ),
        args_schema=WebSearchArgs,
        handler=web_search,
    )

    return registry

