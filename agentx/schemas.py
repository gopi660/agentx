"""Data schemas and Pydantic models for AgentX.

Defines tool execution statuses, tool results, argument validation models,
and the LangGraph AgentState TypedDict.
"""

from enum import Enum
import json
from typing import Any, Dict, List, Optional, TypedDict
from pydantic import BaseModel, Field, field_validator


class ToolStatus(str, Enum):
    """Allowed status values for tool executions."""

    SUCCESS = "success"
    NO_RESULTS = "no_results"
    INVALID_INPUT = "invalid_input"
    ERROR = "error"


class ToolResult(BaseModel):
    """Structured representation of tool execution output."""

    status: ToolStatus = Field(description="Execution outcome status")
    data: Optional[Any] = Field(default=None, description="Extracted payload or computed result")
    message: Optional[str] = Field(default=None, description="Human/LLM readable summary")
    error: Optional[str] = Field(default=None, description="Error detail if failed")
    evidence: Optional[Dict[str, Any]] = Field(
        default=None, description="Structured factual evidence cited by the model"
    )

    def to_tool_message_content(self) -> str:
        """Serialize into a concise JSON string for returning to the LLM."""
        return json.dumps(self.model_dump(), ensure_ascii=False)


# --- Tool Argument Models ---

class CalculatorArgs(BaseModel):
    """Arguments for the safe calculator tool."""

    expression: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Mathematical expression to evaluate (e.g. '(120 * 45) / 2')",
    )

    @field_validator("expression")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Expression cannot be empty or whitespace only.")
        return cleaned


class BrowserArgs(BaseModel):
    """Arguments for the Playwright browser automation tool."""

    url: str = Field(
        ...,
        min_length=3,
        description="Public HTTP or HTTPS URL to visit (e.g. 'https://example.com' or 'example.com')",
    )

    @field_validator("url")
    @classmethod
    def validate_url_syntax(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("URL cannot be empty or whitespace only.")
        if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
            if "://" not in cleaned and not cleaned.lower().startswith(
                ("javascript:", "data:", "about:", "file:", "ftp:", "mailto:", "blob:", "gopher:", "ldap:")
            ):
                cleaned = f"https://{cleaned}"
            else:
                raise ValueError("URL must start with 'http://' or 'https://'")
        return cleaned


class KnowledgeBaseArgs(BaseModel):
    """Arguments for the local knowledge base search tool."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Search query string or keywords to find relevant records",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Search query cannot be empty or whitespace only.")
        return cleaned


class VisibleBrowserArgs(BaseModel):
    """Arguments for the visible headed Google Chrome browser tool."""

    url: str = Field(
        ...,
        min_length=3,
        description="Public HTTP or HTTPS URL to open in a visible browser window (e.g. 'https://www.youtube.com' or 'youtube.com')",
    )
    search_query: Optional[str] = Field(
        default=None,
        description="Optional search query if requesting to search YouTube or another site",
    )
    keep_open: bool = Field(
        default=True,
        description="Whether to keep the browser window open after navigation for user interaction (default: True)",
    )

    @field_validator("url")
    @classmethod
    def validate_url_syntax(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("URL cannot be empty or whitespace only.")
        if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
            if "://" not in cleaned and not cleaned.lower().startswith(
                ("javascript:", "data:", "about:", "file:", "ftp:", "mailto:", "blob:", "gopher:", "ldap:")
            ):
                cleaned = f"https://{cleaned}"
            else:
                raise ValueError("URL must start with 'http://' or 'https://'")
        return cleaned


class WebSearchArgs(BaseModel):
    """Arguments for the web search discovery tool."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=400,
        description="Natural language search query to discover relevant web sources, current commodity/product prices, and real-time facts",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of search results to return (default: 5)",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Search query cannot be empty or whitespace only.")
        return cleaned


class CurrentTimeArgs(BaseModel):
    """Arguments for the current system time tool."""

    timezone_name: Optional[str] = Field(
        default=None,
        description="Optional timezone query or label if inquiring about a specific timezone offset",
    )


# --- LangGraph Workflow State ---

class AgentState(TypedDict):
    """State schema tracked across LangGraph nodes."""

    session_id: str
    user_prompt: str
    messages: List[Dict[str, Any]]
    iteration: int
    pending_tool_calls: List[Dict[str, Any]]
    tool_traces: List[Dict[str, Any]]
    candidate_answer: Optional[str]
    final_answer: Optional[str]
    duplicate_retry_count: int
    empty_retry_count: int
    is_duplicate: bool
    status: str  # "running", "awaiting_tool", "completed", "error", "exhausted"
    error: Optional[str]
    current_model: Optional[str]
    fallback_occurred: bool
    model_history: List[Dict[str, Any]]
    search_queries: List[str]


# --- Final Agent Response ---

class AgentResponse(BaseModel):
    """Final output object returned by AgentX to callers/CLI."""

    final_answer: str
    session_id: str
    tool_traces: List[Dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    duplicate_retries: int = 0
    is_duplicate_handled: bool = False
    model_used: Optional[str] = None
    fallback_occurred: bool = False
    error: Optional[str] = None
