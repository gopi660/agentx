"""Unit tests for tool registry, validation, and Groq schema generation."""

import json
import pytest
from agentx.schemas import ToolResult, ToolStatus
from agentx.tool_registry import ToolRegistry, create_default_registry


def test_default_registry_tools():
    """Verify that default tools are registered with schemas and descriptions."""
    registry = create_default_registry()
    names = registry.list_tool_names()
    assert "calculator" in names
    assert "browser" in names
    assert "open_visible_browser" in names
    assert "current_time" in names
    assert "knowledge_base_search" in names
    assert "web_search" in names

    schemas = registry.get_groq_tools()
    assert len(schemas) == 6


    for s in schemas:
        assert s["type"] == "function"
        fn = s["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"


def test_unknown_tool_rejection():
    """Verify that an unregistered tool name cannot be executed."""
    registry = create_default_registry()
    res = registry.execute("malicious_command_runner", {"cmd": "rm -rf /"})
    assert res.status == ToolStatus.ERROR
    assert "Unrecognized tool" in res.error
    assert "calculator" in res.error


def test_malformed_json_arguments():
    """Verify rejection of unparseable JSON argument strings."""
    registry = create_default_registry()
    res = registry.execute("calculator", "{ this is broken json")
    assert res.status == ToolStatus.INVALID_INPUT
    assert "Malformed JSON" in res.error


def test_invalid_argument_validation():
    """Verify Pydantic argument validation failures return INVALID_INPUT."""
    registry = create_default_registry()

    # Calculator missing 'expression'
    res_calc = registry.execute("calculator", {})
    assert res_calc.status == ToolStatus.INVALID_INPUT
    assert "validation failed" in res_calc.error.lower()

    # Browser with non-http URL
    res_browser = registry.execute("browser", {"url": "file:///etc/shadow"})
    assert res_browser.status == ToolStatus.INVALID_INPUT
    assert "must start with 'http://" in res_browser.error.lower()

    # Knowledge base with empty query
    res_kb = registry.execute("knowledge_base_search", {"query": "   "})
    assert res_kb.status == ToolStatus.INVALID_INPUT
    assert "cannot be empty" in res_kb.error.lower()


def test_successful_tool_execution_via_registry():
    """Verify execution of valid tool calls through registry."""
    registry = create_default_registry()

    # Test calculator via JSON string
    calc_res = registry.execute("calculator", json.dumps({"expression": "40 + 2"}))
    assert calc_res.status == ToolStatus.SUCCESS
    assert calc_res.data == 42

    # Test calculator via dict
    calc_res2 = registry.execute("calculator", {"expression": "100 / 4"})
    assert calc_res2.status == ToolStatus.SUCCESS
    assert calc_res2.data == 25


def test_tool_handler_exception_handling():
    """Verify that unhandled exceptions in custom handlers are safely caught."""
    registry = ToolRegistry()

    from pydantic import BaseModel
    class DummyArgs(BaseModel):
        val: int

    def crashing_handler(val: int):
        raise RuntimeError("Simulated crash")

    registry.register(
        name="crasher",
        description="A tool that crashes",
        args_schema=DummyArgs,
        handler=crashing_handler,
    )

    res = registry.execute("crasher", {"val": 1})
    assert res.status == ToolStatus.ERROR
    assert "Unhandled exception in tool 'crasher'" in res.error
    assert "Simulated crash" in res.error
