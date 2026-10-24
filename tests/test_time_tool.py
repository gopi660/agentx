"""Unit tests for the current system time and timezone tool."""

from datetime import datetime, timezone
from unittest.mock import patch
import pytest

from agentx.schemas import ToolStatus
from agentx.tool_registry import create_default_registry
from agentx.tools.time_tool import get_current_time


def test_get_current_time_structure_and_distinction():
    """Verify get_current_time returns local time, UTC time, and clearly distinguishes timezones."""
    res = get_current_time()

    assert res.status == ToolStatus.SUCCESS
    assert res.data is not None
    data = res.data

    # Check required fields
    assert "local_time" in data
    assert "local_iso" in data
    assert "local_timezone" in data
    assert "local_utc_offset" in data
    assert "utc_time" in data
    assert "utc_iso" in data
    assert "day_of_week" in data
    assert "epoch_timestamp" in data

    # Verify UTC timestamp ends with +00:00 or Z
    assert data["utc_iso"].endswith("+00:00") or data["utc_iso"].endswith("Z")

    # Verify human-readable message clearly distinguishes local timezone from UTC
    assert "Current Local Time:" in res.message
    assert "UTC Time:" in res.message
    assert "UTC" in res.message
    assert data["local_timezone"] in res.message

    # Evidence grounding
    assert res.evidence["source"] == "system_clock"
    assert res.evidence["epoch_timestamp"] > 0


def test_get_current_time_registry_execution():
    """Verify tool execution through ToolRegistry with empty or dict args."""
    registry = create_default_registry()
    tool = registry.get_tool("current_time")
    assert tool is not None

    # Execute with empty dict
    res1 = registry.execute("current_time", {})
    assert res1.status == ToolStatus.SUCCESS

    # Execute with JSON string
    res2 = registry.execute("current_time", '{"timezone_name": "local"}')
    assert res2.status == ToolStatus.SUCCESS


def test_get_current_time_error_handling():
    """Verify graceful handling if datetime calls fail unexpectedly."""
    with patch("agentx.tools.time_tool.datetime") as mock_dt:
        mock_dt.now.side_effect = RuntimeError("Clock hardware malfunction")
        res = get_current_time()

        assert res.status == ToolStatus.ERROR
        assert "Clock hardware malfunction" in res.error
