"""Unit tests for missing-data handling, honest reporting, and anti-hallucination guardrails."""

import json
from unittest.mock import MagicMock
import pytest
from agentx.agent import AgentX
from agentx.schemas import ToolStatus
from agentx.tools.knowledge_base import search_knowledge_base
from agentx.tools.browser import browse_page


def test_knowledge_base_missing_data_status():
    """Verify missing data in knowledge base returns NO_RESULTS status, never fabricated records."""
    res = search_knowledge_base("unrecorded internal policy xyz")
    assert res.status == ToolStatus.NO_RESULTS
    assert res.data == []
    assert res.evidence["matched_ids"] == []
    # Status must NOT be success
    assert res.status != ToolStatus.SUCCESS


def test_browser_missing_or_failed_page_status():
    """Verify failed browser navigation returns ERROR status, not fabricated page summary."""
    # SSRF blocked URL
    res = browse_page("http://127.0.0.1:9090")
    assert res.status == ToolStatus.ERROR
    assert res.data is None
    assert "SSRF" in res.error or "loopback" in res.error.lower() or "blocked" in res.error.lower()


def test_agent_workflow_honest_missing_data_reporting():
    """Verify agent workflow passes NO_RESULTS tool status to model and model reports limitation."""
    mock_client = MagicMock()

    # Step 1: Model requests knowledge base search for non-existent policy
    call_mock = MagicMock()
    call_mock.id = "call_missing_kb"
    call_mock.function.name = "knowledge_base_search"
    call_mock.function.arguments = json.dumps({"query": "quantum entanglement encryption protocol"})

    step1_msg = MagicMock(content=None, tool_calls=[call_mock])

    # Step 2: Model receives tool result (status: no_results) and produces honest response
    step2_msg = MagicMock(
        content=(
            "Based on a search of the internal knowledge base, no records exist regarding a "
            "'quantum entanglement encryption protocol' (status: no_results). I cannot confirm or provide "
            "details on this topic as evidence is unavailable."
        ),
        tool_calls=None,
    )

    mock_client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=step1_msg)]),
        MagicMock(choices=[MagicMock(message=step2_msg)]),
    ]

    agent = AgentX(groq_client=mock_client)
    response = agent.run("What is our quantum encryption protocol?")

    # Verify tool execution trace shows NO_RESULTS status
    assert len(response.tool_traces) == 1
    assert response.tool_traces[0]["tool"] == "knowledge_base_search"
    assert response.tool_traces[0]["status"] == ToolStatus.NO_RESULTS.value

    # Verify model's final answer admits missing information
    assert "no records exist" in response.final_answer.lower()
    assert "cannot confirm" in response.final_answer.lower()


def test_agent_workflow_honest_browser_failure_reporting():
    """Verify agent workflow passes browser ERROR status to model and model admits failure."""
    mock_client = MagicMock()

    # Step 1: Model requests browsing a blocked/private URL
    call_mock = MagicMock()
    call_mock.id = "call_blocked_url"
    call_mock.function.name = "browser"
    call_mock.function.arguments = json.dumps({"url": "http://192.168.1.1/admin"})

    step1_msg = MagicMock(content=None, tool_calls=[call_mock])

    # Step 2: Model receives SSRF error and explains failure honestly
    step2_msg = MagicMock(
        content=(
            "I attempted to access http://192.168.1.1/admin, but the request was blocked by "
            "security policy because 192.168.1.1 is a private IP address. Therefore, I cannot "
            "retrieve or verify any content from that location."
        ),
        tool_calls=None,
    )

    mock_client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=step1_msg)]),
        MagicMock(choices=[MagicMock(message=step2_msg)]),
    ]

    agent = AgentX(groq_client=mock_client)
    response = agent.run("What is on http://192.168.1.1/admin?")

    assert len(response.tool_traces) == 1
    assert response.tool_traces[0]["status"] == ToolStatus.ERROR.value
    assert "blocked by security policy" in response.final_answer.lower()
