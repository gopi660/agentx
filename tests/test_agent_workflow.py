"""Unit tests for LangGraph agent workflow orchestration and tool cycling."""

import json
from unittest.mock import MagicMock
import pytest
from agentx.agent import AgentX
from agentx.config import settings
from agentx.schemas import ToolResult, ToolStatus


def test_direct_answer_without_tools():
    """Verify that questions not requiring tools answer directly."""
    mock_client = MagicMock()
    direct_msg = MagicMock(content="Hello! I am AgentX, ready to assist you.", tool_calls=None)
    mock_client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=direct_msg)])

    agent = AgentX(groq_client=mock_client)
    resp = agent.run("Hello there!")

    assert resp.final_answer == "Hello! I am AgentX, ready to assist you."
    assert len(resp.tool_traces) == 0
    assert resp.iterations == 0


def test_single_calculator_tool_cycle():
    """Verify full loop: user query -> model tool call -> real calculator execution -> final answer."""
    mock_client = MagicMock()

    # Step 1: Model requests calculator tool
    calc_call = MagicMock()
    calc_call.id = "call_calc_01"
    calc_call.function.name = "calculator"
    calc_call.function.arguments = json.dumps({"expression": "125 * 8"})

    step1_msg = MagicMock(content=None, tool_calls=[calc_call])

    # Step 2: Model receives tool result (1000) and formats final answer
    step2_msg = MagicMock(
        content="The calculation 125 * 8 equals 1000.",
        tool_calls=None,
    )

    mock_client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=step1_msg)]),
        MagicMock(choices=[MagicMock(message=step2_msg)]),
    ]

    traces_captured = []
    def capture_trace(trace):
        traces_captured.append(trace)

    agent = AgentX(groq_client=mock_client)
    resp = agent.run("What is 125 times 8?", trace_callback=capture_trace)

    assert resp.final_answer == "The calculation 125 * 8 equals 1000."
    assert len(resp.tool_traces) == 1
    assert resp.tool_traces[0]["tool"] == "calculator"
    assert resp.tool_traces[0]["status"] == ToolStatus.SUCCESS.value
    assert len(traces_captured) == 1
    assert traces_captured[0]["tool"] == "calculator"


def test_knowledge_base_search_flow():
    """Verify knowledge base retrieval and citation loop."""
    mock_client = MagicMock()

    # Step 1: Model requests knowledge base search
    kb_call = MagicMock()
    kb_call.id = "call_kb_01"
    kb_call.function.name = "knowledge_base_search"
    kb_call.function.arguments = json.dumps({"query": "AgentX release schedule"})

    step1_msg = MagicMock(content=None, tool_calls=[kb_call])

    # Step 2: Model receives KB records and cites KB-005
    step2_msg = MagicMock(
        content="According to internal documentation [KB-005], AgentX v2.4 is scheduled for release on November 15, 2026.",
        tool_calls=None,
    )

    mock_client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=step1_msg)]),
        MagicMock(choices=[MagicMock(message=step2_msg)]),
    ]

    agent = AgentX(groq_client=mock_client)
    resp = agent.run("When is AgentX scheduled for release?")

    assert "KB-005" in resp.final_answer
    assert "November 15, 2026" in resp.final_answer
    assert len(resp.tool_traces) == 1
    assert resp.tool_traces[0]["tool"] == "knowledge_base_search"
    assert resp.tool_traces[0]["status"] == ToolStatus.SUCCESS.value


def test_multi_tool_execution_flow():
    """Verify sequential execution of multiple tools across iterations."""
    mock_client = MagicMock()

    # Iteration 1: Model asks KB for release info
    kb_call = MagicMock()
    kb_call.id = "call_kb_multi"
    kb_call.function.name = "knowledge_base_search"
    kb_call.function.arguments = json.dumps({"query": "timeout quotas"})

    step1_msg = MagicMock(content=None, tool_calls=[kb_call])

    # Iteration 2: Model calculates days or seconds
    calc_call = MagicMock()
    calc_call.id = "call_calc_multi"
    calc_call.function.name = "calculator"
    calc_call.function.arguments = json.dumps({"expression": "10 * 60"})

    step2_msg = MagicMock(content=None, tool_calls=[calc_call])

    # Iteration 3: Model final answer
    step3_msg = MagicMock(
        content="Per [KB-006], the timeout quota is 10 seconds, which equals 600 seconds total for a batch of 60.",
        tool_calls=None,
    )

    mock_client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=step1_msg)]),
        MagicMock(choices=[MagicMock(message=step2_msg)]),
        MagicMock(choices=[MagicMock(message=step3_msg)]),
    ]

    agent = AgentX(groq_client=mock_client)
    resp = agent.run("What is the timeout and how many seconds is 10 times 60?")

    assert len(resp.tool_traces) == 2
    assert resp.tool_traces[0]["tool"] == "knowledge_base_search"
    assert resp.tool_traces[1]["tool"] == "calculator"
    assert "KB-006" in resp.final_answer


def test_iteration_limit_enforcement():
    """Verify that an infinite tool loop safely terminates when reaching MAX_TOOL_ITERATIONS."""
    mock_client = MagicMock()

    # Model continuously calls calculator in an infinite loop
    loop_call = MagicMock()
    loop_call.id = "call_infinite_loop"
    loop_call.function.name = "calculator"
    loop_call.function.arguments = json.dumps({"expression": "1 + 1"})

    loop_msg = MagicMock(content=None, tool_calls=[loop_call])
    mock_client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=loop_msg)])

    agent = AgentX(groq_client=mock_client)
    resp = agent.run("Keep calculating forever")

    # Workflow must terminate at MAX_TOOL_ITERATIONS (5) without crashing
    assert resp.iterations >= settings.MAX_TOOL_ITERATIONS
    assert "maximum tool iterations limit" in resp.final_answer.lower()


def test_current_time_agent_workflow():
    """Verify agent invokes current_time tool and produces grounded temporal answer."""
    mock_client = MagicMock()

    time_call = MagicMock()
    time_call.id = "call_time_01"
    time_call.function.name = "current_time"
    time_call.function.arguments = json.dumps({})

    step1_msg = MagicMock(content=None, tool_calls=[time_call])
    step2_msg = MagicMock(
        content="The current local time is 12:30:00 (India Standard Time, UTC+05:30) and UTC is 07:00:00 UTC.",
        tool_calls=None,
    )

    mock_client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=step1_msg)]),
        MagicMock(choices=[MagicMock(message=step2_msg)]),
    ]

    agent = AgentX(groq_client=mock_client)
    resp = agent.run("What is the current time and timezone?")

    assert len(resp.tool_traces) == 1
    assert resp.tool_traces[0]["tool"] == "current_time"
    assert resp.tool_traces[0]["status"] == ToolStatus.SUCCESS.value
    assert "India Standard Time" in resp.final_answer


def test_visible_browser_agent_workflow_and_honesty():
    """Verify agent workflow invokes open_visible_browser and cites honest limitations."""
    from unittest.mock import patch
    mock_client = MagicMock()

    vis_call = MagicMock()
    vis_call.id = "call_vis_01"
    vis_call.function.name = "open_visible_browser"
    vis_call.function.arguments = json.dumps({"url": "https://www.youtube.com", "search_query": "lofi music"})

    step1_msg = MagicMock(content=None, tool_calls=[vis_call])
    step2_msg = MagicMock(
        content="I have opened a visible browser window searching YouTube for 'lofi music'. Note that video playback could not be autonomously verified due to browser autoplay policies.",
        tool_calls=None,
    )

    mock_client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=step1_msg)]),
        MagicMock(choices=[MagicMock(message=step2_msg)]),
    ]

    with patch("agentx.tool_registry.open_visible_browser") as mock_open:
        mock_open.return_value = ToolResult(
            status=ToolStatus.SUCCESS,
            message="Opened visible browser window at YouTube search.",
            data={"url": "https://www.youtube.com/results?search_query=lofi+music", "video_playback_verified": False},
        )

        agent = AgentX(groq_client=mock_client)
        resp = agent.run("Open youtube and search for lofi music in browser")

        assert len(resp.tool_traces) == 1
        assert resp.tool_traces[0]["tool"] == "open_visible_browser"
        assert resp.tool_traces[0]["status"] == ToolStatus.SUCCESS.value
        assert "autoplay policies" in resp.final_answer

