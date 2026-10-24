"""Unit tests for empty/null LLM response prevention, bounded retries, and deduplication preservation."""

import json
from unittest.mock import MagicMock, patch
import pytest

from agentx.agent import AgentX
from agentx.response_manager import SessionHistoryManager
from agentx.schemas import ToolStatus


def test_empty_response_bounded_retry_recovery():
    """Verify that when a model returns empty content after a tool call, the agent prompts for an informative answer."""
    mock_client = MagicMock()

    # Step 1: Model calls browser tool
    call_mock = MagicMock()
    call_mock.id = "call_browser_1"
    call_mock.function.name = "current_time"
    call_mock.function.arguments = "{}"
    msg1 = MagicMock(content=None, tool_calls=[call_mock])

    # Step 2: Model returns empty content (content=None, tool_calls=None)
    msg2_empty = MagicMock(content=None, tool_calls=None)

    # Step 3: Model receives empty response notice and provides an informative answer
    msg3_recovered = MagicMock(content="The verified local time is 15:30:00 IST.", tool_calls=None)

    mock_client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=msg1)]),
        MagicMock(choices=[MagicMock(message=msg2_empty)]),
        MagicMock(choices=[MagicMock(message=msg3_recovered)]),
    ]

    agent = AgentX(groq_client=mock_client)
    response = agent.run("What time is it?")

    assert response.final_answer == "The verified local time is 15:30:00 IST."
    assert response.error is None


def test_empty_response_exhausted_produces_informative_error_message():
    """Verify that if model persistently returns empty content, a clear fallback error is returned instead of blank text."""
    mock_client = MagicMock()

    # Step 1: Tool call that encounters an error
    call_mock = MagicMock()
    call_mock.id = "call_err_1"
    call_mock.function.name = "calculator"
    call_mock.function.arguments = '{"expression": "10 / 0"}'
    msg1 = MagicMock(content=None, tool_calls=[call_mock])

    # Step 2 & 3: Model returns empty content stubbornly across retries
    msg_empty = MagicMock(content="   \n  ", tool_calls=None)

    mock_client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=msg1)]),
        MagicMock(choices=[MagicMock(message=msg_empty)]),
        MagicMock(choices=[MagicMock(message=msg_empty)]),
    ]

    session_mgr = SessionHistoryManager()
    session_id = "test_empty_session"

    agent = AgentX(groq_client=mock_client, session_hist_manager=session_mgr)
    response = agent.run("Calculate 10 / 0", session_id=session_id)

    # Must NOT be empty string or whitespace
    assert response.final_answer is not None
    assert len(response.final_answer.strip()) > 0
    assert "calculator" in response.final_answer
    # Must indicate error status and detail
    assert response.error is not None
    assert "Empty model response" in response.error

    # CRUCIAL: Failed empty response must NOT pollute delivered session history
    assert session_mgr.get_last_delivered_answer(session_id) is None
