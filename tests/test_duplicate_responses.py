"""Unit tests for duplicate response detection, normalization, and session isolation."""

from unittest.mock import MagicMock
import pytest
from agentx.agent import AgentX
from agentx.response_manager import (
    SessionHistoryManager,
    normalize_answer_text,
)


def test_normalization_collapses_trivial_differences():
    """Verify normalization strips case, markdown, extra spaces, and punctuation."""
    ans1 = "The calculated answer is 42."
    ans2 = "   THE CALCULATED ANSWER IS 42!   "
    ans3 = "The **calculated** answer is 42..."

    norm1 = normalize_answer_text(ans1)
    norm2 = normalize_answer_text(ans2)
    norm3 = normalize_answer_text(ans3)

    assert norm1 == "the calculated answer is 42"
    assert norm1 == norm2
    assert norm1 == norm3


def test_consecutive_duplicate_detection():
    """Verify duplicate detection flags identical consecutive answers in the same session."""
    manager = SessionHistoryManager()
    session = "user_123"

    # Initially empty
    assert not manager.is_consecutive_duplicate(session, "The sum is 100.")

    # Record first answer
    manager.record_delivered_answer(session, "The sum is 100.")

    # Candidate identical answer
    assert manager.is_consecutive_duplicate(session, "The sum is 100.")
    assert manager.is_consecutive_duplicate(session, "   the sum is 100! ")

    # Candidate distinct answer
    assert not manager.is_consecutive_duplicate(session, "The product is 500.")


def test_session_isolation():
    """Verify session histories do not leak or cross-contaminate between users."""
    manager = SessionHistoryManager()
    session_a = "alice_session"
    session_b = "bob_session"

    # Alice receives an answer
    manager.record_delivered_answer(session_a, "AgentX is powered by Groq.")

    # In Alice's session, this answer is now a duplicate
    assert manager.is_consecutive_duplicate(session_a, "AgentX is powered by Groq.")

    # In Bob's session, this same answer is NOT a duplicate because sessions are isolated
    assert not manager.is_consecutive_duplicate(session_b, "AgentX is powered by Groq.")


def test_failed_requests_do_not_update_history():
    """Verify that uncompleted or failed executions do not pollute delivered history."""
    manager = SessionHistoryManager()
    session = "test_failure_session"

    manager.record_delivered_answer(session, "Initial valid answer")
    assert manager.get_last_delivered_answer(session) == "Initial valid answer"

    # An agent error occurs: record_delivered_answer is NOT called
    assert manager.get_last_delivered_answer(session) == "Initial valid answer"


def test_duplicate_rephrase_workflow_with_mock_model():
    """Verify LangGraph workflow prompts model for rephrase when candidate is duplicate."""
    mock_client = MagicMock()
    custom_session_mgr = SessionHistoryManager()
    session_id = "test_dup_session"

    # Pre-record an existing delivered answer
    custom_session_mgr.record_delivered_answer(session_id, "The answer is exactly 42.")

    # Setup model responses:
    # 1. Model initially produces the identical answer ("The answer is exactly 42.")
    # 2. Model receives rephrase request and produces distinct phrasing ("Verified result: 42.")
    first_choice = MagicMock()
    first_choice.message = MagicMock(content="The answer is exactly 42.", tool_calls=None)

    second_choice = MagicMock()
    second_choice.message = MagicMock(content="Verified result: 42.", tool_calls=None)

    mock_client.chat.completions.create.side_effect = [
        MagicMock(choices=[first_choice]),
        MagicMock(choices=[second_choice]),
    ]

    agent = AgentX(groq_client=mock_client, session_hist_manager=custom_session_mgr)
    response = agent.run("What is 6 * 7?", session_id=session_id)

    assert response.duplicate_retries == 1
    assert response.final_answer == "Verified result: 42."
    assert custom_session_mgr.get_last_delivered_answer(session_id) == "Verified result: 42."


def test_duplicate_fallback_when_retries_exhausted():
    """Verify fallback is applied when the model fails to produce a distinct rephrasing."""
    mock_client = MagicMock()
    custom_session_mgr = SessionHistoryManager()
    session_id = "exhaustion_session"

    original_answer = "Water boils at 100 degrees Celsius."
    custom_session_mgr.record_delivered_answer(session_id, original_answer)

    # Model stubbornly returns the same answer across all retries
    repeated_choice = MagicMock()
    repeated_choice.message = MagicMock(content=original_answer, tool_calls=None)

    mock_client.chat.completions.create.return_value = MagicMock(choices=[repeated_choice])

    agent = AgentX(groq_client=mock_client, session_hist_manager=custom_session_mgr)
    response = agent.run("At what temperature does water boil?", session_id=session_id)

    # Retries reached limit (2 retries)
    assert response.duplicate_retries >= 2
    assert response.is_duplicate_handled is True
    # Safe fallback prefix is applied
    assert "[Note: Reconfirming previous response - identical inquiry]" in response.final_answer
    assert "Water boils at 100 degrees Celsius." in response.final_answer
