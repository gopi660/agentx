"""Unit tests for stateful conversation memory, checkpointing, and session isolation in AgentX."""

import json
from unittest.mock import MagicMock, patch
import pytest
from langgraph.checkpoint.memory import MemorySaver

from agentx.agent import AgentX
from agentx.config import settings
from agentx.schemas import ToolStatus


def _make_choice(content=None, tool_calls=None):
    msg = MagicMock()
    msg.role = "assistant"
    msg.content = content
    msg.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = msg
    return MagicMock(choices=[choice], usage=None)


def test_remember_information_across_consecutive_messages():
    """Verify follow-up questions retain context from earlier turns."""
    mock_client = MagicMock()
    checkpointer = MemorySaver()
    agent = AgentX(groq_client=mock_client, checkpointer=checkpointer)

    # Turn 1: User announces project name
    resp_1 = _make_choice(content="Understood! Your project is AgentX.")
    # Turn 2: User asks follow-up
    resp_2 = _make_choice(content="Your project is called AgentX.")

    mock_client.chat.completions.create.side_effect = [resp_1, resp_2]

    # Turn 1
    r1 = agent.run("My project is AgentX.")
    assert "Understood" in r1.final_answer

    # Turn 2
    r2 = agent.run("What is my project called?")
    assert r2.final_answer == "Your project is called AgentX."

    # Inspect messages sent in turn 2
    assert mock_client.chat.completions.create.call_count == 2
    call2_args = mock_client.chat.completions.create.call_args_list[1]
    call2_messages = call2_args.kwargs.get("messages") or call2_args[1].get("messages")

    # Verify conversation history includes turn 1 user, turn 1 assistant, and turn 2 user
    user_msgs = [m["content"] for m in call2_messages if m["role"] == "user"]
    asst_msgs = [m["content"] for m in call2_messages if m["role"] == "assistant"]

    assert "My project is AgentX." in user_msgs
    assert "What is my project called?" in user_msgs
    assert "Understood! Your project is AgentX." in asst_msgs


def test_separate_thread_ids_isolated_memory():
    """Verify distinct session/thread IDs do not leak or share conversation context."""
    mock_client = MagicMock()
    checkpointer = MemorySaver()
    agent = AgentX(groq_client=mock_client, checkpointer=checkpointer)

    resp_a = _make_choice(content="Noted, secret alpha registered.")
    resp_b = _make_choice(content="I do not have access to your secret.")
    mock_client.chat.completions.create.side_effect = [resp_a, resp_b]

    # Session A sets secret
    agent.run("My secret token is ALPHA_SECRET_999", session_id="session_user_alice")

    # Session B inquires
    agent.run("What is my secret token?", session_id="session_user_bob")

    # Inspect messages sent in Session B
    call_b_args = mock_client.chat.completions.create.call_args_list[1]
    call_b_messages = call_b_args.kwargs.get("messages") or call_b_args[1].get("messages")

    # ALPHA_SECRET_999 MUST NOT be present in session B messages
    b_text = json.dumps(call_b_messages)
    assert "ALPHA_SECRET_999" not in b_text
    assert "What is my secret token?" in b_text


def test_start_new_session_clears_context():
    """Verify starting a new session resets memory so previous facts are not retained."""
    mock_client = MagicMock()
    checkpointer = MemorySaver()
    agent = AgentX(groq_client=mock_client, checkpointer=checkpointer)

    resp_1 = _make_choice(content="Emerald recorded as your favorite color.")
    resp_2 = _make_choice(content="I do not know your favorite color.")
    mock_client.chat.completions.create.side_effect = [resp_1, resp_2]

    # Turn 1 in initial session
    initial_sid = agent.active_session_id
    agent.run("My favorite color is emerald.")

    # Start fresh session
    new_sid = agent.start_new_session()
    assert new_sid != initial_sid

    # Turn 2 in fresh session
    agent.run("What is my favorite color?")

    # Messages in turn 2 should not contain emerald
    call2_args = mock_client.chat.completions.create.call_args_list[1]
    call2_messages = call2_args.kwargs.get("messages") or call2_args[1].get("messages")
    all_content = json.dumps(call2_messages)
    assert "emerald" not in all_content.lower()


def test_clear_session_explicitly_clears_memory():
    """Verify clear_session purges existing thread checkpoint and session state."""
    mock_client = MagicMock()
    checkpointer = MemorySaver()
    agent = AgentX(groq_client=mock_client, checkpointer=checkpointer)

    resp_1 = _make_choice(content="Session initialized.")
    resp_2 = _make_choice(content="Fresh session started.")
    mock_client.chat.completions.create.side_effect = [resp_1, resp_2]

    sid = "test_clear_sid"
    agent.run("Store keyword: OMEGA_VALUE", session_id=sid)

    # Explicitly clear session
    agent.clear_session(session_id=sid)

    # Query again in same session ID
    agent.run("What was the keyword?", session_id=sid)

    call2_args = mock_client.chat.completions.create.call_args_list[1]
    call2_messages = call2_args.kwargs.get("messages") or call2_args[1].get("messages")
    all_content = json.dumps(call2_messages)
    assert "OMEGA_VALUE" not in all_content


def test_tool_calls_and_results_preserved_across_turns():
    """Verify tool calls and tool messages remain intact in conversation history across turns."""
    mock_client = MagicMock()
    checkpointer = MemorySaver()
    agent = AgentX(groq_client=mock_client, checkpointer=checkpointer)

    # Turn 1, Step 1: Model requests calculator
    tc = MagicMock(id="calc_turn1")
    tc.function = MagicMock(name="calculator")
    tc.function.name = "calculator"
    tc.function.arguments = json.dumps({"expression": "25 * 4"})
    step1_resp = _make_choice(content=None, tool_calls=[tc])

    # Turn 1, Step 2: Model completes calculation
    step2_resp = _make_choice(content="The calculation 25 * 4 is 100.")

    # Turn 2, Step 1: User asks follow up referencing that number
    turn2_resp = _make_choice(content="Adding 50 to 100 gives 150.")

    mock_client.chat.completions.create.side_effect = [step1_resp, step2_resp, turn2_resp]

    # Turn 1
    r1 = agent.run("Calculate 25 * 4")
    assert r1.final_answer == "The calculation 25 * 4 is 100."
    assert len(r1.tool_traces) == 1

    # Turn 2
    r2 = agent.run("Now add 50 to that number.")
    assert "150" in r2.final_answer

    # Verify Turn 2's LLM prompt contained the tool call and tool message from Turn 1
    turn2_call = mock_client.chat.completions.create.call_args_list[2]
    turn2_messages = turn2_call.kwargs.get("messages") or turn2_call[1].get("messages")

    roles = [m["role"] for m in turn2_messages]
    assert "tool" in roles
    tool_msg = next(m for m in turn2_messages if m["role"] == "tool")
    assert "100" in tool_msg["content"]
    assert tool_msg["name"] == "calculator"


def test_message_not_duplicated_in_state():
    """Verify current user message is not added multiple times to state in a single turn."""
    mock_client = MagicMock()
    checkpointer = MemorySaver()
    agent = AgentX(groq_client=mock_client, checkpointer=checkpointer)

    mock_client.chat.completions.create.return_value = _make_choice(content="Hello!")

    agent.run("Hi AgentX!")

    call_args = mock_client.chat.completions.create.call_args_list[0]
    messages = call_args.kwargs.get("messages") or call_args[1].get("messages")

    user_msgs = [m for m in messages if m.get("role") == "user" and m.get("content") == "Hi AgentX!"]
    assert len(user_msgs) == 1


def test_checkpointer_failure_fallback_to_memory_saver():
    """Verify AgentX handles SQLite connection or checkpointer failure by falling back to MemorySaver."""
    with patch("sqlite3.connect", side_effect=Exception("Database lock or disk error")):
        with patch.object(settings, "MEMORY_BACKEND", "sqlite"):
            agent = AgentX(groq_client=MagicMock())
            # Should have fallen back gracefully to MemorySaver
            assert isinstance(agent.checkpointer, MemorySaver)
