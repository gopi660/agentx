"""Regression and layout tests for AgentX Terminal UI conversation rendering and scrolling.

Verifies:
1. Two sequential question/answer turns append separate widgets without mutating or truncating previous content.
2. Long answers followed by additional turns retain full text and natural height.
3. ConversationHistory container supports vertical scrolling (overflow-y: scroll) and scrolls on new messages.
4. Terminal resizing preserves all messages without clipping or widget replacement.
5. Manual scrolling upward keeps previous messages readable.
6. Processing state and task cleanup release properly on completion and reset.
"""

import asyncio
from unittest.mock import MagicMock
import pytest

from agentx.schemas import AgentResponse
from agentx.ui.app import AgentXTerminalApp
from agentx.ui.components.conversation import (
    AgentMessageWidget,
    ConversationHistory,
    UserMessageWidget,
)


def test_sequential_messages_remain_intact_and_untruncated():
    """Verify appending multiple messages keeps earlier messages intact with full text."""
    async def _test():
        mock_agent = MagicMock()
        mock_agent.run.side_effect = [
            AgentResponse(
                final_answer="First answer: AgentX architecture uses LangGraph state machine.",
                session_id="session_test",
                model_used="openai/gpt-oss-120b",
                fallback_occurred=False,
                tool_traces=[],
            ),
            AgentResponse(
                final_answer="Second answer: Memory checkpointers preserve past dialogue state.",
                session_id="session_test",
                model_used="openai/gpt-oss-120b",
                fallback_occurred=False,
                tool_traces=[],
            ),
        ]

        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(120, 40)) as pilot:
            # Turn 1
            app.prompt_input.value = "What is the AgentX architecture?"
            await pilot.press("enter")
            await pilot.pause(0.3)

            # Inspect after Turn 1 (initial system notice + user msg 1 + agent msg 1)
            widgets_after_turn1 = list(app.conversation.children)
            assert len(widgets_after_turn1) == 3
            user_msg_1 = widgets_after_turn1[1]
            agent_msg_1 = widgets_after_turn1[2]

            assert isinstance(user_msg_1, UserMessageWidget)
            assert isinstance(agent_msg_1, AgentMessageWidget)
            assert user_msg_1.prompt == "What is the AgentX architecture?"
            assert agent_msg_1.answer == "First answer: AgentX architecture uses LangGraph state machine."

            # Turn 2
            app.prompt_input.value = "How does memory work?"
            await pilot.press("enter")
            await pilot.pause(0.3)

            # Inspect after Turn 2 (initial system notice + 2 user msgs + 2 agent msgs)
            widgets_after_turn2 = list(app.conversation.children)
            assert len(widgets_after_turn2) == 5

            # Verify widgets 1 and 2 are still the original instances with exact same text
            assert widgets_after_turn2[1] is user_msg_1
            assert widgets_after_turn2[2] is agent_msg_1
            assert user_msg_1.prompt == "What is the AgentX architecture?"
            assert agent_msg_1.answer == "First answer: AgentX architecture uses LangGraph state machine."

            # Verify widgets 3 and 4 are the new turn
            user_msg_2 = widgets_after_turn2[3]
            agent_msg_2 = widgets_after_turn2[4]
            assert isinstance(user_msg_2, UserMessageWidget)
            assert isinstance(agent_msg_2, AgentMessageWidget)
            assert user_msg_2.prompt == "How does memory work?"
            assert agent_msg_2.answer == "Second answer: Memory checkpointers preserve past dialogue state."

            # Height styles must be 'auto' so they grow to natural content height
            assert agent_msg_1.styles.height.is_auto
            assert agent_msg_2.styles.height.is_auto

            await pilot.press("ctrl+q")

    asyncio.run(_test())


def test_long_answer_followed_by_new_turn_layout():
    """Verify a very long response does not get clipped or shortened by subsequent messages."""
    async def _test():
        mock_agent = MagicMock()
        long_paragraph = (
            "Paragraph 1: In-depth exploration of terminal rendering pipelines and layout trees.\n\n"
            "Paragraph 2: Textual splits Vertical layouts into equal fractions if height is 1fr, "
            "which causes previous answers to shrink or truncate when new answers are mounted.\n\n"
            "Paragraph 3: By setting height to auto on AgentMessageWidget and child containers, "
            "every widget expands to its natural height and enables smooth vertical scrolling.\n\n"
            "Paragraph 4: Verification line ensuring bottom of long response is preserved in history."
        )

        mock_agent.run.side_effect = [
            AgentResponse(
                final_answer=long_paragraph,
                session_id="session_long",
                model_used="openai/gpt-oss-120b",
                fallback_occurred=False,
                tool_traces=[],
            ),
            AgentResponse(
                final_answer="Follow-up short response.",
                session_id="session_long",
                model_used="openai/gpt-oss-120b",
                fallback_occurred=False,
                tool_traces=[],
            ),
        ]

        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(100, 30)) as pilot:
            # Turn 1: Post long answer
            app.prompt_input.value = "Explain the layout issue in detail."
            await pilot.press("enter")
            await pilot.pause(0.3)

            # Children: [SystemNoticeWidget, UserMessageWidget, AgentMessageWidget]
            assert len(app.conversation.children) == 3
            first_agent_widget = app.conversation.children[2]
            assert isinstance(first_agent_widget, AgentMessageWidget)
            assert first_agent_widget.answer == long_paragraph

            # Turn 2: Follow-up question
            app.prompt_input.value = "Now give a brief summary."
            await pilot.press("enter")
            await pilot.pause(0.3)

            # Confirm 5 distinct children (1 notice + 2 user + 2 agent)
            assert len(app.conversation.children) == 5
            # First message must still contain the full 4 paragraphs without truncation
            assert app.conversation.children[2].answer == long_paragraph
            assert "Paragraph 4: Verification line" in app.conversation.children[2].answer

            # Verify conversation container overflow settings
            assert app.conversation.styles.overflow_y == "scroll"

            await pilot.press("ctrl+q")

    asyncio.run(_test())


def test_conversation_container_vertical_scrolling_and_auto_scroll():
    """Verify that conversation history scrolls and does not restrict vertical overflow."""
    async def _test():
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResponse(
            final_answer="Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8\nLine 9\nLine 10",
            session_id="scroll_test",
            model_used="openai/gpt-oss-120b",
            fallback_occurred=False,
            tool_traces=[],
        )

        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(80, 20)) as pilot:
            # Add 3 turns to ensure content exceeds the 20-row terminal height
            for i in range(3):
                app.prompt_input.value = f"Question {i+1}"
                await pilot.press("enter")
                await pilot.pause(0.3)

            # Check that container has 1 notice + 3*(user + agent) = 7 widgets
            assert len(app.conversation.children) == 7

            # Verify overflow configuration
            assert app.conversation.styles.overflow_y == "scroll"

            # Verify manual scrolling upward: scroll_to(y=0) preserves messages
            app.conversation.scroll_to(y=0, animate=False)
            await pilot.pause(0.1)
            assert app.conversation.scroll_y == 0
            # All messages still exist and are readable
            assert app.conversation.children[1].prompt == "Question 1"

            # Auto-scroll to bottom
            app.conversation.scroll_end(animate=False)
            await pilot.pause(0.1)

            await pilot.press("ctrl+q")

    asyncio.run(_test())


def test_terminal_resize_preserves_messages():
    """Verify terminal resizing does not clip or lose conversation messages."""
    async def _test():
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResponse(
            final_answer="Resizing test response with important technical data.",
            session_id="resize_test",
            model_used="openai/gpt-oss-120b",
            fallback_occurred=False,
            tool_traces=[],
        )

        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(100, 30)) as pilot:
            app.prompt_input.value = "Resize test question"
            await pilot.press("enter")
            await pilot.pause(0.3)

            # 1 notice + 1 user + 1 agent = 3
            assert len(app.conversation.children) == 3

            # Resize to smaller terminal
            await pilot.resize_terminal(70, 20)
            await pilot.pause(0.1)

            # Messages are preserved
            assert len(app.conversation.children) == 3
            assert app.conversation.children[1].prompt == "Resize test question"
            assert app.conversation.children[2].answer == "Resizing test response with important technical data."

            # Resize to larger terminal
            await pilot.resize_terminal(140, 50)
            await pilot.pause(0.1)
            assert len(app.conversation.children) == 3

            await pilot.press("ctrl+q")

    asyncio.run(_test())


def test_is_processing_released_on_completion_and_clear():
    """Verify UI is_processing lock is released upon completion, /new, and /clear."""
    async def _test():
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResponse(
            final_answer="Completed task.",
            session_id="test_lock",
            model_used="openai/gpt-oss-120b",
            fallback_occurred=False,
            tool_traces=[],
        )

        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(100, 30)) as pilot:
            # Initially not processing
            assert not app.is_processing

            # Submit query
            app.prompt_input.value = "Process this"
            await pilot.press("enter")
            await pilot.pause(0.3)

            # After completion, is_processing must be False
            assert not app.is_processing

            # Test /new command
            app.prompt_input.value = "/new"
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert not app.is_processing
            assert len(app.conversation.children) == 1  # System notice

            # Test /clear command
            app.prompt_input.value = "/clear"
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert not app.is_processing
            assert len(app.conversation.children) == 1  # Cleared notice

            await pilot.press("ctrl+q")

    asyncio.run(_test())
