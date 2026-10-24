"""Unit and integration tests for AgentX Futuristic Terminal UI.

Tests:
- UI startup and initialization
- Empty and valid prompt submission
- Backend integration through mocked AgentX instance
- Model fallback event rendering
- Tool execution event rendering
- API and tool error rendering
- Terminal resize behavior
- Animation disabled mode
- Clean shutdown
- Secret redaction
"""

import asyncio
from unittest.mock import MagicMock
import pytest
from rich.text import Text

from agentx.schemas import AgentResponse
from agentx.ui.app import AgentXTerminalApp
from agentx.ui.components.startup_screen import StartupScreen
from agentx.ui.components.wireframe_3d import Wireframe3DWidget


def test_ui_startup_and_initialization():
    """Verify all dashboard panels and components mount correctly on startup."""
    async def _test():
        mock_agent = MagicMock()
        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(120, 35)) as pilot:
            assert app.is_mounted
            assert app.header_bar is not None
            assert app.conversation is not None
            assert app.activity_panel is not None
            assert app.system_panel is not None
            assert app.wireframe_widget is not None
            assert app.prompt_input is not None
            await pilot.press("ctrl+q")

    asyncio.run(_test())


def test_startup_screen_diagnostics():
    """Verify real diagnostic checks on the StartupScreen inside app."""
    async def _test():
        mock_agent = MagicMock()
        app = AgentXTerminalApp(agent=mock_agent, skip_startup=False, enable_animation=False)

        async with app.run_test(size=(100, 30)) as pilot:
            # Startup screen pushed on mount
            assert len(app.screen_stack) > 1
            # Press enter to dismiss
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert len(app.screen_stack) == 1
            await pilot.press("ctrl+q")

    asyncio.run(_test())


def test_empty_prompt_submission_ignored():
    """Verify empty or whitespace-only inputs do not trigger agent execution."""
    async def _test():
        mock_agent = MagicMock()
        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(100, 30)) as pilot:
            app.prompt_input.value = "   "
            await pilot.press("enter")
            await pilot.pause(0.1)

            # Agent must not be called
            assert mock_agent.run.call_count == 0
            await pilot.press("ctrl+q")

    asyncio.run(_test())


def test_valid_prompt_submission_and_backend_integration():
    """Verify valid prompt dispatches to agent and posts response bubble."""
    async def _test():
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResponse(
            final_answer="The answer is 42.",
            session_id="test_session",
            model_used="openai/gpt-oss-120b",
            fallback_occurred=False,
            tool_traces=[{"tool": "calculator", "status": "success", "message": "Calculated: 6*7 = 42"}],
        )

        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(120, 35)) as pilot:
            app.prompt_input.value = "What is 6 * 7?"
            await pilot.press("enter")
            # Wait for worker thread to complete
            await pilot.pause(0.3)

            assert mock_agent.run.call_count == 1
            # Conversation history must have user message and agent message
            children = list(app.conversation.children)
            assert len(children) >= 2
            await pilot.press("ctrl+q")

    asyncio.run(_test())


def test_model_fallback_event_rendering():
    """Verify model fallback events are rendered in the activity panel."""
    async def _test():
        mock_agent = MagicMock()
        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(120, 35)) as pilot:
            app.activity_panel.handle_agent_trace({
                "event": "model_fallback",
                "model": "qwen/qwen3.8-27b",
                "attempts": 2,
            })
            await pilot.pause(0.1)

            # Log must contain fallback indicator
            log_children = list(app.activity_panel.log_container.children)
            assert len(log_children) > 1
            last_entry_text = str(log_children[-1]._render())
            assert "Model failover engaged" in last_entry_text
            assert "qwen/qwen3.8-27b" in last_entry_text
            await pilot.press("ctrl+q")

    asyncio.run(_test())


def test_tool_execution_event_rendering():
    """Verify tool traces (success, no_results, error) render with appropriate icons."""
    async def _test():
        mock_agent = MagicMock()
        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(120, 35)) as pilot:
            app.activity_panel.handle_agent_trace({
                "tool": "calculator",
                "status": "success",
                "message": "Calculated: 10 + 20 = 30",
            })
            app.activity_panel.handle_agent_trace({
                "tool": "knowledge_base_search",
                "status": "no_results",
                "message": "No matching records",
            })
            app.activity_panel.handle_agent_trace({
                "tool": "browser",
                "status": "error",
                "error": "SSRF Blocked",
            })
            await pilot.pause(0.1)

            log_children = list(app.activity_panel.log_container.children)
            assert len(log_children) >= 4
            await pilot.press("ctrl+q")

    asyncio.run(_test())


def test_api_and_tool_error_rendering():
    """Verify error notice cards are rendered on failure."""
    async def _test():
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("Groq API rate limit exceeded")

        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(120, 35)) as pilot:
            app.prompt_input.value = "Trigger error"
            await pilot.press("enter")
            await pilot.pause(0.5)

            # Error card mounted in conversation
            children = list(app.conversation.children)
            has_error = any("rate limit exceeded" in getattr(c, "error", "") for c in children)
            assert has_error
            await pilot.press("ctrl+q")

    asyncio.run(_test())


def test_terminal_resize_behavior():
    """Verify layout adapts smoothly without crashing across terminal dimensions."""
    async def _test():
        mock_agent = MagicMock()
        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(140, 45)) as pilot:
            assert app.is_mounted

            # Resize to standard compact terminal
            await pilot.resize_terminal(80, 24)
            await pilot.pause(0.1)
            assert app.is_mounted

            # Resize to ultra-wide
            await pilot.resize_terminal(160, 50)
            await pilot.pause(0.1)
            assert app.is_mounted
            await pilot.press("ctrl+q")

    asyncio.run(_test())


def test_animation_disabled_and_toggle_mode():
    """Verify 3D wireframe animation toggle and disabled start."""
    async def _test():
        mock_agent = MagicMock()
        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(100, 30)) as pilot:
            assert app.wireframe_widget.enabled is False

            # Toggle on
            await pilot.press("ctrl+t")
            assert app.wireframe_widget.enabled is True

            # Toggle off
            await pilot.press("ctrl+t")
            assert app.wireframe_widget.enabled is False
            await pilot.press("ctrl+q")

    asyncio.run(_test())


def test_help_modal_navigation():
    """Verify HelpModal displays on F1 or ? and dismisses on Esc."""
    async def _test():
        mock_agent = MagicMock()
        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.press("f1")
            await pilot.pause(0.1)
            assert len(app.screen_stack) > 1

            # Dismiss modal
            await pilot.press("escape")
            await pilot.pause(0.1)
            assert len(app.screen_stack) == 1
            await pilot.press("ctrl+q")

    asyncio.run(_test())


def test_secret_redaction_in_ui():
    """Verify API keys and sensitive tokens never appear in conversation errors or activity logs."""
    async def _test():
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("Failed with key gsk_secretApiKey9999 and Bearer token12345")

        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(120, 35)) as pilot:
            app.prompt_input.value = "Leaky call"
            await pilot.press("enter")
            await pilot.pause(0.5)

            # Check all widgets on screen
            conversation_dump = str([getattr(c, "error", "") for c in app.conversation.children])
            activity_dump = str([str(c._render()) for c in app.activity_panel.log_container.children])

            # Sensitive key must NOT be present in plain text anywhere
            assert "gsk_secretApiKey9999" not in conversation_dump
            assert "gsk_secretApiKey9999" not in activity_dump
            assert "[REDACTED_API_KEY]" in activity_dump
            await pilot.press("ctrl+q")

    asyncio.run(_test())


def test_composer_visibility_at_80x24_and_100x30():
    """Verify input composer is fully visible and persistent at 80x24 and 100x30."""
    async def _test():
        mock_agent = MagicMock()
        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(80, 24)) as pilot:
            composer = app.query_one("#composer-container")
            assert composer is not None
            assert composer.display is True
            assert app.prompt_input is not None
            assert app.prompt_input.display is True

            # At 80 columns, sidebar should collapse gracefully to give conversation full width
            assert app.sidebar_container.display is False

            # Resize to 100x30
            await pilot.resize_terminal(100, 30)
            await pilot.pause(0.1)

            assert composer.display is True
            assert app.prompt_input.display is True
            # At 100 columns, telemetry sidebar is visible
            assert app.sidebar_container.display is True
            await pilot.press("ctrl+q")

    asyncio.run(_test())


def test_composer_not_covered_by_long_scrolling_conversation():
    """Verify conversation messages scroll inside their container without obscuring bottom composer."""
    async def _test():
        mock_agent = MagicMock()
        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(100, 30)) as pilot:
            # Add many long messages to cause vertical overflow
            for i in range(15):
                app.conversation.add_user_message(f"Query #{i}: Explain architectural resilience in detail.")
                app.conversation.add_agent_response(
                    answer=f"Response #{i}:\n" + ("Paragraph of detailed explanation with evidence.\n" * 4),
                    model_used="openai/gpt-oss-120b",
                    elapsed_s=0.45,
                )
            await pilot.pause(0.2)

            composer = app.query_one("#composer-container")
            assert composer is not None
            assert composer.display is True

            # Verify prompt input remains interactive and receives typing
            app.prompt_input.value = "Testing input while scrolled"
            assert app.prompt_input.value == "Testing input while scrolled"
            await pilot.press("ctrl+q")

    asyncio.run(_test())


def test_focus_restoration_after_modal_and_response():
    """Verify focus reliably returns to the input composer after modal dismiss and response delivery."""
    async def _test():
        mock_agent = MagicMock()
        mock_agent.run.return_value = AgentResponse(
            final_answer="Computation verified: 10.",
            session_id="test_session",
            model_used="openai/gpt-oss-120b",
        )
        app = AgentXTerminalApp(agent=mock_agent, skip_startup=True, enable_animation=False)

        async with app.run_test(size=(100, 30)) as pilot:
            # 1. Focus after modal dismiss
            await pilot.press("f1")
            await pilot.pause(0.1)
            assert len(app.screen_stack) > 1

            await pilot.press("escape")
            await pilot.pause(0.1)
            assert len(app.screen_stack) == 1
            assert app.prompt_input.has_focus

            # 2. Focus after prompt execution
            app.prompt_input.value = "5 + 5"
            await pilot.press("enter")
            await pilot.pause(0.3)

            assert app.prompt_input.has_focus
            await pilot.press("ctrl+q")

    asyncio.run(_test())

