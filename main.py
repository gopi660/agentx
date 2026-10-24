"""AgentX CLI - Command-Line Interface for AgentX.

Provides an interactive terminal session and single-query execution with
concise operational tool traces, model fallback observability, and robust error handling.
"""

import argparse
import sys
import uuid
from typing import Any, Dict

# Ensure Windows terminal outputs UTF-8 cleanly without charmap encoding errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agentx.agent import AgentX
from agentx.config import settings


def format_operational_event(event: Dict[str, Any]) -> str:
    """Format operational tool traces and model fallback events cleanly for the terminal."""
    if event.get("event") == "model_fallback":
        model = event.get("model", "unknown")
        attempts = event.get("attempts", 1)
        return f"  [⟳ MODEL FALLBACK] Primary model unavailable. Successfully switched to: {model} (Attempt {attempts})"

    tool_name = event.get("tool", "unknown_tool")
    args = event.get("arguments", "{}")
    status = event.get("status", "unknown")
    summary = event.get("message", "")

    # Clean display for status
    status_icon = "✓" if status == "success" else ("○" if status == "no_results" else "✗")

    return f"  [{status_icon} TOOL EXECUTION] Tool: {tool_name} | Status: {status.upper()}\n    Args: {args}\n    Result: {summary}"


def run_interactive(session_id: str, agent: AgentX) -> None:
    """Run interactive REPL loop in the terminal."""
    models_chain = settings.get_ordered_models()
    primary = models_chain[0] if models_chain else settings.GROQ_MODEL
    fallbacks = " -> ".join(models_chain[1:]) if len(models_chain) > 1 else "None"

    print("=" * 68)
    print(" AgentX — Tool-Using AI Agent with Browser Automation & Fallback")
    print(f" Session ID      : {session_id}")
    print(f" Primary Model   : {primary}")
    print(f" Fallback Models : {fallbacks}")
    print(" Available Tools : calculator, browser, open_visible_browser, web_search, current_time, knowledge_base_search")
    print(" Commands        : '/new' to start fresh conversation, '/clear' to clear history, 'exit' to quit.")
    print("=" * 68)
    print()

    active_sid = session_id

    while True:
        try:
            user_input = input("AgentX> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting AgentX session. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Session closed. Goodbye!")
            break

        if user_input.lower() in ("/new", "new", "/reset"):
            active_sid = agent.start_new_session()
            print(f"\n[Started new conversation session: {active_sid}]\n")
            continue

        if user_input.lower() in ("/clear", "clear"):
            agent.clear_session(active_sid)
            print(f"\n[Cleared conversation session: {active_sid}]\n")
            continue

        print("\n[AgentX is processing inquiry...]")

        def on_trace(event: Dict[str, Any]) -> None:
            print(format_operational_event(event))

        try:
            response = agent.run(
                user_prompt=user_input,
                session_id=active_sid,
                trace_callback=on_trace,
            )

            print("\n--- FINAL ANSWER ---")
            print(response.final_answer)
            model_info = response.model_used or primary
            if response.fallback_occurred:
                print(f"[Model: {model_info} (Fallback Engaged)]")
            else:
                print(f"[Model: {model_info}]")
            print("--------------------\n")

        except Exception as e:
            print(f"\n[Error executing query]: {str(e)}\n")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AgentX - Tool-Using AI Agent with Browser Automation"
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Run a single inquiry and exit instead of starting interactive REPL",
    )
    parser.add_argument(
        "-s", "--session",
        type=str,
        default=None,
        help="Custom session identifier for response tracking",
    )
    args = parser.parse_args()

    # Verify API key configuration before running
    if not settings.is_api_key_configured():
        print("\n[CONFIGURATION ERROR]")
        print("GROQ_API_KEY is not set or contains placeholder values.")
        print("To run live inquiries:")
        print("  1. Copy .env.example to .env:  cp .env.example .env")
        print("  2. Add your Groq API key:     GROQ_API_KEY=gsk_...")
        print("You can get a free key at: https://console.groq.com/keys\n")
        print("Note: Automated tests (pytest) do NOT require an API key.\n")
        sys.exit(1)

    session_id = args.session or f"session_{uuid.uuid4().hex[:8]}"

    try:
        agent = AgentX()
    except Exception as e:
        print(f"\n[Initialization Failure]: {str(e)}\n")
        sys.exit(1)

    if args.query:
        print(f"[AgentX Session: {session_id}] Processing: {args.query}\n")

        def on_trace(event: Dict[str, Any]) -> None:
            print(format_operational_event(event))

        resp = agent.run(
            user_prompt=args.query,
            session_id=session_id,
            trace_callback=on_trace,
        )
        print("\n--- FINAL ANSWER ---")
        print(resp.final_answer)
        model_info = resp.model_used or settings.GROQ_MODEL
        if resp.fallback_occurred:
            print(f"[Model: {model_info} (Fallback Engaged)]")
        else:
            print(f"[Model: {model_info}]")
        print("--------------------")
    else:
        run_interactive(session_id, agent)


if __name__ == "__main__":
    main()
