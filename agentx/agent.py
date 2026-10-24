import logging
import sqlite3
from typing import Any, Callable, Dict, Optional
import uuid
from groq import Groq

from agentx.config import settings
from agentx.fallback import ModelFallbackManager
from agentx.graph import build_agent_graph
from agentx.response_manager import SessionHistoryManager, session_manager
from agentx.schemas import AgentResponse
from agentx.tool_registry import ToolRegistry, create_default_registry

logger = logging.getLogger("agentx.agent")


class AgentX:
    """AgentX Autonomous Agent instance with stateful conversation memory."""

    def __init__(
        self,
        groq_client: Optional[Groq] = None,
        tool_registry: Optional[ToolRegistry] = None,
        session_hist_manager: Optional[SessionHistoryManager] = None,
        fallback_manager: Optional[ModelFallbackManager] = None,
        trace_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        checkpointer: Optional[Any] = None,
        memory_backend: Optional[str] = None,
    ):
        """Initialize AgentX.

        Args:
            groq_client: Optional Groq client instance. If omitted, initialized
                         lazily using settings.get_groq_api_key().
            tool_registry: Optional custom ToolRegistry (defaults to default registry).
            session_hist_manager: Optional SessionHistoryManager for answer deduplication.
            fallback_manager: Optional ModelFallbackManager for model resilience.
            trace_callback: Optional callback receiving operational tool execution events.
            checkpointer: Optional custom LangGraph checkpointer instance.
            memory_backend: Optional backend override ("sqlite" or "memory").
        """
        self.tool_registry = tool_registry or create_default_registry()
        self.session_manager = session_hist_manager or session_manager
        self.fallback_manager = fallback_manager or ModelFallbackManager()
        self.trace_callback = trace_callback
        self.active_session_id = f"session_{uuid.uuid4().hex[:8]}"
        self._sqlite_conn: Optional[sqlite3.Connection] = None

        if groq_client:
            self.groq_client = groq_client
        else:
            api_key = settings.get_groq_api_key()
            self.groq_client = Groq(api_key=api_key, timeout=settings.GROQ_TIMEOUT_SECONDS)

        # Initialize checkpointer for stateful conversation memory
        if checkpointer is not None:
            self.checkpointer = checkpointer
        else:
            backend = (memory_backend or settings.MEMORY_BACKEND or "sqlite").strip().lower()
            if backend == "sqlite":
                try:
                    from langgraph.checkpoint.sqlite import SqliteSaver
                    db_path = settings.MEMORY_DB_PATH
                    db_path.parent.mkdir(parents=True, exist_ok=True)
                    conn = sqlite3.connect(str(db_path), check_same_thread=False)
                    saver = SqliteSaver(conn)
                    saver.setup()
                    self.checkpointer = saver
                    self._sqlite_conn = conn
                except Exception as e:
                    logger.warning("Failed to initialize SqliteSaver (%s). Falling back to MemorySaver.", e)
                    from langgraph.checkpoint.memory import MemorySaver
                    self.checkpointer = MemorySaver()
                    self._sqlite_conn = None
            else:
                from langgraph.checkpoint.memory import MemorySaver
                self.checkpointer = MemorySaver()
                self._sqlite_conn = None

        # Build the compiled LangGraph workflow with checkpointer attached
        self.app = build_agent_graph(
            groq_client=self.groq_client,
            tool_registry=self.tool_registry,
            session_hist_manager=self.session_manager,
            fallback_manager=self.fallback_manager,
            trace_callback=self.trace_callback,
            checkpointer=self.checkpointer,
        )

    def start_new_session(self) -> str:
        """Start a fresh conversation session with a new unique session identifier.

        Returns:
            Newly generated session identifier.
        """
        self.active_session_id = f"session_{uuid.uuid4().hex[:8]}"
        return self.active_session_id

    def clear_session(self, session_id: Optional[str] = None) -> None:
        """Clear conversation memory and deduplication history for a session.

        Args:
            session_id: Session identifier to clear (defaults to active_session_id).
        """
        sid = session_id or self.active_session_id
        if self.checkpointer:
            if hasattr(self.checkpointer, "delete_thread"):
                try:
                    self.checkpointer.delete_thread(sid)
                except Exception as e:
                    logger.warning("Error deleting thread checkpoint for %s: %s", sid, e)
            if self._sqlite_conn:
                try:
                    with self._sqlite_conn:
                        self._sqlite_conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (sid,))
                        self._sqlite_conn.execute("DELETE FROM writes WHERE thread_id = ?", (sid,))
                except Exception as e:
                    logger.warning("Error clearing sqlite rows for %s: %s", sid, e)

        self.session_manager.clear_session(sid)

    def run(
        self,
        user_prompt: str,
        session_id: Optional[str] = None,
        trace_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> AgentResponse:
        """Run an inquiry through the AgentX agent workflow with stateful conversation memory.

        Args:
            user_prompt: User's question or instruction.
            session_id: Identifier for the current conversation session (reused across turns).
            trace_callback: Optional per-run callback for tool trace events.

        Returns:
            AgentResponse containing the verified final answer, tool traces, model used, and metadata.
        """
        sid = session_id or self.active_session_id

        if not user_prompt or not user_prompt.strip():
            return AgentResponse(
                final_answer="Please provide a non-empty question or instruction.",
                session_id=sid,
                error="Empty user prompt.",
            )

        # Notice: Do not pass messages: [] here so LangGraph checkpointer preserves previous turns!
        initial_state = {
            "session_id": sid,
            "user_prompt": user_prompt.strip(),
            "iteration": 0,
            "pending_tool_calls": [],
            "tool_traces": [],
            "candidate_answer": None,
            "final_answer": None,
            "duplicate_retry_count": 0,
            "empty_retry_count": 0,
            "is_duplicate": False,
            "status": "initialized",
            "error": None,
            "current_model": None,
            "fallback_occurred": False,
            "model_history": [],
            "search_queries": [],
        }

        config = {"configurable": {"thread_id": sid}}

        # If a run-specific trace callback is provided, build a customized graph invocation
        if trace_callback and trace_callback != self.trace_callback:
            custom_app = build_agent_graph(
                groq_client=self.groq_client,
                tool_registry=self.tool_registry,
                session_hist_manager=self.session_manager,
                fallback_manager=self.fallback_manager,
                trace_callback=trace_callback,
                checkpointer=self.checkpointer,
            )
            final_state = custom_app.invoke(initial_state, config=config)
        else:
            final_state = self.app.invoke(initial_state, config=config)

        return AgentResponse(
            final_answer=final_state.get("final_answer") or "No answer was produced.",
            session_id=sid,
            tool_traces=final_state.get("tool_traces", []),
            iterations=final_state.get("iteration", 0),
            duplicate_retries=final_state.get("duplicate_retry_count", 0),
            is_duplicate_handled=final_state.get("is_duplicate", False),
            model_used=final_state.get("current_model"),
            fallback_occurred=final_state.get("fallback_occurred", False),
            error=final_state.get("error"),
        )
