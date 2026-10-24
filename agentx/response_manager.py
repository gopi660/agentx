"""Session response history and consecutive duplicate prevention manager.

Prevents identical consecutive responses within a user session by normalizing
formatting differences, prompting the LLM for fact-preserving rephrasings,
and falling back cleanly when retries are exhausted.
"""

from collections import deque
import re
from typing import Deque, Dict, Optional
from agentx.config import settings


def normalize_answer_text(text: str) -> str:
    """Normalize text by collapsing whitespace, stripping punctuation and markdown.

    This ensures that trivial differences (such as punctuation or capitalization)
    do not bypass consecutive duplicate detection.
    """
    if not text:
        return ""

    # Convert to lowercase
    normalized = text.lower()

    # Remove markdown formatting characters (*, _, `, #, >, -)
    normalized = re.sub(r"[\*_`#>~]", " ", normalized)

    # Remove standard punctuation
    normalized = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_~()?'\"]", " ", normalized)

    # Collapse multiple whitespace characters and newlines into a single space
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


class SessionHistoryManager:
    """Manages delivered answer history per session in memory."""

    def __init__(self, max_history_per_session: int = 10):
        self._max_history = max_history_per_session
        self._session_histories: Dict[str, Deque[str]] = {}

    def get_last_delivered_answer(self, session_id: str) -> Optional[str]:
        """Retrieve the last successfully delivered final answer for a session."""
        history = self._session_histories.get(session_id)
        if history and len(history) > 0:
            return history[-1]
        return None

    def is_consecutive_duplicate(self, session_id: str, candidate_answer: str) -> bool:
        """Check whether candidate_answer is identical to the last delivered answer in this session."""
        last_answer = self.get_last_delivered_answer(session_id)
        if not last_answer:
            return False

        norm_candidate = normalize_answer_text(candidate_answer)
        norm_last = normalize_answer_text(last_answer)

        return norm_candidate == norm_last

    def record_delivered_answer(self, session_id: str, answer: str) -> None:
        """Record a successfully delivered answer into the session history.

        Note: Failed requests or unconfirmed answers must NOT call this method.
        """
        if session_id not in self._session_histories:
            self._session_histories[session_id] = deque(maxlen=self._max_history)
        self._session_histories[session_id].append(answer)

    def clear_session(self, session_id: str) -> None:
        """Clear history for a specific session."""
        if session_id in self._session_histories:
            del self._session_histories[session_id]

    def clear_all(self) -> None:
        """Clear all session histories (e.g. between tests or on system reset)."""
        self._session_histories.clear()

    @staticmethod
    def build_rephrase_instruction(previous_answer: str) -> str:
        """Construct the prompt instructing the model to rephrase without altering facts."""
        return (
            "[SYSTEM NOTICE: DUPLICATE RESPONSE DETECTED]\n"
            "Your previous final answer in this active session was identical to this candidate answer.\n"
            "Please provide a meaningfully rephrased response using different sentence structure and wording, "
            "while STRICTLY PRESERVING all verified numbers, calculated values, source citations, and stated uncertainties.\n"
            "CRITICAL: Do NOT invent new facts or alter any numeric results."
        )

    @staticmethod
    def apply_duplicate_fallback(candidate_answer: str) -> str:
        """Produce a safe, documented fallback indicating confirmed duplicate response.

        Preserves truth and factuality without hallucinating differences.
        """
        prefix = "[Note: Reconfirming previous response - identical inquiry]\n"
        if candidate_answer.startswith(prefix):
            return candidate_answer
        return f"{prefix}{candidate_answer}"


# Global session history manager singleton
session_manager = SessionHistoryManager()
