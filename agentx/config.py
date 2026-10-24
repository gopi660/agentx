"""Configuration module for AgentX.

Loads environment variables, handles application settings, parses primary and
fallback model chains, and provides safe access to credentials without exposing secrets.
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load .env file from project root if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings:
    """AgentX runtime configuration settings."""

    # Model configuration
    # Verified available Groq models in prioritized fallback order
    DEFAULT_MODEL: str = "openai/gpt-oss-120b"
    DEFAULT_FALLBACK_MODELS: str = "qwen/qwen3.8-27b,openai/gpt-oss-20b"

    GROQ_MODEL: str = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    GROQ_FALLBACK_MODELS: str = os.getenv("GROQ_FALLBACK_MODELS", DEFAULT_FALLBACK_MODELS)

    # API key (retrieved safely via get_groq_api_key)
    _GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")

    # LangGraph & Loop limits
    MAX_TOOL_ITERATIONS: int = int(os.getenv("MAX_TOOL_ITERATIONS", "5"))
    MAX_DUPLICATE_RETRIES: int = int(os.getenv("MAX_DUPLICATE_RETRIES", "2"))

    # Model Fallback & Retry limits
    MAX_PER_MODEL_RETRIES: int = 1
    MAX_RATE_LIMIT_WAIT_SECONDS: float = 5.0
    GROQ_TIMEOUT_SECONDS: float = float(os.getenv("GROQ_TIMEOUT_SECONDS", "25.0"))

    # Tool limits
    CALCULATOR_MAX_EXPR_LENGTH: int = 200
    CALCULATOR_MAX_MAGNITUDE: float = 1e15
    CALCULATOR_MAX_EXPONENT: int = 100

    # Browser settings
    CHROME_CHANNEL: str = os.getenv("CHROME_CHANNEL", "chrome")
    BROWSER_TIMEOUT_MS: int = int(os.getenv("BROWSER_TIMEOUT_MS", "15000"))  # 15 seconds for headless
    BROWSER_VISIBLE_TIMEOUT_MS: int = int(os.getenv("BROWSER_VISIBLE_TIMEOUT_MS", "30000"))  # 30 seconds for visible Chrome
    BROWSER_MAX_TEXT_LENGTH: int = 4000  # Max extracted characters
    DNS_CACHE_TTL_SECONDS: float = float(os.getenv("DNS_CACHE_TTL_SECONDS", "60.0"))
    DNS_CACHE_MAX_ENTRIES: int = int(os.getenv("DNS_CACHE_MAX_ENTRIES", "500"))

    # Web search settings
    SEARCH_PROVIDER: str = os.getenv("SEARCH_PROVIDER", "duckduckgo")
    TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
    SERPAPI_API_KEY: Optional[str] = os.getenv("SERPAPI_API_KEY")
    SEARCH_MAX_RESULTS: int = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
    SEARCH_TIMEOUT_SECONDS: float = float(os.getenv("SEARCH_TIMEOUT_SECONDS", "10.0"))
    MAX_SEARCH_ATTEMPTS_PER_REQUEST: int = int(os.getenv("MAX_SEARCH_ATTEMPTS_PER_REQUEST", "3"))
    CURRENT_YEAR: int = int(os.getenv("CURRENT_YEAR", "2026"))

    # Stateful Conversation Memory settings
    MEMORY_BACKEND: str = os.getenv("MEMORY_BACKEND", "sqlite")
    MEMORY_DB_PATH: Path = PROJECT_ROOT / "data" / "conversations.db"
    MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "20"))

    # Knowledge base location
    KNOWLEDGE_BASE_PATH: Path = PROJECT_ROOT / "data" / "knowledge_base.json"

    @classmethod
    def get_ordered_models(cls) -> List[str]:
        """Construct an ordered, deduplicated list of models from configuration.

        Order:
        1. Primary GROQ_MODEL
        2. Fallback models from GROQ_FALLBACK_MODELS (comma-delimited)

        Returns:
            List of unique model IDs maintaining order.
        """
        primary = (os.getenv("GROQ_MODEL") or cls.GROQ_MODEL).strip()
        fallback_raw = os.getenv("GROQ_FALLBACK_MODELS", cls.GROQ_FALLBACK_MODELS)

        ordered: List[str] = []
        if primary:
            ordered.append(primary)

        if fallback_raw:
            for piece in fallback_raw.split(","):
                clean = piece.strip()
                if clean and clean not in ordered:
                    ordered.append(clean)

        return ordered

    @classmethod
    def get_groq_api_key(cls) -> str:
        """Retrieve the Groq API key, verifying it is configured.

        Raises:
            ValueError: If the API key is missing or set to a placeholder.
        """
        key = os.getenv("GROQ_API_KEY") or cls._GROQ_API_KEY
        if not key or key.strip() == "" or "your_groq_api_key_here" in key:
            raise ValueError(
                "GROQ_API_KEY is not configured. Please set GROQ_API_KEY in your .env "
                "file or system environment variables. You can obtain a free key at "
                "https://console.groq.com/keys"
            )
        return key.strip()

    @classmethod
    def is_api_key_configured(cls) -> bool:
        """Check whether a non-placeholder API key is set."""
        try:
            cls.get_groq_api_key()
            return True
        except ValueError:
            return False

    def get_tavily_api_key(self) -> Optional[str]:
        """Retrieve optional Tavily API key without logging or exposing secrets."""
        key = getattr(self, "TAVILY_API_KEY", None) or os.getenv("TAVILY_API_KEY")
        if key and str(key).strip() and "your_tavily" not in str(key):
            return str(key).strip()
        return None

    def get_serpapi_api_key(self) -> Optional[str]:
        """Retrieve optional SerpAPI key without logging or exposing secrets."""
        key = getattr(self, "SERPAPI_API_KEY", None) or os.getenv("SERPAPI_API_KEY")
        if key and str(key).strip() and "your_serpapi" not in str(key):
            return str(key).strip()
        return None


settings = Settings()

