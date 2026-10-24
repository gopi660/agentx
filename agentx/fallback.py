"""Groq Model Fallback and Resilience Manager for AgentX.

Provides reliable, bounded fallback across an ordered chain of Groq models.
Explicitly classifies errors (authentication vs transient vs rate-limit vs model-not-found)
and ensures credentials and sensitive details are never leaked into logs or error messages.
"""

from enum import Enum
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    Groq,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from agentx.config import settings

logger = logging.getLogger("agentx.fallback")


class ErrorCategory(str, Enum):
    """Categorization of errors for determining fallback eligibility."""

    AUTHENTICATION = "authentication"          # 401 Unauthorized (Ineligible)
    BAD_REQUEST = "bad_request"                # 400 Bad request / malformed schema (Ineligible)
    MODEL_UNAVAILABLE = "model_unavailable"    # 404 / 403 / deprecated model (Eligible)
    RATE_LIMIT = "rate_limit"                  # 429 Rate limit (Eligible / bounded wait)
    TRANSIENT_SERVER = "transient_server"      # 5xx Server error (Eligible)
    CONNECTION_TIMEOUT = "connection_timeout"  # Connection error / timeout (Eligible)
    PROGRAMMING_ERROR = "programming_error"    # Unexpected local exceptions (Ineligible)


def sanitize_message(msg: str) -> str:
    """Sanitize error messages to ensure API keys and bearer tokens are never leaked."""
    if not msg:
        return ""
    # Redact Bearer tokens
    sanitized = re.sub(r"Bearer\s+[a-zA-Z0-9_\-\.]+", "Bearer [REDACTED_TOKEN]", msg)
    # Redact groq API keys: gsk_...
    sanitized = re.sub(r"gsk_[a-zA-Z0-9_\-]+", "[REDACTED_API_KEY]", sanitized)
    return sanitized


class AllModelsFailedError(Exception):
    """Raised when all configured primary and fallback models have been exhausted."""

    def __init__(self, failures: List[Dict[str, Any]]):
        self.failures = failures
        summary_lines = []
        for f in failures:
            summary_lines.append(
                f"- Model '{f['model']}' failed ({f['category']}): {f['error']}"
            )
        sanitized_summary = "\n".join(summary_lines)
        super().__init__(
            f"All configured Groq models failed to complete the request:\n{sanitized_summary}"
        )


class ModelCallResult:
    """Metadata and response returned from a successful model call."""

    def __init__(
        self,
        response: Any,
        model_used: str,
        attempts: int,
        fallback_occurred: bool,
        history: List[Dict[str, Any]],
    ):
        self.response = response
        self.model_used = model_used
        self.attempts = attempts
        self.fallback_occurred = fallback_occurred
        self.history = history


class ModelFallbackManager:
    """Manages ordered model prioritization, error classification, and fallback execution."""

    def __init__(
        self,
        models: Optional[List[str]] = None,
        max_rate_limit_wait: Optional[float] = None,
        on_fallback_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Initialize manager with prioritized models.

        Args:
            models: Ordered list of model IDs. Defaults to settings.get_ordered_models().
            max_rate_limit_wait: Max seconds to wait for 429 retry-after.
            on_fallback_callback: Optional callback receiving fallback event info for traces.
        """
        raw_models = models or settings.get_ordered_models()
        # Deduplicate while strictly preserving order
        self.models: List[str] = []
        for m in raw_models:
            clean = m.strip()
            if clean and clean not in self.models:
                self.models.append(clean)

        self.max_rate_limit_wait = (
            max_rate_limit_wait
            if max_rate_limit_wait is not None
            else settings.MAX_RATE_LIMIT_WAIT_SECONDS
        )
        self.on_fallback_callback = on_fallback_callback

    def classify_error(self, exc: Exception) -> ErrorCategory:
        """Classify an exception to determine whether it is eligible for fallback."""
        if isinstance(exc, AuthenticationError):
            return ErrorCategory.AUTHENTICATION

        if isinstance(exc, RateLimitError):
            return ErrorCategory.RATE_LIMIT

        if isinstance(exc, (NotFoundError, PermissionDeniedError)):
            return ErrorCategory.MODEL_UNAVAILABLE

        if isinstance(exc, (APITimeoutError, APIConnectionError, TimeoutError, ConnectionError)):
            return ErrorCategory.CONNECTION_TIMEOUT

        if isinstance(exc, InternalServerError):
            return ErrorCategory.TRANSIENT_SERVER

        if isinstance(exc, BadRequestError):
            msg = str(exc).lower()
            # If the bad request is explicitly regarding the model ID itself (e.g. decommissioned)
            if "model" in msg and ("not found" in msg or "unknown" in msg or "deprecated" in msg):
                return ErrorCategory.MODEL_UNAVAILABLE
            return ErrorCategory.BAD_REQUEST

        # Check for rate limit / TPM quota / message size limits in exception message
        exc_str = str(exc).lower()
        if (
            "rate_limit_exceeded" in exc_str
            or "tokens per minute" in exc_str
            or "tpm" in exc_str
            or "request too large" in exc_str
            or "reduce your message size" in exc_str
        ):
            return ErrorCategory.RATE_LIMIT

        if isinstance(exc, APIStatusError):
            status = getattr(exc, "status_code", None)
            if status == 401:
                return ErrorCategory.AUTHENTICATION
            if status in (429, 413):
                return ErrorCategory.RATE_LIMIT
            if status in (403, 404):
                return ErrorCategory.MODEL_UNAVAILABLE
            if status and 500 <= status < 600:
                return ErrorCategory.TRANSIENT_SERVER
            if status == 400:
                return ErrorCategory.BAD_REQUEST

        # Unexpected local programming errors (TypeError, KeyError, etc.)
        return ErrorCategory.PROGRAMMING_ERROR

    def _extract_retry_after(self, exc: Exception) -> Optional[float]:
        """Extract retry-after seconds if provided in response headers."""
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", {})
            retry_val = headers.get("retry-after") or headers.get("Retry-After")
            if retry_val:
                try:
                    return float(retry_val)
                except ValueError:
                    pass
        # Also check error message for retry timing
        match = re.search(r"try again in ([0-9\.]+)s", str(exc), re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None

    def execute_with_fallback(
        self,
        groq_client: Groq,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = "auto",
        **extra_kwargs,
    ) -> ModelCallResult:
        """Execute chat completion across prioritized models with controlled fallback.

        Args:
            groq_client: Authenticated Groq client instance.
            messages: Conversation messages.
            tools: Registered tool schemas.
            tool_choice: Tool selection strategy ("auto").
            **extra_kwargs: Extra API parameters.

        Returns:
            ModelCallResult with the successful model response and metadata.

        Raises:
            AuthenticationError: On 401 unauthorized (ineligible for fallback).
            BadRequestError: On 400 invalid request/tool schema (ineligible for fallback).
            AllModelsFailedError: When all configured models fail.
        """
        if not self.models:
            raise ValueError("No models configured in ModelFallbackManager.")

        primary_model = self.models[0]
        failures: List[Dict[str, Any]] = []
        total_attempts = 0
        history: List[Dict[str, Any]] = []

        for model_idx, model_id in enumerate(self.models):
            is_fallback_model = model_idx > 0
            model_attempts = 0

            while model_attempts <= settings.MAX_PER_MODEL_RETRIES:
                total_attempts += 1
                model_attempts += 1

                try:
                    call_params = {
                        "model": model_id,
                        "messages": messages,
                        **extra_kwargs,
                    }
                    if tools:
                        call_params["tools"] = tools
                        call_params["tool_choice"] = tool_choice

                    # Perform API call
                    response = groq_client.chat.completions.create(**call_params)

                    # Success!
                    fallback_occurred = (model_id != primary_model) or (len(failures) > 0)
                    history_entry = {
                        "model": model_id,
                        "attempt": total_attempts,
                        "status": "success",
                        "fallback": fallback_occurred,
                    }
                    history.append(history_entry)

                    if fallback_occurred and self.on_fallback_callback:
                        try:
                            self.on_fallback_callback({
                                "event": "fallback_success",
                                "primary_model": primary_model,
                                "successful_model": model_id,
                                "attempts": total_attempts,
                            })
                        except Exception:
                            pass

                    return ModelCallResult(
                        response=response,
                        model_used=model_id,
                        attempts=total_attempts,
                        fallback_occurred=fallback_occurred,
                        history=history,
                    )

                except Exception as exc:
                    category = self.classify_error(exc)
                    sanitized_err = sanitize_message(str(exc))

                    logger.warning(
                        "Groq call failed for model '%s' (attempt %d, category: %s): %s",
                        model_id,
                        total_attempts,
                        category.value,
                        sanitized_err,
                    )

                    history.append({
                        "model": model_id,
                        "attempt": total_attempts,
                        "status": "failed",
                        "category": category.value,
                        "error": sanitized_err,
                    })

                    # --- Ineligible for fallback policies ---
                    if category == ErrorCategory.AUTHENTICATION:
                        # Stop immediately: Authentication failure affects entire API key
                        raise AuthenticationError(
                            message=f"Groq authentication failed: {sanitized_err}. Check your GROQ_API_KEY.",
                            response=getattr(exc, "response", None),
                            body=getattr(exc, "body", None),
                        ) from exc

                    if category == ErrorCategory.BAD_REQUEST:
                        # Stop immediately: Request or tool schema is malformed
                        raise BadRequestError(
                            message=f"Invalid request to Groq API: {sanitized_err}",
                            response=getattr(exc, "response", None),
                            body=getattr(exc, "body", None),
                        ) from exc

                    if category == ErrorCategory.PROGRAMMING_ERROR:
                        # Local unexpected programming exception; re-raise directly
                        raise exc

                    # --- Rate limit bounded retry check for current model ---
                    if category == ErrorCategory.RATE_LIMIT and model_attempts <= settings.MAX_PER_MODEL_RETRIES:
                        # If error is a TPM token limit / request too large, retrying the same model immediately won't help
                        if "request too large" in sanitized_err.lower() or "tokens per minute" in sanitized_err.lower() or "tpm" in sanitized_err.lower():
                            pass
                        else:
                            retry_after = self._extract_retry_after(exc)
                            if retry_after is not None and retry_after <= self.max_rate_limit_wait:
                                time.sleep(retry_after)
                                continue  # Retry same model once

                    # Record model failure and break inner retry loop to move to next fallback model
                    failures.append({
                        "model": model_id,
                        "category": category.value,
                        "error": sanitized_err,
                    })

                    if self.on_fallback_callback:
                        try:
                            next_model = (
                                self.models[model_idx + 1]
                                if model_idx + 1 < len(self.models)
                                else None
                            )
                            self.on_fallback_callback({
                                "event": "model_failure",
                                "failed_model": model_id,
                                "next_model": next_model,
                                "category": category.value,
                                "error": sanitized_err,
                            })
                        except Exception:
                            pass

                    break  # Move to next model in self.models

        # All models exhausted
        raise AllModelsFailedError(failures)
