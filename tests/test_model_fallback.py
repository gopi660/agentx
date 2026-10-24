"""Comprehensive test suite for Groq model fallback, resilience, and error classification.

Tests cover all 16 required fallback and resilience scenarios:
1. Primary model succeeds: no fallback occurs.
2. Primary receives a retryable failure; first fallback succeeds.
3. Primary and first fallback fail; second fallback succeeds.
4. All models fail: a clear sanitized error is returned.
5. Authentication failure stops the fallback process immediately.
6. Model-specific invalid-ID/access failure follows documented policy.
7. HTTP 429 handling respects bounded retry behavior.
8. Timeout or transient HTTP 5xx triggers fallback.
9. Duplicate model IDs are removed while order is preserved.
10. Empty or malformed fallback configuration is handled safely.
11. Tool-call responses from fallback models are processed correctly.
12. Previously executed tools are not repeated after a later model failure.
13. LangGraph state and tool results survive a model switch.
14. Maximum tool iterations remain enforced.
15. Duplicate-response prevention remains functional after fallback.
16. API keys and sensitive details never appear in error messages or logs.
"""

import json
from unittest.mock import MagicMock, patch
import pytest
from groq import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from agentx.agent import AgentX
from agentx.config import settings
from agentx.fallback import (
    AllModelsFailedError,
    ErrorCategory,
    ModelFallbackManager,
    sanitize_message,
)
from agentx.response_manager import SessionHistoryManager
from agentx.schemas import ToolStatus


MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]


def _make_mock_response(content="Test response", tool_calls=None):
    """Create a mock Groq ChatCompletion response."""
    choice = MagicMock()
    choice.message = MagicMock(content=content, tool_calls=tool_calls)
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# -------------------------------------------------------------------------
# Test 1: Primary model succeeds: no fallback occurs.
# -------------------------------------------------------------------------
def test_primary_model_succeeds_no_fallback():
    mgr = ModelFallbackManager(models=MODELS)
    client = MagicMock()
    client.chat.completions.create.return_value = _make_mock_response("Primary success")

    res = mgr.execute_with_fallback(client, [{"role": "user", "content": "Hi"}])
    assert res.model_used == "openai/gpt-oss-120b"
    assert res.fallback_occurred is False
    assert res.attempts == 1
    assert client.chat.completions.create.call_count == 1
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-oss-120b"


# -------------------------------------------------------------------------
# Test 2: Primary receives a retryable failure; first fallback succeeds.
# -------------------------------------------------------------------------
def test_primary_retryable_failure_first_fallback_succeeds():
    mgr = ModelFallbackManager(models=MODELS)
    client = MagicMock()

    # Primary fails with 500; First fallback succeeds
    err_500 = InternalServerError("Server busy", response=MagicMock(status_code=500), body={})
    success_resp = _make_mock_response("Fallback 1 success")

    client.chat.completions.create.side_effect = [err_500, success_resp]

    res = mgr.execute_with_fallback(client, [{"role": "user", "content": "Hi"}])
    assert res.model_used == "qwen/qwen3.8-27b"
    assert res.fallback_occurred is True
    assert res.attempts == 2
    assert client.chat.completions.create.call_count == 2
    assert client.chat.completions.create.call_args_list[0].kwargs["model"] == "openai/gpt-oss-120b"
    assert client.chat.completions.create.call_args_list[1].kwargs["model"] == "qwen/qwen3.8-27b"


# -------------------------------------------------------------------------
# Test 3: Primary and first fallback fail; second fallback succeeds.
# -------------------------------------------------------------------------
def test_primary_and_first_fallback_fail_second_fallback_succeeds():
    mgr = ModelFallbackManager(models=MODELS)
    client = MagicMock()

    timeout_err = APITimeoutError(request=MagicMock())
    err_503 = InternalServerError("Service unavailable", response=MagicMock(status_code=503), body={})
    success_resp = _make_mock_response("Fallback 2 success")

    client.chat.completions.create.side_effect = [timeout_err, err_503, success_resp]

    res = mgr.execute_with_fallback(client, [{"role": "user", "content": "Hi"}])
    assert res.model_used == "openai/gpt-oss-20b"
    assert res.fallback_occurred is True
    assert res.attempts == 3
    assert client.chat.completions.create.call_args_list[2].kwargs["model"] == "openai/gpt-oss-20b"


# -------------------------------------------------------------------------
# Test 4: All models fail: a clear sanitized error is returned.
# -------------------------------------------------------------------------
def test_all_models_fail_clear_sanitized_error():
    mgr = ModelFallbackManager(models=MODELS)
    client = MagicMock()

    err = InternalServerError("Service overloaded with key gsk_supersecret123", response=MagicMock(status_code=500), body={})
    client.chat.completions.create.side_effect = err

    with pytest.raises(AllModelsFailedError) as exc_info:
        mgr.execute_with_fallback(client, [{"role": "user", "content": "Hi"}])

    error_text = str(exc_info.value)
    assert "All configured Groq models failed" in error_text
    assert "openai/gpt-oss-120b" in error_text
    assert "qwen/qwen3.8-27b" in error_text
    assert "openai/gpt-oss-20b" in error_text
    # Sensitive key must be sanitized
    assert "gsk_supersecret123" not in error_text
    assert "[REDACTED_API_KEY]" in error_text


# -------------------------------------------------------------------------
# Test 5: Authentication failure stops the fallback process immediately.
# -------------------------------------------------------------------------
def test_authentication_failure_stops_fallback():
    mgr = ModelFallbackManager(models=MODELS)
    client = MagicMock()

    auth_err = AuthenticationError("Invalid API key provided", response=MagicMock(status_code=401), body={})
    client.chat.completions.create.side_effect = auth_err

    with pytest.raises(AuthenticationError) as exc_info:
        mgr.execute_with_fallback(client, [{"role": "user", "content": "Hi"}])

    # Only 1 attempt made: must NOT try fallback models on 401
    assert client.chat.completions.create.call_count == 1
    assert "authentication failed" in str(exc_info.value).lower()


# -------------------------------------------------------------------------
# Test 6: Model-specific invalid-ID/access failure follows documented policy.
# -------------------------------------------------------------------------
def test_model_specific_invalid_id_or_access_failure():
    mgr = ModelFallbackManager(models=MODELS)
    client = MagicMock()

    not_found = NotFoundError("Model 'openai/gpt-oss-120b' not found or deprecated", response=MagicMock(status_code=404), body={})
    success_resp = _make_mock_response("Second model success")

    client.chat.completions.create.side_effect = [not_found, success_resp]

    res = mgr.execute_with_fallback(client, [{"role": "user", "content": "Hi"}])
    assert res.model_used == "qwen/qwen3.8-27b"
    assert res.fallback_occurred is True
    # The not_found model must not be retried; cleanly switches to next
    assert client.chat.completions.create.call_count == 2


# -------------------------------------------------------------------------
# Test 7: HTTP 429 handling respects bounded retry behavior.
# -------------------------------------------------------------------------
def test_http_429_bounded_retry_behavior():
    mgr = ModelFallbackManager(models=MODELS, max_rate_limit_wait=1.0)
    client = MagicMock()

    # Case A: 429 with small retry-after <= bound: retries same model once
    rate_err = RateLimitError("Rate limit reached. Please try again in 0.1s.", response=MagicMock(status_code=429, headers={"retry-after": "0.01"}), body={})
    success_resp = _make_mock_response("Success on retry")

    client.chat.completions.create.side_effect = [rate_err, success_resp]

    with patch("time.sleep") as mock_sleep:
        res = mgr.execute_with_fallback(client, [{"role": "user", "content": "Hi"}])
        assert res.model_used == "openai/gpt-oss-120b"
        assert res.attempts == 2
        mock_sleep.assert_called_once()

    # Case B: 429 with retry-after exceeding bound: falls back immediately without sleeping
    client_b = MagicMock()
    excessive_rate_err = RateLimitError("Rate limit. Try again in 60s.", response=MagicMock(status_code=429, headers={"retry-after": "60.0"}), body={})
    client_b.chat.completions.create.side_effect = [excessive_rate_err, _make_mock_response("Fallback success")]

    with patch("time.sleep") as mock_sleep:
        res_b = mgr.execute_with_fallback(client_b, [{"role": "user", "content": "Hi"}])
        assert res_b.model_used == "qwen/qwen3.8-27b"
        mock_sleep.assert_not_called()


# -------------------------------------------------------------------------
# Test 8: Timeout or transient HTTP 5xx triggers fallback.
# -------------------------------------------------------------------------
def test_timeout_or_transient_5xx_triggers_fallback():
    mgr = ModelFallbackManager(models=MODELS)
    client = MagicMock()

    conn_err = APIConnectionError(request=MagicMock())
    success_resp = _make_mock_response("Recovered via fallback")

    client.chat.completions.create.side_effect = [conn_err, success_resp]

    res = mgr.execute_with_fallback(client, [{"role": "user", "content": "Hi"}])
    assert res.model_used == "qwen/qwen3.8-27b"
    assert res.fallback_occurred is True


# -------------------------------------------------------------------------
# Test 9: Duplicate model IDs are removed while order is preserved.
# -------------------------------------------------------------------------
def test_duplicate_model_ids_removed_order_preserved():
    duplicates = [
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
    ]
    mgr = ModelFallbackManager(models=duplicates)
    assert mgr.models == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]


# -------------------------------------------------------------------------
# Test 10: Empty or malformed fallback configuration is handled safely.
# -------------------------------------------------------------------------
def test_empty_or_malformed_fallback_configuration_handled():
    with patch.dict("os.environ", {"GROQ_MODEL": "openai/gpt-oss-120b", "GROQ_FALLBACK_MODELS": " ,  , ,, "}):
        ordered = settings.get_ordered_models()
        assert ordered == ["openai/gpt-oss-120b"]

    with patch.dict("os.environ", {"GROQ_MODEL": "  custom/model-1  ", "GROQ_FALLBACK_MODELS": "  custom/model-2 , custom/model-1 "}):
        ordered = settings.get_ordered_models()
        assert ordered == ["custom/model-1", "custom/model-2"]


# -------------------------------------------------------------------------
# Test 11: Tool-call responses from fallback models are processed correctly.
# -------------------------------------------------------------------------
def test_tool_call_responses_from_fallback_model_processed():
    mock_client = MagicMock()
    fb_mgr = ModelFallbackManager(models=["primary-fail-model", "fallback-success-model"])

    # Turn 1: primary fails with 500, fallback returns tool call for calculator
    err_500 = InternalServerError("Crash", response=MagicMock(status_code=500), body={})
    tc = MagicMock()
    tc.id = "call_calc"
    tc.function.name = "calculator"
    tc.function.arguments = json.dumps({"expression": "25 * 4"})
    tool_resp = _make_mock_response(content=None, tool_calls=[tc])

    # Turn 2: primary fails again with 500, fallback returns final answer
    final_resp = _make_mock_response(content="The calculated result is 100.")

    mock_client.chat.completions.create.side_effect = [err_500, tool_resp, err_500, final_resp]

    agent = AgentX(groq_client=mock_client, fallback_manager=fb_mgr)
    response = agent.run("Calculate 25 * 4")

    assert response.final_answer == "The calculated result is 100."
    assert len(response.tool_traces) == 1
    assert response.tool_traces[0]["tool"] == "calculator"
    assert response.tool_traces[0]["status"] == ToolStatus.SUCCESS.value
    assert response.model_used == "fallback-success-model"
    assert response.fallback_occurred is True


# -------------------------------------------------------------------------
# Test 12: Previously executed tools are not repeated after a later model failure.
# -------------------------------------------------------------------------
def test_previously_executed_tools_not_repeated_after_model_failure():
    mock_client = MagicMock()
    fb_mgr = ModelFallbackManager(models=["model-a", "model-b"])

    # Step 1: Model A requests calculator
    tc = MagicMock()
    tc.id = "call_calc_once"
    tc.function.name = "calculator"
    tc.function.arguments = json.dumps({"expression": "10 + 20"})
    step1_resp = _make_mock_response(content=None, tool_calls=[tc])

    # Step 2: Model A fails with 503; Model B takes over and finishes
    err_503 = InternalServerError("Unavailable", response=MagicMock(status_code=503), body={})
    step2_resp = _make_mock_response(content="Sum is 30.")

    mock_client.chat.completions.create.side_effect = [step1_resp, err_503, step2_resp]

    agent = AgentX(groq_client=mock_client, fallback_manager=fb_mgr)

    # Spy on tool_registry.execute
    orig_execute = agent.tool_registry.execute
    execute_calls = []

    def spy_execute(name, args):
        execute_calls.append((name, args))
        return orig_execute(name, args)

    agent.tool_registry.execute = spy_execute

    resp = agent.run("What is 10 + 20?")

    assert resp.final_answer == "Sum is 30."
    # Crucial: calculator must have executed EXACTLY ONCE across the entire run
    assert len(execute_calls) == 1
    assert execute_calls[0][0] == "calculator"
    assert len(resp.tool_traces) == 1
    assert resp.model_used == "model-b"
    assert resp.fallback_occurred is True


# -------------------------------------------------------------------------
# Test 13: LangGraph state and tool results survive a model switch.
# -------------------------------------------------------------------------
def test_langgraph_state_and_tool_results_survive_model_switch():
    mock_client = MagicMock()
    fb_mgr = ModelFallbackManager(models=["model-step1", "model-step2"])

    # Step 1: model-step1 requests KB search
    tc = MagicMock(id="call_kb", function=MagicMock(name="knowledge_base_search", arguments=json.dumps({"query": "SSRF"})))
    resp1 = _make_mock_response(content=None, tool_calls=[tc])

    # Step 2: model-step1 fails with timeout; model-step2 produces answer citing KB-002
    err_timeout = APITimeoutError(request=MagicMock())
    resp2 = _make_mock_response(content="Per policy [KB-002], private IP ranges are blocked.")

    mock_client.chat.completions.create.side_effect = [resp1, err_timeout, resp2]

    agent = AgentX(groq_client=mock_client, fallback_manager=fb_mgr)
    resp = agent.run("What is SSRF policy?")

    assert "KB-002" in resp.final_answer
    assert resp.model_used == "model-step2"
    assert resp.fallback_occurred is True


# -------------------------------------------------------------------------
# Test 14: Maximum tool iterations remain enforced.
# -------------------------------------------------------------------------
def test_maximum_tool_iterations_remain_enforced():
    mock_client = MagicMock()
    fb_mgr = ModelFallbackManager(models=["loop-model"])

    # Loop-model endlessly requests calculator
    tc = MagicMock(id="call_loop", function=MagicMock(name="calculator", arguments=json.dumps({"expression": "1 + 1"})))
    loop_resp = _make_mock_response(content=None, tool_calls=[tc])
    mock_client.chat.completions.create.return_value = loop_resp

    agent = AgentX(groq_client=mock_client, fallback_manager=fb_mgr)
    resp = agent.run("Loop forever")

    assert resp.iterations >= settings.MAX_TOOL_ITERATIONS
    assert "maximum tool iterations limit" in resp.final_answer.lower()


# -------------------------------------------------------------------------
# Test 15: Duplicate-response prevention remains functional after fallback.
# -------------------------------------------------------------------------
def test_duplicate_response_prevention_functional_after_fallback():
    mock_client = MagicMock()
    fb_mgr = ModelFallbackManager(models=["failing-primary", "working-fallback"])
    session_mgr = SessionHistoryManager()
    session_id = "dup_fallback_session"

    session_mgr.record_delivered_answer(session_id, "The answer is exactly 42.")

    # Primary fails with 500; fallback returns duplicate answer ("The answer is exactly 42.")
    err_500 = InternalServerError("Crash", response=MagicMock(status_code=500), body={})
    dup_resp = _make_mock_response(content="The answer is exactly 42.")
    # Rephrase request is sent to fallback which returns distinct phrasing
    rephrased_resp = _make_mock_response(content="Result verified: 42.")

    mock_client.chat.completions.create.side_effect = [err_500, dup_resp, rephrased_resp]

    agent = AgentX(groq_client=mock_client, session_hist_manager=session_mgr, fallback_manager=fb_mgr)
    resp = agent.run("What is 6 * 7?", session_id=session_id)

    assert resp.final_answer == "Result verified: 42."
    assert resp.duplicate_retries == 1
    assert resp.fallback_occurred is True


# -------------------------------------------------------------------------
# Test 16: API keys and sensitive details never appear in error messages or logs.
# -------------------------------------------------------------------------
def test_api_keys_and_sensitive_details_never_leak():
    sensitive_raw = "Connection refused to https://api.groq.com with key gsk_liveSecretKey99999 and Bearer gsk_liveSecretKey99999"
    sanitized = sanitize_message(sensitive_raw)

    assert "gsk_liveSecretKey99999" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized
    assert "Bearer [REDACTED_TOKEN]" in sanitized
