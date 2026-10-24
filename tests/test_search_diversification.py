"""Unit tests for search query diversification, stale-year sanitation, provider failover, and final response synthesis."""

import json
from unittest.mock import MagicMock, patch
import pytest

from agentx.agent import AgentX
from agentx.config import settings
from agentx.schemas import ToolResult, ToolStatus
from agentx.tools.search import (
    are_queries_equivalent,
    generate_query_alternatives,
    normalize_query_semantics,
    sanitize_search_query,
    web_search,
)


def _make_choice(content=None, tool_calls=None):
    msg = MagicMock()
    msg.role = "assistant"
    msg.content = content
    msg.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = msg
    return MagicMock(choices=[choice], usage=None)


def test_successful_search_returns_structured_evidence():
    """Verify successful search parses titles, links, snippets, and dates."""
    mock_ddg = [
        {
            "title": "Crucial 8GB DDR4 3200MHz RAM - Amazon.in",
            "href": "https://amazon.in/dp/B08C4Z69LN",
            "body": "Buy Crucial 8GB DDR4 RAM online at best price in India: Rs 1,899 with warranty.",
            "date": "2026-02-15",
        }
    ]
    with patch("agentx.tools.search._search_duckduckgo", return_value=mock_ddg):
        res = web_search("Crucial 8GB DDR4 RAM India")

    assert res.status == ToolStatus.SUCCESS
    assert len(res.data) == 1
    assert res.data[0]["title"] == "Crucial 8GB DDR4 3200MHz RAM - Amazon.in"
    assert "https://amazon.in/dp/B08C4Z69LN" in res.message
    assert "Rs 1,899" in res.message
    assert res.evidence["result_count"] == 1


def test_empty_search_results_no_fabrication():
    """Verify empty search results return NO_RESULTS and do not hallucinate data."""
    with patch("agentx.tools.search._search_duckduckgo", return_value=[]):
        res = web_search("nonexistent_rare_ram_module_xyz")

    assert res.status == ToolStatus.NO_RESULTS
    assert res.data == []
    assert "No public search results found" in res.message
    assert "Suggested query refinements" in res.message


def test_provider_timeout_and_http_errors():
    """Verify network exceptions in provider are handled safely without crashing."""
    with patch("agentx.tools.search._search_duckduckgo", side_effect=Exception("Connection timed out after 10s")):
        res = web_search("RAM price India")

    assert res.status == ToolStatus.ERROR
    assert "Connection timed out" in res.error


def test_duplicate_query_suppression():
    """Verify semantic equivalency detection and duplicate query suppression."""
    q1 = "RAM price India"
    q2 = "price of RAM India"
    q3 = "8GB DDR4 RAM price India"
    q4 = "8GB DDR4 RAM price in India INR"

    assert are_queries_equivalent(q1, q2) is True
    assert are_queries_equivalent(q3, q4) is True
    assert are_queries_equivalent(q1, q3) is False

    # Normalized tokens check
    norm_q1 = normalize_query_semantics(q1)
    norm_q2 = normalize_query_semantics(q2)
    assert norm_q1 == norm_q2


def test_stale_year_sanitization_and_current_year_generation():
    """Verify stale past years (e.g. 2024) are updated to current year (2026)."""
    stale_query = "DDR4 8GB price India 2024"
    cleaned = sanitize_search_query(stale_query)
    assert "2024" not in cleaned
    assert "2026" in cleaned

    alternatives = generate_query_alternatives("8GB DDR4 RAM price in India")
    assert len(alternatives) >= 2
    # At least one alternative mentions 2026 or online
    alt_text = " ".join(alternatives)
    assert "2026" in alt_text or "online" in alt_text


def test_fallback_provider_when_primary_empty():
    """Verify search attempts fallback provider (e.g. Tavily) when configured and DDG is empty."""
    mock_tavily_results = [
        {
            "title": "Kingston FURY Beast 8GB DDR4 3200MHz",
            "url": "https://mdcomputers.in/kingston-fury-8gb-ddr4.html",
            "snippet": "Price in India: INR 1,750. In stock.",
            "date": "2026-03-01",
        }
    ]

    with patch("agentx.tools.search._search_duckduckgo", return_value=[]):
        with patch.object(settings, "TAVILY_API_KEY", "mock_tvly_key"):
            with patch("agentx.tools.search._search_tavily", return_value=mock_tavily_results) as mock_tav:
                res = web_search("Kingston 8GB DDR4 RAM India")
                mock_tav.assert_called_once()
                assert res.status == ToolStatus.SUCCESS
                assert "mdcomputers.in" in res.message


def test_final_response_when_all_searches_fail():
    """Verify agent never leaves a blank response when search attempts fail, and suggests refinements."""
    mock_client = MagicMock()

    # Step 1: Model calls search for "RAM price in India"
    tc1 = MagicMock(id="search_1")
    tc1.function.name = "web_search"
    tc1.function.arguments = json.dumps({"query": "RAM price in India"})
    step1_resp = _make_choice(content=None, tool_calls=[tc1])

    # Step 2: Model returns empty response (empty string or None)
    step2_empty = _make_choice(content="")
    step3_empty = _make_choice(content="   ")

    mock_client.chat.completions.create.side_effect = [step1_resp, step2_empty, step3_empty]

    with patch("agentx.tools.search._search_duckduckgo", return_value=[]):
        agent = AgentX(groq_client=mock_client)
        resp = agent.run("what is the RAM price in India")

    # Assert response is not blank
    assert resp.final_answer and len(resp.final_answer.strip()) > 0
    # Must explain that prices could not be verified and suggest refinements
    assert "could not" in resp.final_answer.lower() or "unverified" in resp.final_answer.lower() or "not locate" in resp.final_answer.lower()
    assert "refining" in resp.final_answer.lower() or "ddr4" in resp.final_answer.lower() or "amazon" in resp.final_answer.lower()


def test_maximum_search_attempts_per_request_enforced():
    """Verify at most MAX_SEARCH_ATTEMPTS_PER_REQUEST (3) distinct searches occur in one request."""
    mock_client = MagicMock()

    # Model endlessly requests search with different queries
    call_idx = [0]
    def make_search_call(*args, **kwargs):
        idx = call_idx[0]
        call_idx[0] += 1
        tc = MagicMock(id=f"search_{idx}")
        tc.function.name = "web_search"
        tc.function.arguments = json.dumps({"query": f"distinct ram query {idx}"})
        return _make_choice(content=None, tool_calls=[tc])

    mock_client.chat.completions.create.side_effect = make_search_call

    with patch("agentx.tools.search._search_duckduckgo", return_value=[]):
        agent = AgentX(groq_client=mock_client)
        resp = agent.run("Find ram prices")

    # Filter search traces
    search_traces = [t for t in resp.tool_traces if t.get("tool") == "web_search"]
    successful_executions = [t for t in search_traces if "Search limit reached" not in str(t.get("message", ""))]
    assert len(successful_executions) <= settings.MAX_SEARCH_ATTEMPTS_PER_REQUEST
