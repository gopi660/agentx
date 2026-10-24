"""Unit and integration tests for general-purpose web search discovery and research workflow."""

from unittest.mock import MagicMock, patch
import pytest

from agentx.agent import AgentX
from agentx.schemas import ToolStatus
from agentx.tools.search import web_search, _extract_date_snippet


def test_date_snippet_extraction():
    """Verify regex date extraction recognizes standard and relative dates."""
    assert _extract_date_snippet("Updated on 24 October 2026 by Reuters") == "24 October 2026"
    assert _extract_date_snippet("Published 2026-10-24 in Finance") == "2026-10-24"
    assert _extract_date_snippet("3 days ago — Gold prices rose significantly") == "3 days ago"
    assert _extract_date_snippet("No date mentioned in this article.") is None


def test_web_search_invalid_inputs():
    """Verify empty and whitespace search queries are rejected."""
    res_empty = web_search("")
    assert res_empty.status == ToolStatus.INVALID_INPUT
    assert "non-empty string" in res_empty.error

    res_spaces = web_search("   \n\t  ")
    assert res_spaces.status == ToolStatus.INVALID_INPUT
    assert "whitespace only" in res_spaces.error


def test_web_search_duckduckgo_success():
    """Verify DuckDuckGo search provider returns formatted, safe search results."""
    mock_raw = [
        {
            "title": "DDR5 RAM Price Trends in India (October 2026)",
            "href": "https://example-tech.com/ram-prices-2026",
            "body": "24 October 2026 — Latest prices for DDR5 16GB and 32GB RAM kits in India.",
        },
        {
            "title": "Computer Memory Deals",
            "href": "https://example-hardware.in/deals",
            "body": "Find current discounts on desktop and laptop RAM modules.",
        },
    ]

    with patch("agentx.tools.search.validate_url_for_ssrf", return_value=(True, None, "93.184.215.14")):
        with patch("agentx.tools.search._search_duckduckgo", return_value=mock_raw):
            res = web_search("current RAM prices in India", max_results=2)

    assert res.status == ToolStatus.SUCCESS
    assert len(res.data) == 2
    assert res.data[0]["title"] == "DDR5 RAM Price Trends in India (October 2026)"
    assert res.data[0]["url"] == "https://example-tech.com/ram-prices-2026"
    assert res.data[0]["date"] == "24 October 2026"
    assert "example-tech.com" in res.message
    assert res.evidence["result_count"] == 2


def test_web_search_empty_results():
    """Verify empty search returns NO_RESULTS status without fabricating data."""
    with patch("agentx.tools.search._search_duckduckgo", return_value=[]):
        res = web_search("xyznonexistenttopic987654321")

    assert res.status == ToolStatus.NO_RESULTS
    assert res.data == []
    assert "No public search results found" in res.message
    assert res.evidence["results"] == []


def test_web_search_ssrf_filtering():
    """Verify discovered search result URLs pointing to internal/private IPs are filtered."""
    mock_mixed = [
        {
            "title": "Public Guide",
            "url": "https://public-news.com/gold",
            "snippet": "Public gold market rates.",
            "date": "24 Oct 2026",
        },
        {
            "title": "Internal Metadata Service",
            "url": "http://169.254.169.254/latest/meta-data",
            "snippet": "Cloud metadata",
            "date": None,
        },
        {
            "title": "Private Intranet",
            "url": "http://192.168.1.1/router",
            "snippet": "Router config",
            "date": None,
        },
    ]

    with patch("agentx.tools.search._search_duckduckgo", return_value=mock_mixed):
        with patch("agentx.tools.search.validate_url_for_ssrf") as mock_validate:
            # Emulate real SSRF validation: allow public-news.com, block 169.254 and 192.168
            def _ssrf_side_effect(url):
                if "169.254" in url or "192.168" in url:
                    return False, "Prohibited IP address", None
                return True, None, "93.184.216.34"
            mock_validate.side_effect = _ssrf_side_effect
            res = web_search("gold prices and intranet")

    assert res.status == ToolStatus.SUCCESS
    # Only the public URL should be retained
    assert len(res.data) == 1
    assert res.data[0]["url"] == "https://public-news.com/gold"
    assert res.evidence["blocked_unsafe_count"] == 2


def test_web_search_provider_error_handling():
    """Verify provider network exceptions return structured ToolStatus.ERROR."""
    with patch("agentx.tools.search._search_duckduckgo", side_effect=RuntimeError("Search engine rate limited")):
        res = web_search("unfamiliar topic inquiry")

    assert res.status == ToolStatus.ERROR
    assert "Search engine rate limited" in res.error
    assert res.evidence["query"] == "unfamiliar topic inquiry"


def test_agent_workflow_general_research_with_web_search():
    """Verify full agent workflow uses web_search to discover sources and produces grounded answers."""
    mock_client = MagicMock()

    # Turn 1: Model requests web_search for unfamiliar query
    call_mock = MagicMock()
    call_mock.id = "call_search_1"
    call_mock.function.name = "web_search"
    call_mock.function.arguments = '{"query": "anemometer weather equipment price in india"}'

    step1_msg = MagicMock(content=None, tool_calls=[call_mock])

    # Turn 2: Model receives verified search results and outputs grounded answer with citations
    step2_msg = MagicMock(
        content=(
            "Based on verified search results, digital anemometers in India currently range from "
            "₹1,200 for basic handheld sensors to ₹8,500 for professional solar-powered weather stations. "
            "Sources: [IndiaMART Weather Tools](https://example.com/anemometer) (updated 24 Oct 2026)."
        ),
        tool_calls=None,
    )

    mock_client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=step1_msg)]),
        MagicMock(choices=[MagicMock(message=step2_msg)]),
    ]

    mock_search_results = [
        {
            "title": "IndiaMART Weather Tools",
            "href": "https://example.com/anemometer",
            "body": "24 Oct 2026 — Portable digital anemometer and wind speed meter pricing.",
        }
    ]

    with patch("agentx.tools.search._search_duckduckgo", return_value=mock_search_results):
        agent = AgentX(groq_client=mock_client)
        response = agent.run("What is the current price of anemometer weather equipment in India?")

    assert len(response.tool_traces) == 1
    assert response.tool_traces[0]["tool"] == "web_search"
    assert response.tool_traces[0]["status"] == ToolStatus.SUCCESS.value
    assert "anemometer" in response.final_answer.lower()
    assert "https://example.com/anemometer" in response.final_answer
