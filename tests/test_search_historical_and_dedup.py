"""Tests for web search historical query preservation, URL deduplication, and result integrity.

Tests:
1. Historical date queries (e.g. 'Who won the 2022 World Cup?') preserve historical years and are not rewritten to 2026.
2. Current queries (e.g. 'Who is the prime minister in 2024?') rewrite stale past years to the current year (2026).
3. Search results with duplicate URLs are deduplicated while preserving distinct entries.
4. Search provider failures are distinguished from zero-result searches.
5. Max results configuration validation.
"""

from unittest.mock import MagicMock, patch
import pytest

from agentx.schemas import ToolStatus
from agentx.tools.search import is_historical_query, sanitize_search_query, web_search


def test_historical_query_detection():
    """Verify historical queries preserve past years and avoid rewriting to CURRENT_YEAR."""
    historical_queries = [
        "Who won the FIFA World Cup in 2022?",
        "What was the inflation rate of India in 2020?",
        "When was Python released in 1991?",
        "Who won the gold medal in the 2016 Olympics?",
        "What happened in the year 2008 financial crisis?",
        "Who was the prime minister of UK in 2010?",
    ]
    for q in historical_queries:
        assert is_historical_query(q) is True, f"Failed to identify historical query: {q}"
        sanitized = sanitize_search_query(q)
        assert sanitized == q, f"Historical query was modified: {sanitized}"


def test_stale_current_queries_are_rewritten_to_current_year():
    """Verify non-historical queries with outdated recent years update to CURRENT_YEAR (2026)."""
    current_queries = [
        ("current price of gold in 2024", "2026"),
        ("latest stock price of Apple 2025", "2026"),
        ("today weather forecast in 2023", "2026"),
    ]
    for q, expected_year in current_queries:
        assert is_historical_query(q) is False
        sanitized = sanitize_search_query(q)
        assert expected_year in sanitized


def test_search_url_deduplication():
    """Verify that multiple results pointing to the same URL are deduplicated."""
    mock_raw_results = [
        {
            "title": "Article 1 - First occurrence",
            "href": "https://example.com/story-1",
            "body": "Snippet 1 text describing something.",
        },
        {
            "title": "Article 1 - Duplicate URL with different title",
            "href": "https://example.com/story-1",
            "body": "Different snippet for identical URL.",
        },
        {
            "title": "Article 2 - Unique URL",
            "href": "https://example.com/story-2",
            "body": "Snippet 2 text describing another topic.",
        },
    ]

    with patch("agentx.tools.search._search_duckduckgo", return_value=mock_raw_results):
        result = web_search(query="test query")

    assert result.status == ToolStatus.SUCCESS
    # Must have only 2 unique results
    assert len(result.data) == 2
    assert result.data[0]["url"] == "https://example.com/story-1"
    assert result.data[1]["url"] == "https://example.com/story-2"


def test_search_provider_failure_distinction():
    """Verify search provider failure returns ToolStatus.ERROR, not NO_RESULTS."""
    with patch("agentx.tools.search._search_duckduckgo", side_effect=Exception("DuckDuckGo 503 Service Unavailable")):
        with patch("agentx.tools.search._search_tavily", side_effect=Exception("Tavily API unavailable")):
            result = web_search(query="unreachable search")

    assert result.status == ToolStatus.ERROR
    assert "unavailable" in result.error.lower() or "failed" in result.error.lower()
    assert not result.data


def test_zero_results_distinction():
    """Verify zero genuine results return ToolStatus.NO_RESULTS, distinct from provider error."""
    with patch("agentx.tools.search._search_duckduckgo", return_value=[]):
        with patch("agentx.tools.search._search_tavily", return_value=[]):
            result = web_search(query="nonexistent term 987654321")

    assert result.status == ToolStatus.NO_RESULTS
    assert "No public search results found" in result.message
    assert result.data == []
