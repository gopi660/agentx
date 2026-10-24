"""Unit and integration tests for browser content extraction and cleanup in browse_page.

Tests:
1. Successful extraction of final_url, title, main_heading, and visible text.
2. Truncation reporting when text exceeds BROWSER_MAX_TEXT_LENGTH.
3. Empty page content returns ToolStatus.NO_RESULTS.
4. Navigation timeout returns ToolStatus.ERROR.
5. Navigation failure returns ToolStatus.ERROR with clean diagnostic.
6. SSRF-prohibited URLs rejected before navigation.
7. DOM cleanup strips scripts, styles, nav, and cookie banners.
"""

from unittest.mock import MagicMock, patch
import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from agentx.config import settings
from agentx.schemas import ToolStatus
from agentx.tools.browser import browse_page


def _setup_mock_playwright():
    """Helper to construct a mock Playwright browser hierarchy."""
    mock_p = MagicMock()
    mock_browser = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()

    mock_p.chromium.launch.return_value = mock_browser
    mock_browser.new_context.return_value = mock_context
    mock_context.new_page.return_value = mock_page

    return mock_p, mock_browser, mock_context, mock_page


def test_browse_page_successful_extraction():
    """Verify browse_page extracts title, final_url, main_heading, and clean content."""
    mock_p, mock_browser, mock_context, mock_page = _setup_mock_playwright()
    mock_page.url = "https://example.com/articles/final"
    mock_page.title.return_value = "Example Article Title"

    mock_h1 = MagicMock()
    mock_h1.first = mock_h1
    mock_h1.count.return_value = 1
    mock_h1.text_content.return_value = "Main Article Heading"

    mock_body = MagicMock()
    mock_body.count.return_value = 1
    mock_body.inner_text.return_value = "Paragraph 1 of the article.\n\nParagraph 2 with crucial facts."

    mock_page.locator.side_effect = lambda selector: mock_h1 if selector == "h1" else mock_body
    mock_page.goto.return_value = MagicMock(status=200)

    with patch("agentx.tools.browser.sync_playwright") as mock_sync_pw:
        mock_sync_pw.return_value.start.return_value = mock_p
        with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "example.com")):
            result = browse_page(url="https://example.com/articles/start")

    assert result.status == ToolStatus.SUCCESS
    assert result.data["final_url"] == "https://example.com/articles/final"
    assert result.data["title"] == "Example Article Title"
    assert result.data["main_heading"] == "Main Article Heading"
    assert "Paragraph 1 of the article." in result.data["extracted_text"]
    assert "Paragraph 2 with crucial facts." in result.data["extracted_text"]
    assert result.data["status_code"] == 200
    assert not result.data["text_truncated"]

    # Tool message must contain meaningful visible text
    assert "Extracted Content:" in result.message
    assert "Paragraph 1 of the article." in result.message

    # DOM cleanup evaluate must have been called
    mock_page.evaluate.assert_called_once()
    eval_script = mock_page.evaluate.call_args[0][0]
    assert "cookie" in eval_script
    assert "script" in eval_script

    # Browser and playwright must be closed
    mock_browser.close.assert_called_once()
    mock_p.stop.assert_called_once()


def test_browse_page_text_truncation_reporting():
    """Verify browse_page caps text length and reports truncation."""
    mock_p, mock_browser, mock_context, mock_page = _setup_mock_playwright()
    mock_page.url = "https://example.com/long-page"
    mock_page.title.return_value = "Very Long Page"

    long_text = "Data point " * 1000  # 11000 chars, exceeds default 4000
    mock_body = MagicMock()
    mock_body.count.return_value = 1
    mock_body.inner_text.return_value = long_text

    mock_page.locator.side_effect = lambda selector: mock_body
    mock_page.goto.return_value = MagicMock(status=200)

    with patch("agentx.tools.browser.sync_playwright") as mock_sync_pw:
        mock_sync_pw.return_value.start.return_value = mock_p
        with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "example.com")):
            result = browse_page(url="https://example.com/long-page")

    assert result.status == ToolStatus.SUCCESS
    assert result.data["text_truncated"] is True
    assert len(result.data["extracted_text"]) <= settings.BROWSER_MAX_TEXT_LENGTH
    assert "Page content was truncated" in result.message


def test_browse_page_empty_content_returns_no_results():
    """Verify empty or whitespace-only page returns ToolStatus.NO_RESULTS."""
    mock_p, mock_browser, mock_context, mock_page = _setup_mock_playwright()
    mock_page.url = "https://example.com/blank"
    mock_page.title.return_value = "Blank Document"

    mock_body = MagicMock()
    mock_body.count.return_value = 1
    mock_body.inner_text.return_value = "   \n\t  \n  "

    mock_page.locator.side_effect = lambda selector: mock_body
    mock_page.goto.return_value = MagicMock(status=200)

    with patch("agentx.tools.browser.sync_playwright") as mock_sync_pw:
        mock_sync_pw.return_value.start.return_value = mock_p
        with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "example.com")):
            result = browse_page(url="https://example.com/blank")

    assert result.status == ToolStatus.NO_RESULTS
    assert "contained no readable text content" in result.message
    assert result.data["extracted_text"] == ""
    assert mock_browser.close.call_count == 1
    assert mock_p.stop.call_count == 1


def test_browse_page_navigation_timeout_handling():
    """Verify navigation timeout raises clean ToolResult.ERROR without crashing."""
    mock_p, mock_browser, mock_context, mock_page = _setup_mock_playwright()
    mock_page.goto.side_effect = PlaywrightTimeoutError("Navigation timeout 10000ms exceeded")

    with patch("agentx.tools.browser.sync_playwright") as mock_sync_pw:
        mock_sync_pw.return_value.start.return_value = mock_p
        with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "example.com")):
            result = browse_page(url="https://example.com/slow-site")

    assert result.status == ToolStatus.ERROR
    assert "Navigation timed out" in result.error
    assert mock_browser.close.call_count == 1
    assert mock_p.stop.call_count == 1


def test_browse_page_navigation_failure_handling():
    """Verify connection refusal or network error returns clean ToolResult.ERROR."""
    mock_p, mock_browser, mock_context, mock_page = _setup_mock_playwright()
    mock_page.goto.side_effect = Exception("Page.goto: net::ERR_CONNECTION_REFUSED at https://example.com/down")

    with patch("agentx.tools.browser.sync_playwright") as mock_sync_pw:
        mock_sync_pw.return_value.start.return_value = mock_p
        with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "example.com")):
            result = browse_page(url="https://example.com/down")

    assert result.status == ToolStatus.ERROR
    assert "Navigation failed" in result.error
    assert "ERR_CONNECTION_REFUSED" in result.error
    assert mock_browser.close.call_count == 1
    assert mock_p.stop.call_count == 1


def test_browse_page_ssrf_rejection():
    """Verify SSRF targets (e.g. metadata endpoint) are rejected before launch."""
    result = browse_page(url="http://169.254.169.254/latest/meta-data/")
    assert result.status == ToolStatus.ERROR
    assert "blocked for security reasons" in result.error.lower() or "ssrf" in result.error.lower()
