"""Comprehensive test suite for AgentX browser automation, Google Chrome integration, and SSRF security.

Covers all 16 reliability, lifecycle, security, and compatibility requirements:
1. Chrome channel selection and visible launch configuration
2. Chrome missing or launch failure detection and instruction
3. Successful navigation to a simple page
4. Navigation timeout handling
5. Slow or repeated DNS validation (caching and TTL)
6. Safe main-document navigation
7. Rejection of unsafe main-document URLs
8. Rejection of unsafe redirects
9. Blocking prohibited sub-resources without unnecessarily failing safe pages
10. SSRF protection for loopback, private, link-local, and metadata destinations
11. URL validation under DNS changes or cache expiry
12. Browser remains open according to the configured lifecycle (keep_open=True)
13. Graceful handling of user-closed browsers and tabs
14. Reliable cleanup without leaked processes or unhandled exceptions
15. YouTube search URL construction and encoding
16. Existing browser tool registration and argument compatibility
"""

import threading
import time
from unittest.mock import MagicMock, call, patch
import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from agentx.config import settings
from agentx.safety.url_validation import (
    clear_dns_cache,
    is_ip_allowed,
    parse_ip_literal,
    resolve_and_validate_hostname,
    validate_url_for_ssrf,
)
from agentx.schemas import BrowserArgs, ToolStatus, VisibleBrowserArgs
from agentx.tool_registry import create_default_registry
from agentx.tools.browser import (
    _visible_browser_manager,
    browse_page,
    cleanup_visible_browser,
    open_visible_browser,
)


@pytest.fixture(autouse=True)
def clean_browser_state():
    """Ensure a clean browser state and DNS cache before and after each test."""
    clear_dns_cache()
    cleanup_visible_browser()
    yield
    cleanup_visible_browser()
    clear_dns_cache()


# ==============================================================================
# Requirement 1: Chrome Channel Selection & Visible Launch Configuration
# ==============================================================================

def test_chrome_channel_selection_and_visible_launch_configuration():
    """Verify that open_visible_browser requests the installed Chrome channel and launches headed."""
    with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "93.184.216.34")):
        with patch("agentx.tools.browser.sync_playwright") as mock_pw_ctx:
            mock_pw = MagicMock()
            mock_pw_ctx.return_value.start.return_value = mock_pw
            mock_browser = MagicMock()
            mock_browser.is_connected.return_value = True
            mock_pw.chromium.launch.return_value = mock_browser

            mock_ctx = MagicMock()
            mock_browser.new_context.return_value = mock_ctx
            mock_page = MagicMock()
            mock_page.is_closed.return_value = False
            mock_ctx.new_page.return_value = mock_page

            mock_response = MagicMock()
            mock_response.status = 200
            mock_page.goto.return_value = mock_response
            mock_page.title.return_value = "Example Page"

            res = open_visible_browser("https://example.com", display_duration_s=0.01)

            assert res.status == ToolStatus.SUCCESS
            mock_pw.chromium.launch.assert_called_once_with(
                channel=settings.CHROME_CHANNEL,
                headless=False,
                args=["--no-sandbox", "--disable-setuid-sandbox"],
            )


# ==============================================================================
# Requirement 2: Chrome Missing or Launch Failure Reporting
# ==============================================================================

def test_chrome_missing_or_launch_failure_reports_clear_instruction():
    """Verify that when Chrome is missing or cannot launch, a clear user instruction is returned without silent Chromium fallback."""
    with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "93.184.216.34")):
        with patch("agentx.tools.browser.sync_playwright") as mock_pw_ctx:
            mock_pw = MagicMock()
            mock_pw_ctx.return_value.start.return_value = mock_pw
            mock_pw.chromium.launch.side_effect = Exception("Executable doesn't exist at C:\\Program Files\\Google\\Chrome\\chrome.exe")

            res = open_visible_browser("https://example.com")

            assert res.status == ToolStatus.ERROR
            assert "Google Chrome is not installed" in res.error or "CHROME_CHANNEL" in res.error
            assert "https://www.google.com/chrome" in res.error


# ==============================================================================
# Requirement 3: Successful Navigation to a Simple Page
# ==============================================================================

def test_successful_navigation_to_simple_page_mocked():
    """Verify that visible navigation correctly loads a simple page and extracts status and title."""
    with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "93.184.216.34")):
        with patch("agentx.tools.browser.sync_playwright") as mock_pw_ctx:
            mock_pw = MagicMock()
            mock_pw_ctx.return_value.start.return_value = mock_pw
            mock_browser = MagicMock()
            mock_browser.is_connected.return_value = True
            mock_pw.chromium.launch.return_value = mock_browser

            mock_ctx = MagicMock()
            mock_browser.new_context.return_value = mock_ctx
            mock_page = MagicMock()
            mock_page.is_closed.return_value = False
            mock_ctx.new_page.return_value = mock_page

            mock_response = MagicMock()
            mock_response.status = 200
            mock_page.goto.return_value = mock_response
            mock_page.title.return_value = "Example Domain"

            res = open_visible_browser("https://example.com", display_duration_s=0.01)

            assert res.status == ToolStatus.SUCCESS
            assert res.data["title"] == "Example Domain"
            assert res.data["status_code"] == 200
            assert "Opened Google Chrome at 'https://example.com'" in res.message


# ==============================================================================
# Requirement 4: Navigation Timeout Handling
# ==============================================================================

def test_navigation_timeout_handling_in_visible_browser():
    """Verify visible navigation timeouts return a clean ERROR without unhandled exceptions."""
    with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "93.184.216.34")):
        with patch("agentx.tools.browser.sync_playwright") as mock_pw_ctx:
            mock_pw = MagicMock()
            mock_pw_ctx.return_value.start.return_value = mock_pw
            mock_browser = MagicMock()
            mock_browser.is_connected.return_value = True
            mock_pw.chromium.launch.return_value = mock_browser

            mock_ctx = MagicMock()
            mock_browser.new_context.return_value = mock_ctx
            mock_page = MagicMock()
            mock_page.is_closed.return_value = False
            mock_ctx.new_page.return_value = mock_page

            mock_page.goto.side_effect = PlaywrightTimeoutError("Navigation timeout of 30000ms exceeded")

            res = open_visible_browser("https://slow.example.com", timeout_ms=5000)

            assert res.status == ToolStatus.ERROR
            assert "timed out after 5.0s" in res.error.lower()


# ==============================================================================
# Requirement 5: DNS Validation Caching & Rebinding Defense
# ==============================================================================

def test_dns_validation_caches_repeated_lookups_within_ttl():
    """Verify repeated DNS lookups for the same hostname are cached, eliminating redundant socket.getaddrinfo calls."""
    mock_addr = [(2, 1, 0, "", ("93.184.216.34", 443))]
    with patch("socket.getaddrinfo", return_value=mock_addr) as mock_dns:
        safe1, reason1, ip1 = validate_url_for_ssrf("https://cachetest.example.com/page1")
        assert safe1 is True
        assert ip1 == "93.184.216.34"
        assert mock_dns.call_count == 1

        # Second lookup for same host must hit cache and NOT invoke getaddrinfo
        safe2, reason2, ip2 = validate_url_for_ssrf("https://cachetest.example.com/subresource.js")
        assert safe2 is True
        assert ip2 == "93.184.216.34"
        assert mock_dns.call_count == 1


def test_dns_rebinding_dual_homed_defense():
    """Verify that if a domain resolves to multiple IPs and ANY IP is private/loopback, it fails closed."""
    mock_addr = [
        (2, 1, 0, "", ("93.184.216.34", 80)),
        (2, 1, 0, "", ("127.0.0.1", 80)),
    ]
    with patch("socket.getaddrinfo", return_value=mock_addr):
        safe, reason, ip = validate_url_for_ssrf("http://dualhomed.example.com")
        assert safe is False
        assert "loopback" in reason.lower() or "prohibited" in reason.lower()


# ==============================================================================
# Requirement 6 & 7: Main-Document Navigation (Safe Allowed, Unsafe Rejected)
# ==============================================================================

def test_safe_main_document_navigation_allowed():
    """Verify route interceptor allows legitimate main-document navigation requests."""
    with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "93.184.216.34")):
        with patch("agentx.tools.browser.sync_playwright") as mock_pw_ctx:
            mock_pw = MagicMock()
            mock_pw_ctx.return_value.start.return_value = mock_pw
            mock_browser = MagicMock()
            mock_browser.is_connected.return_value = True
            mock_pw.chromium.launch.return_value = mock_browser

            mock_ctx = MagicMock()
            mock_browser.new_context.return_value = mock_ctx
            mock_page = MagicMock()
            mock_page.is_closed.return_value = False
            mock_ctx.new_page.return_value = mock_page

            route_fn = None
            def fake_route(pattern, handler):
                nonlocal route_fn
                route_fn = handler
            mock_page.route.side_effect = fake_route

            def fake_goto(url, **kwargs):
                mock_route = MagicMock()
                mock_route.request.url = url
                mock_route.request.is_navigation_request.return_value = True
                route_fn(mock_route)
                mock_route.continue_.assert_called()
                return MagicMock(status=200)

            mock_page.goto.side_effect = fake_goto
            mock_page.title.return_value = "Safe Page"

            res = open_visible_browser("https://safe.example.com", display_duration_s=0.01)
            assert res.status == ToolStatus.SUCCESS


def test_rejection_of_unsafe_main_document_urls():
    """Verify pre-navigation and main-document navigation to prohibited IPs are rejected."""
    bad_urls = [
        "http://127.0.0.1:8080/admin",
        "http://192.168.1.1/setup",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]:3000/",
    ]
    for bad_url in bad_urls:
        res = open_visible_browser(bad_url)
        assert res.status == ToolStatus.ERROR
        assert "SSRF" in res.error or "Blocked" in res.error


# ==============================================================================
# Requirement 8: Rejection of Unsafe Redirects
# ==============================================================================

def test_rejection_of_unsafe_redirects_during_navigation():
    """Verify route interception halts and aborts navigation when redirected to a prohibited destination."""
    with patch("agentx.tools.browser.validate_url_for_ssrf") as mock_validate:
        def ssrf_checker(url):
            if "169.254.169.254" in url or "metadata" in url:
                return False, "Access to cloud metadata address is prohibited.", "169.254.169.254"
            return True, None, "93.184.216.34"
        mock_validate.side_effect = ssrf_checker

        with patch("agentx.tools.browser.sync_playwright") as mock_pw_ctx:
            mock_pw = MagicMock()
            mock_pw_ctx.return_value.start.return_value = mock_pw
            mock_browser = MagicMock()
            mock_browser.is_connected.return_value = True
            mock_pw.chromium.launch.return_value = mock_browser

            mock_ctx = MagicMock()
            mock_browser.new_context.return_value = mock_ctx
            mock_page = MagicMock()
            mock_page.is_closed.return_value = False
            mock_ctx.new_page.return_value = mock_page

            route_fn = None
            def fake_route(pattern, handler):
                nonlocal route_fn
                route_fn = handler
            mock_page.route.side_effect = fake_route

            def fake_goto(url, **kwargs):
                mock_route = MagicMock()
                mock_route.request.url = "http://169.254.169.254/latest/meta-data"
                mock_route.request.is_navigation_request.return_value = True
                route_fn(mock_route)
                mock_route.abort.assert_called_with("blockedbyclient")
                raise RuntimeError("net::ERR_BLOCKED_BY_CLIENT")

            mock_page.goto.side_effect = fake_goto

            res = open_visible_browser("https://open-redirector.com")
            assert res.status == ToolStatus.ERROR
            assert "Security violation" in res.error or "blocked" in res.error.lower()


# ==============================================================================
# Requirement 9: Blocking Prohibited Sub-resources Without Failing Safe Pages
# ==============================================================================

def test_blocking_prohibited_subresources_without_failing_page():
    """Verify that if a 3rd-party sub-resource (tracker/image) resolves to a private IP, it is aborted without failing the main page."""
    with patch("agentx.tools.browser.validate_url_for_ssrf") as mock_validate:
        def ssrf_checker(url):
            if "bad-pixel.com" in url or "10.0.0.1" in url:
                return False, "Access to private network address is prohibited.", "10.0.0.1"
            return True, None, "93.184.216.34"
        mock_validate.side_effect = ssrf_checker

        with patch("agentx.tools.browser.sync_playwright") as mock_pw_ctx:
            mock_pw = MagicMock()
            mock_pw_ctx.return_value.start.return_value = mock_pw
            mock_browser = MagicMock()
            mock_browser.is_connected.return_value = True
            mock_pw.chromium.launch.return_value = mock_browser

            mock_ctx = MagicMock()
            mock_browser.new_context.return_value = mock_ctx
            mock_page = MagicMock()
            mock_page.is_closed.return_value = False
            mock_ctx.new_page.return_value = mock_page

            route_fn = None
            def fake_route(pattern, handler):
                nonlocal route_fn
                route_fn = handler
            mock_page.route.side_effect = fake_route

            def fake_goto(url, **kwargs):
                # Simulate subresource request (image/beacon)
                mock_sub_route = MagicMock()
                mock_sub_route.request.url = "http://bad-pixel.com/beacon.png"
                mock_sub_route.request.is_navigation_request.return_value = False
                mock_sub_route.request.resource_type = "image"
                route_fn(mock_sub_route)
                mock_sub_route.abort.assert_called_with("blockedbyclient")
                return MagicMock(status=200)

            mock_page.goto.side_effect = fake_goto
            mock_page.title.return_value = "Clean Safe Page With Blocked Pixel"

            res = open_visible_browser("https://safe-site.com", display_duration_s=0.01)

            # Crucial assertion: the page must SUCCEED, not fail!
            assert res.status == ToolStatus.SUCCESS
            assert res.data["title"] == "Clean Safe Page With Blocked Pixel"


# ==============================================================================
# Requirement 10: SSRF Protection for Loopback, Private, Link-Local & Metadata
# ==============================================================================

def test_comprehensive_ssrf_protection_destinations():
    """Verify loopback, RFC1918 private, link-local, carrier-grade NAT, and cloud metadata are blocked."""
    prohibited_destinations = [
        # Loopback
        ("127.0.0.1", "loopback"),
        ("127.1.2.3", "loopback"),
        ("::1", "loopback"),
        # Private LAN
        ("10.0.0.1", "private"),
        ("172.16.0.1", "private"),
        ("192.168.1.1", "private"),
        ("fc00::1", "private"),
        ("fd00::1", "private"),
        # Link-local & Multicast
        ("169.254.1.1", "link-local"),
        ("fe80::1", "link-local"),
        ("224.0.0.1", "multicast"),
        # Cloud Metadata
        ("169.254.169.254", "cloud metadata"),
        ("fd00:ec2::254", "cloud metadata"),
        # IPv4-mapped IPv6
        ("::ffff:127.0.0.1", "ipv4-mapped"),
        ("::ffff:10.0.0.1", "ipv4-mapped"),
        # NAT64 WKP prohibited
        ("64:ff9b::127.0.0.1", "nat64"),
        ("64:ff9b::192.168.1.1", "nat64"),
    ]
    for ip, label in prohibited_destinations:
        safe, reason = is_ip_allowed(ip)
        assert safe is False, f"Prohibited IP {ip} ({label}) was incorrectly permitted!"
        assert reason is not None


# ==============================================================================
# Requirement 11: URL Validation Under DNS Changes / Cache Expiry
# ==============================================================================

def test_dns_validation_under_cache_clear_or_ip_change():
    """Verify that when cache expires or is cleared, new DNS resolution reflects the changed destination."""
    with patch("socket.getaddrinfo") as mock_dns:
        # Initial resolution: public IP
        mock_dns.return_value = [(2, 1, 0, "", ("93.184.216.34", 443))]
        safe1, _, _ = validate_url_for_ssrf("https://dynamic.example.com")
        assert safe1 is True

        # DNS changes to private IP (simulating DNS rebinding or dynamic update)
        mock_dns.return_value = [(2, 1, 0, "", ("192.168.1.50", 443))]
        # Clear cache to simulate TTL expiry
        clear_dns_cache()

        safe2, reason2, ip2 = validate_url_for_ssrf("https://dynamic.example.com")
        assert safe2 is False
        assert "private" in reason2.lower()


# ==============================================================================
# Requirement 12: Browser Remains Open According to Configured Lifecycle
# ==============================================================================

def test_visible_browser_remains_open_when_keep_open_is_true():
    """Verify visible browser remains open when keep_open=True and display_duration_s is omitted."""
    with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "93.184.216.34")):
        with patch("agentx.tools.browser.sync_playwright") as mock_pw_ctx:
            mock_pw = MagicMock()
            mock_pw_ctx.return_value.start.return_value = mock_pw
            mock_browser = MagicMock()
            mock_browser.is_connected.return_value = True
            mock_pw.chromium.launch.return_value = mock_browser

            mock_ctx = MagicMock()
            mock_browser.new_context.return_value = mock_ctx
            mock_page = MagicMock()
            mock_page.is_closed.return_value = False
            mock_ctx.new_page.return_value = mock_page

            mock_response = MagicMock()
            mock_response.status = 200
            mock_page.goto.return_value = mock_response
            mock_page.title.return_value = "Persistent Page"

            res = open_visible_browser("https://example.com", keep_open=True, display_duration_s=None)

            assert res.status == ToolStatus.SUCCESS
            assert res.data["browser_kept_open"] is True

            # The page and browser must NOT have been closed!
            mock_page.close.assert_not_called()
            mock_browser.close.assert_not_called()
            mock_pw.stop.assert_not_called()


# ==============================================================================
# Requirement 13: Graceful Handling of User-Closed Browsers and Tabs
# ==============================================================================

def test_graceful_handling_of_user_closed_browser():
    """Verify that if a user closes the Chrome window, the next invocation detects it and re-launches cleanly."""
    with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "93.184.216.34")):
        with patch("agentx.tools.browser.sync_playwright") as mock_pw_ctx:
            mock_pw = MagicMock()
            mock_pw_ctx.return_value.start.return_value = mock_pw

            mock_browser1 = MagicMock()
            mock_browser1.is_connected.return_value = True
            mock_page1 = MagicMock()
            mock_page1.is_closed.return_value = False

            mock_browser2 = MagicMock()
            mock_browser2.is_connected.return_value = True
            mock_page2 = MagicMock()
            mock_page2.is_closed.return_value = False

            mock_pw.chromium.launch.side_effect = [mock_browser1, mock_browser2]

            mock_ctx1 = MagicMock()
            mock_ctx1.new_page.return_value = mock_page1
            mock_browser1.new_context.return_value = mock_ctx1

            mock_ctx2 = MagicMock()
            mock_ctx2.new_page.return_value = mock_page2
            mock_browser2.new_context.return_value = mock_ctx2

            # Session 1: opens successfully
            res1 = open_visible_browser("https://example.com", keep_open=True, display_duration_s=None)
            assert res1.status == ToolStatus.SUCCESS

            # Simulate user closing the Chrome window/tab
            mock_page1.is_closed.return_value = True
            mock_browser1.is_connected.return_value = False

            # Session 2: detects closed window and launches a fresh instance without crashing
            res2 = open_visible_browser("https://example.com", keep_open=True, display_duration_s=None)
            assert res2.status == ToolStatus.SUCCESS
            assert mock_pw.chromium.launch.call_count == 2


# ==============================================================================
# Requirement 14: Reliable Cleanup Without Leaked Processes
# ==============================================================================

def test_reliable_cleanup_without_leaks_or_exceptions():
    """Verify cleanup_visible_browser safely closes handles without raising unhandled exceptions even if called multiple times."""
    with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "93.184.216.34")):
        with patch("agentx.tools.browser.sync_playwright") as mock_pw_ctx:
            mock_pw = MagicMock()
            mock_pw_ctx.return_value.start.return_value = mock_pw
            mock_browser = MagicMock()
            mock_browser.is_connected.return_value = True
            mock_pw.chromium.launch.return_value = mock_browser

            mock_ctx = MagicMock()
            mock_browser.new_context.return_value = mock_ctx
            mock_page = MagicMock()
            mock_page.is_closed.return_value = False
            mock_ctx.new_page.return_value = mock_page

            res = open_visible_browser("https://example.com", keep_open=True, display_duration_s=None)
            assert res.status == ToolStatus.SUCCESS

            # Explicit cleanup
            cleanup_visible_browser()
            mock_page.close.assert_called_once()
            mock_context_close = mock_ctx.close
            mock_context_close.assert_called_once()
            mock_browser.close.assert_called_once()
            mock_pw.stop.assert_called_once()

            # Calling cleanup again should be a safe no-op
            cleanup_visible_browser()


# ==============================================================================
# Requirement 15: YouTube Search URL Construction, Encoding & Playback Honesty
# ==============================================================================

def test_youtube_search_url_construction_encoding_and_playback_honesty():
    """Verify YouTube query formatting properly percent-encodes search terms and honestly reports unverified playback."""
    with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "142.250.190.46")):
        with patch("agentx.tools.browser.sync_playwright") as mock_pw_ctx:
            mock_pw = MagicMock()
            mock_pw_ctx.return_value.start.return_value = mock_pw
            mock_browser = MagicMock()
            mock_browser.is_connected.return_value = True
            mock_pw.chromium.launch.return_value = mock_browser

            mock_ctx = MagicMock()
            mock_browser.new_context.return_value = mock_ctx
            mock_page = MagicMock()
            mock_page.is_closed.return_value = False
            mock_ctx.new_page.return_value = mock_page

            mock_response = MagicMock()
            mock_response.status = 200
            mock_page.goto.return_value = mock_response
            mock_page.title.return_value = "lofi music - YouTube"

            # Mock video element present
            mock_video_locator = MagicMock()
            mock_video_locator.count.return_value = 1
            mock_page.locator.return_value = mock_video_locator

            # Query with special characters requiring percent encoding
            res = open_visible_browser(
                url="https://www.youtube.com",
                search_query="lofi music & chill #1",
                display_duration_s=0.01,
            )

            assert res.status == ToolStatus.SUCCESS
            assert "search_query=lofi+music+%26+chill+%231" in res.data["url"]
            assert res.data["is_video_destination"] is True
            assert res.data["video_element_detected"] is True
            # Crucial: video playback must be honestly reported as unverified
            assert res.data["video_playback_verified"] is False
            assert "video_playback_verified" in res.evidence
            assert ("Search results page remained open" in res.message or "cannot be autonomously verified" in res.message or "Playback verification note" in res.message)



# ==============================================================================
# Requirement 16: Tool Registration, Schemas & Bare URL Normalization
# ==============================================================================

def test_browser_tool_registration_and_schemas():
    """Verify that browser and open_visible_browser are registered in default registry and schemas accept bare URLs."""
    registry = create_default_registry()
    tool_names = registry.list_tool_names()

    assert "browser" in tool_names
    assert "open_visible_browser" in tool_names

    # Test Bare URL normalization in BrowserArgs
    b_args = BrowserArgs(url="example.com")
    assert b_args.url == "https://example.com"

    # Test Bare URL normalization in VisibleBrowserArgs
    v_args = VisibleBrowserArgs(url="youtube.com", search_query="lofi music")
    assert v_args.url == "https://youtube.com"
    assert v_args.keep_open is True

    # Test non-HTTP schemes are rejected
    with pytest.raises(ValueError):
        BrowserArgs(url="javascript:alert(1)")
    with pytest.raises(ValueError):
        VisibleBrowserArgs(url="file:///etc/passwd")


# ==============================================================================
# Requirement 17: Dedicated Worker Thread & Cross-Thread Safety Regression Tests
# ==============================================================================

def test_consecutive_browser_calls_from_different_calling_threads():
    """Verify consecutive browser navigation calls from different threads succeed without 'Cannot switch to a different thread'."""
    with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "93.184.216.34")):
        with patch("agentx.tools.browser.sync_playwright") as mock_pw_ctx:
            mock_pw = MagicMock()
            mock_pw_ctx.return_value.start.return_value = mock_pw
            mock_browser = MagicMock()
            mock_browser.is_connected.return_value = True
            mock_pw.chromium.launch.return_value = mock_browser

            mock_ctx = MagicMock()
            mock_browser.new_context.return_value = mock_ctx
            mock_page = MagicMock()
            mock_page.is_closed.return_value = False
            mock_ctx.new_page.return_value = mock_page

            mock_response = MagicMock(status=200)
            mock_page.goto.return_value = mock_response
            mock_page.title.return_value = "Thread Test Page"

            results = []
            errors = []

            def worker_run(url: str):
                try:
                    res = open_visible_browser(url, keep_open=True)
                    results.append(res)
                except Exception as e:
                    errors.append(e)

            # Thread 1: first navigation
            t1 = threading.Thread(target=worker_run, args=("https://example.com/one",))
            t1.start()
            t1.join()

            # Thread 2: second navigation on a completely different calling thread
            t2 = threading.Thread(target=worker_run, args=("https://example.com/two",))
            t2.start()
            t2.join()

            assert len(errors) == 0, f"Encountered unexpected exceptions: {errors}"
            assert len(results) == 2
            assert results[0].status == ToolStatus.SUCCESS
            assert results[1].status == ToolStatus.SUCCESS
            assert results[0].error is None
            assert results[1].error is None


def test_multiple_sequential_navigation_requests():
    """Verify multiple sequential navigation requests succeed and reuse persistent Chrome session."""
    with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "93.184.216.34")):
        with patch("agentx.tools.browser.sync_playwright") as mock_pw_ctx:
            mock_pw = MagicMock()
            mock_pw_ctx.return_value.start.return_value = mock_pw
            mock_browser = MagicMock()
            mock_browser.is_connected.return_value = True
            mock_pw.chromium.launch.return_value = mock_browser

            mock_ctx = MagicMock()
            mock_browser.new_context.return_value = mock_ctx
            mock_page = MagicMock()
            mock_page.is_closed.return_value = False
            mock_ctx.new_page.return_value = mock_page

            mock_response = MagicMock(status=200)
            mock_page.goto.return_value = mock_response
            mock_page.title.return_value = "Sequential Page"

            urls = ["https://example.com/1", "https://example.com/2", "https://example.com/3"]
            for url in urls:
                res = open_visible_browser(url, keep_open=True)
                assert res.status == ToolStatus.SUCCESS
                assert res.data["browser_kept_open"] is True

            # Launched Chrome only once across all 3 navigations
            assert mock_pw.chromium.launch.call_count == 1
            assert mock_page.goto.call_count == 3


def test_concurrent_browser_requests_serialization():
    """Verify concurrent requests from multiple threads are serialized safely without race conditions."""
    with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "93.184.216.34")):
        with patch("agentx.tools.browser.sync_playwright") as mock_pw_ctx:
            mock_pw = MagicMock()
            mock_pw_ctx.return_value.start.return_value = mock_pw
            mock_browser = MagicMock()
            mock_browser.is_connected.return_value = True
            mock_pw.chromium.launch.return_value = mock_browser

            mock_ctx = MagicMock()
            mock_browser.new_context.return_value = mock_ctx
            mock_page = MagicMock()
            mock_page.is_closed.return_value = False
            mock_ctx.new_page.return_value = mock_page

            mock_response = MagicMock(status=200)
            mock_page.goto.return_value = mock_response
            mock_page.title.return_value = "Concurrent Page"

            results = []
            threads = []

            def worker_task(i: int):
                res = open_visible_browser(f"https://example.com/concurrent/{i}", keep_open=True)
                results.append(res)

            for i in range(4):
                t = threading.Thread(target=worker_task, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            assert len(results) == 4
            for r in results:
                assert r.status == ToolStatus.SUCCESS
                assert r.error is None


def test_browser_closure_and_session_recovery():
    """Verify that if a user closes the Chrome window, the worker detects it and recovers with a fresh window."""
    with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "93.184.216.34")):
        with patch("agentx.tools.browser.sync_playwright") as mock_pw_ctx:
            mock_pw = MagicMock()
            mock_pw_ctx.return_value.start.return_value = mock_pw

            mock_browser1 = MagicMock()
            mock_browser1.is_connected.return_value = True
            mock_page1 = MagicMock()
            mock_page1.is_closed.return_value = False

            mock_browser2 = MagicMock()
            mock_browser2.is_connected.return_value = True
            mock_page2 = MagicMock()
            mock_page2.is_closed.return_value = False

            mock_pw.chromium.launch.side_effect = [mock_browser1, mock_browser2]

            mock_ctx1 = MagicMock()
            mock_ctx1.new_page.return_value = mock_page1
            mock_browser1.new_context.return_value = mock_ctx1

            mock_ctx2 = MagicMock()
            mock_ctx2.new_page.return_value = mock_page2
            mock_browser2.new_context.return_value = mock_ctx2

            # Request 1: establishes active session
            res1 = open_visible_browser("https://example.com/1", keep_open=True)
            assert res1.status == ToolStatus.SUCCESS

            # Simulate user closing the window
            mock_page1.is_closed.return_value = True
            mock_browser1.is_connected.return_value = False

            # Request 2: worker detects closed window and launches new window cleanly
            res2 = open_visible_browser("https://example.com/2", keep_open=True)
            assert res2.status == ToolStatus.SUCCESS
            assert mock_pw.chromium.launch.call_count == 2


def test_navigation_timeout_and_worker_cleanup():
    """Verify navigation timeouts on the worker thread return a clean ERROR without deadlocks."""
    with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "93.184.216.34")):
        with patch("agentx.tools.browser.sync_playwright") as mock_pw_ctx:
            mock_pw = MagicMock()
            mock_pw_ctx.return_value.start.return_value = mock_pw
            mock_browser = MagicMock()
            mock_browser.is_connected.return_value = True
            mock_pw.chromium.launch.return_value = mock_browser

            mock_ctx = MagicMock()
            mock_browser.new_context.return_value = mock_ctx
            mock_page = MagicMock()
            mock_page.is_closed.return_value = False
            mock_ctx.new_page.return_value = mock_page

            mock_page.goto.side_effect = PlaywrightTimeoutError("Navigation timeout of 5000ms exceeded")

            res = open_visible_browser("https://timeout.example.com", timeout_ms=5000)
            assert res.status == ToolStatus.ERROR
            assert "timed out after 5.0s" in res.error.lower()


def test_application_shutdown_worker_cleanup():
    """Verify that cleanup_visible_browser cleanly terminates the worker thread and releases Playwright handles."""
    with patch("agentx.tools.browser.validate_url_for_ssrf", return_value=(True, None, "93.184.216.34")):
        with patch("agentx.tools.browser.sync_playwright") as mock_pw_ctx:
            mock_pw = MagicMock()
            mock_pw_ctx.return_value.start.return_value = mock_pw
            mock_browser = MagicMock()
            mock_browser.is_connected.return_value = True
            mock_pw.chromium.launch.return_value = mock_browser

            mock_ctx = MagicMock()
            mock_browser.new_context.return_value = mock_ctx
            mock_page = MagicMock()
            mock_page.is_closed.return_value = False
            mock_ctx.new_page.return_value = mock_page

            res = open_visible_browser("https://example.com", keep_open=True)
            assert res.status == ToolStatus.SUCCESS
            assert _visible_browser_manager.is_active() is True

            # Full application shutdown cleanup
            cleanup_visible_browser(stop_worker=True)

            assert _visible_browser_manager.is_active() is False
            mock_page.close.assert_called_once()
            mock_ctx.close.assert_called_once()
            mock_browser.close.assert_called_once()
            mock_pw.stop.assert_called_once()

