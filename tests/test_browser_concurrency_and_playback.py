"""Unit and integration tests for browser thread safety, concurrency serialization, and YouTube playback states."""

import concurrent.futures
import threading
import time
from unittest.mock import MagicMock, patch
import pytest

from agentx.schemas import ToolStatus
from agentx.tools.browser import VisibleBrowserManager, open_visible_browser


def test_browser_operations_serialized_on_dedicated_worker_thread():
    """Verify consecutive browser calls from multiple caller threads execute strictly on AgentX-BrowserWorker."""
    worker_threads_observed = set()

    def fake_do_navigate(self, **kwargs):
        # Record the thread where execution actually occurs
        current_t = threading.current_thread()
        worker_threads_observed.add(current_t.name)
        return MagicMock(status=ToolStatus.SUCCESS, data={"url": kwargs.get("clean_url")})

    mgr = VisibleBrowserManager()

    with patch.object(VisibleBrowserManager, "_do_navigate", fake_do_navigate):
        caller_threads = []

        def caller_task(url: str):
            caller_threads.append(threading.current_thread().name)
            return mgr.navigate(
                clean_url=url,
                timeout_ms=1000,
                channel="chrome",
                keep_open=True,
                display_duration_s=None,
                is_video_site=False,
                video_note="",
            )

        # Call from 3 distinct caller threads (simulating different Textual @work threads)
        t1 = threading.Thread(target=caller_task, args=("https://example.com/page1",), name="TextualWorker-1")
        t2 = threading.Thread(target=caller_task, args=("https://example.com/page2",), name="TextualWorker-2")
        t3 = threading.Thread(target=caller_task, args=("https://example.com/page3",), name="TextualWorker-3")

        t1.start()
        t1.join()
        t2.start()
        t2.join()
        t3.start()
        t3.join()

    # 3 distinct caller threads invoked the manager
    assert len(set(caller_threads)) == 3
    # BUT all execution was strictly serialized on the dedicated AgentX-BrowserWorker thread
    assert worker_threads_observed == {"AgentX-BrowserWorker"}

    mgr.cleanup()


def test_browser_concurrent_requests_handled_safely():
    """Verify multiple concurrent requests from thread pool do not deadlock or raise thread-switch errors."""
    mgr = VisibleBrowserManager()

    def fake_do_navigate(self, **kwargs):
        time.sleep(0.05)
        return MagicMock(status=ToolStatus.SUCCESS, data={"url": kwargs.get("clean_url")})

    with patch.object(VisibleBrowserManager, "_do_navigate", fake_do_navigate):
        urls = [f"https://example.com/item/{i}" for i in range(5)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(
                    mgr.navigate,
                    clean_url=u,
                    timeout_ms=2000,
                    channel="chrome",
                    keep_open=True,
                    display_duration_s=None,
                    is_video_site=False,
                    video_note="",
                )
                for u in urls
            ]
            results = [f.result(timeout=5) for f in futures]

    assert len(results) == 5
    for r in results:
        assert r.status == ToolStatus.SUCCESS

    mgr.cleanup()


def test_browser_recovery_after_window_close():
    """Verify that if a user closes the browser window, manager detects inactivity and relaunches cleanly."""
    mgr = VisibleBrowserManager()

    # Mock an open page that becomes closed
    mock_browser = MagicMock()
    mock_browser.is_connected.return_value = True
    mock_page = MagicMock()
    mock_page.is_closed.return_value = False

    mgr._browser = mock_browser
    mgr._page = mock_page

    assert mgr._is_active_internal() is True

    # User closes page
    mock_page.is_closed.return_value = True
    assert mgr._is_active_internal() is False


def test_youtube_playback_verification_success_state():
    """Verify YouTube interaction workflow identifies video, navigates to watch page, and verifies advancing playback."""
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

            # Initial URL is search results
            mock_page.url = "https://www.youtube.com/results?search_query=madan+gowri"
            mock_page.title.return_value = "madan gowri - YouTube"

            # Mock finding organic video title locator on search page
            mock_video_link = MagicMock()
            mock_video_link.first = mock_video_link
            mock_video_link.count.return_value = 1
            mock_video_link.is_visible.return_value = True
            mock_video_link.get_attribute.side_effect = lambda attr: "/watch?v=sample123" if attr == "href" else "Madan Gowri Latest Video"
            mock_video_link.inner_text.return_value = "Madan Gowri Latest Video"

            def mock_locator(sel):
                if "video-title" in sel or "/watch" in sel:
                    return mock_video_link
                # Consent or dismiss buttons
                m = MagicMock()
                m.count.return_value = 0
                return m

            mock_page.locator.side_effect = mock_locator

            # When clicked, page transitions to watch page
            def fake_click(timeout=None):
                mock_page.url = "https://www.youtube.com/watch?v=sample123"
            mock_video_link.click.side_effect = fake_click

            # Mock observable video playback advancing from 1.0s to 3.2s
            eval_metrics = [
                {"currentTime": 1.0, "duration": 600.0, "paused": False, "readyState": 4},
                {"currentTime": 3.2, "duration": 600.0, "paused": False, "readyState": 4},
            ]
            eval_idx = {"count": 0}

            def fake_evaluate(script):
                if "currentTime" in script:
                    val = eval_metrics[eval_idx["count"] % 2]
                    eval_idx["count"] += 1
                    return val
                return True

            mock_page.evaluate.side_effect = fake_evaluate

            res = open_visible_browser(
                url="https://www.youtube.com",
                search_query="madan gowri",
                display_duration_s=0.01,
            )

            assert res.status == ToolStatus.SUCCESS
            assert res.data["search_results_opened"] is True
            assert res.data["video_page_opened"] is True
            assert res.data["playback_attempted"] is True
            assert res.data["video_playback_verified"] is True
            assert res.data["playback_evidence"]["is_playing"] is True
            assert res.data["playback_evidence"]["verified_time_s"] == 3.2
            assert "verified video playback" in res.message


def test_youtube_playback_restricted_honesty():
    """Verify YouTube interaction honestly reports restriction when video remains paused under autoplay policies."""
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
            mock_page.url = "https://www.youtube.com/watch?v=sample123"
            mock_page.title.return_value = "Educational Lecture - YouTube"

            mock_loc = MagicMock()
            mock_loc.count.return_value = 0
            mock_page.locator.return_value = mock_loc

            # Video currentTime stays at 0.0 and paused=True
            def fake_evaluate(script):
                if "currentTime" in script:
                    return {"currentTime": 0.0, "duration": 300.0, "paused": True, "readyState": 4}
                return True

            mock_page.evaluate.side_effect = fake_evaluate

            res = open_visible_browser(
                url="https://www.youtube.com/watch?v=sample123",
                display_duration_s=0.01,
            )

            assert res.status == ToolStatus.SUCCESS
            assert res.data["video_page_opened"] is True
            assert res.data["playback_attempted"] is True
            assert res.data["video_playback_verified"] is False
            assert "autoplay policy restricted" in res.data["limitation_note"]
            assert "Playback verification note" in res.message
