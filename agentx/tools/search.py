"""General-purpose Web Search Discovery tool for AgentX.

Provides dynamic, real-time web source discovery for current world facts,
commodity and product prices, news, and unfamiliar topics.

Supports:
1. DuckDuckGo Search (default, zero configuration/API key required).
2. Tavily Search API (optional, via TAVILY_API_KEY).
3. SerpAPI (optional, via SERPAPI_API_KEY).

Security & Safety:
- Strict SSRF defense on all discovered URLs before returning them to the agent.
- Safe argument validation via Pydantic WebSearchArgs.
- Never logs, outputs, or exposes secret API tokens.
- Honest error reporting on network or provider failures.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from agentx.config import settings
from agentx.safety.url_validation import validate_url_for_ssrf
from agentx.schemas import ToolResult, ToolStatus

logger = logging.getLogger("agentx.tools.search")

# Regex to detect publication dates or relative recency in search snippets
_DATE_REGEX = re.compile(
    r"\b(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|\d{4}-\d{2}-\d{2}|\d+\s+(?:hours?|days?|weeks?|months?|years?)\s+ago)\b",
    re.IGNORECASE,
)


def _extract_date_snippet(text: str) -> Optional[str]:
    """Attempt to extract an observable publication date or relative timestamp from snippet."""
    if not text:
        return None
    match = _DATE_REGEX.search(text)
    if match:
        return match.group(0).strip()
    return None


# Stopwords filtered during semantic normalization
_STOPWORDS = {
    "what", "is", "the", "of", "in", "for", "a", "an", "and", "or", "to",
    "on", "at", "by", "with", "from", "about", "tell", "me", "can", "you",
    "please", "find", "show", "give", "how", "much", "does", "cost", "rate",
    "price", "prices", "inr", "rupee", "rupees", "rs", "online"
}

_STALE_YEAR_REGEX = re.compile(r"\b(201\d|202[0-5])\b")
_CONVERSATIONAL_PREFIX_REGEX = re.compile(
    r"^(?:what\s+is\s+the|tell\s+me\s+the|can\s+you\s+find|how\s+much\s+is|what\s+are\s+the|show\s+me|find\s+the)\s+",
    re.IGNORECASE,
)


def normalize_query_semantics(query: str) -> str:
    """Normalize query into sorted keywords canonical form to detect semantic equivalence."""
    if not query:
        return ""
    cleaned = re.sub(r"[^\w\s]", " ", query.lower())
    tokens = [w for w in cleaned.split() if w and w not in _STOPWORDS]
    if not tokens:
        tokens = cleaned.split()
    return " ".join(sorted(tokens))


def are_queries_equivalent(q1: str, q2: str) -> bool:
    """Check if two search queries are semantically equivalent."""
    if not q1 or not q2:
        return False
    return normalize_query_semantics(q1) == normalize_query_semantics(q2)


_HISTORICAL_KEYWORDS = {
    "history", "historical", "timeline", "archive", "founded", "released",
    "launched", "invented", "born", "died", "won", "winner", "held", "happened",
    "occurred", "during", "past", "was", "were"
}


def is_historical_query(query: str) -> bool:
    """Check if query is asking about a past/historical event rather than current state."""
    q_lower = query.lower()
    words = set(re.findall(r"\b[a-z]+\b", q_lower))
    if words & _HISTORICAL_KEYWORDS:
        if not any(w in words for w in ("current", "today", "now", "latest")):
            return True
        if "history" in words or "historical" in words:
            return True
    if re.search(r"\b(?:in|during|of|back\s+in)\s+(?:19\d\d|20[0-2]\d)\b", q_lower):
        if not any(w in words for w in ("current", "today", "now", "latest")):
            return True
    return False


def sanitize_search_query(query: str) -> str:
    """Sanitize search query by removing filler and updating stale years for current inquiries."""
    if not query:
        return ""
    q = query.strip()
    # Remove conversational filler prefixes
    q = _CONVERSATIONAL_PREFIX_REGEX.sub("", q).strip()

    # Do not rewrite historical date queries
    if not is_historical_query(q):
        current_year_str = str(settings.CURRENT_YEAR)
        q = _STALE_YEAR_REGEX.sub(current_year_str, q)

    # Collapse whitespace
    q = re.sub(r"\s+", " ", q).strip()
    return q


def generate_query_alternatives(query: str) -> List[str]:
    """Generate useful diversified query alternatives without stale years."""
    sanitized = sanitize_search_query(query)
    current_year = str(settings.CURRENT_YEAR)
    alternatives = []

    # Alternative 1: Current-year targeted
    if current_year not in sanitized:
        alternatives.append(f"{sanitized} {current_year}")

    # Alternative 2: Online commerce focused
    if "online" not in sanitized.lower():
        alternatives.append(f"{sanitized} online")

    # Alternative 3: Broad search if query contains specific capacities/types
    has_specifics = bool(re.search(r"\b(\d+gb|ddr\d|kit|dimm|sodimm)\b", sanitized, re.IGNORECASE))
    if has_specifics:
        broad = re.sub(r"\b(\d+gb|ddr\d|kit|dimm|sodimm)\b", "", sanitized, flags=re.IGNORECASE)
        broad = re.sub(r"\s+", " ", broad).strip()
        if broad and broad != sanitized:
            if current_year not in broad:
                alternatives.append(f"{broad} {current_year}")
            else:
                alternatives.append(broad)

    return alternatives


def _search_duckduckgo(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Execute DuckDuckGo text search using the installed ddgs / duckduckgo_search library."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise RuntimeError("DuckDuckGo search package ('duckduckgo_search' or 'ddgs') is not installed.")

    raw_results = []
    clean_q = sanitize_search_query(query)

    try:
        with DDGS() as ddgs:
            generator = ddgs.text(clean_q, max_results=max_results)
            if generator:
                raw_results = list(generator)
    except Exception as e:
        logger.warning("DuckDuckGo primary search error on '%s': %s", clean_q, e)
        raw_results = []

    # If specific query returned 0 results, try diversified broader alternative
    if not raw_results:
        alternatives = generate_query_alternatives(clean_q)
        for alt_q in alternatives[:2]:
            try:
                with DDGS() as ddgs:
                    gen = ddgs.text(alt_q, max_results=max_results)
                    if gen:
                        res_list = list(gen)
                        if res_list:
                            raw_results = res_list
                            logger.info("DuckDuckGo diversified query '%s' succeeded with %d results.", alt_q, len(res_list))
                            break
            except Exception:
                continue

    formatted = []
    for r in raw_results:
        url = r.get("href") or r.get("url") or r.get("link")
        title = r.get("title") or "Untitled"
        body = r.get("body") or r.get("snippet") or ""
        if url:
            formatted.append({
                "title": title.strip(),
                "url": url.strip(),
                "snippet": body.strip(),
                "date": _extract_date_snippet(body),
            })
    return formatted


def _search_tavily(query: str, api_key: str, max_results: int) -> List[Dict[str, Any]]:
    """Execute Tavily Search API request."""
    payload = json.dumps({
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
    }).encode("utf-8")

    req = Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "AgentX/2.4"},
        method="POST",
    )
    with urlopen(req, timeout=settings.SEARCH_TIMEOUT_SECONDS) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results = []
    for item in data.get("results", []):
        url = item.get("url")
        if url:
            body = item.get("content") or ""
            results.append({
                "title": item.get("title", "Untitled").strip(),
                "url": url.strip(),
                "snippet": body.strip(),
                "date": item.get("published_date") or _extract_date_snippet(body),
            })
    return results


def _search_serpapi(query: str, api_key: str, max_results: int) -> List[Dict[str, Any]]:
    """Execute SerpAPI Google Search request."""
    params = urlencode({
        "q": query,
        "api_key": api_key,
        "engine": "google",
        "num": max_results,
    })
    req = Request(
        f"https://serpapi.com/search.json?{params}",
        headers={"User-Agent": "AgentX/2.4"},
        method="GET",
    )
    with urlopen(req, timeout=settings.SEARCH_TIMEOUT_SECONDS) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results = []
    for item in data.get("organic_results", [])[:max_results]:
        url = item.get("link")
        if url:
            body = item.get("snippet") or ""
            date_val = item.get("date") or _extract_date_snippet(body)
            results.append({
                "title": item.get("title", "Untitled").strip(),
                "url": url.strip(),
                "snippet": body.strip(),
                "date": date_val,
            })
    return results


def web_search(query: str, max_results: Optional[int] = None) -> ToolResult:
    """Execute general-purpose web search discovering verified public sources.

    Args:
        query: Search query string.
        max_results: Optional maximum number of results (default from settings: 5).

    Returns:
        ToolResult with status SUCCESS, NO_RESULTS, INVALID_INPUT, or ERROR.
    """
    if not query or not isinstance(query, str):
        return ToolResult(
            status=ToolStatus.INVALID_INPUT,
            error="Search query must be a non-empty string.",
        )

    clean_query = sanitize_search_query(query)
    if not clean_query:
        return ToolResult(
            status=ToolStatus.INVALID_INPUT,
            error="Search query cannot be whitespace only.",
        )

    count = max_results or settings.SEARCH_MAX_RESULTS
    count = max(1, min(count, 10))

    provider = (settings.SEARCH_PROVIDER or "duckduckgo").strip().lower()
    tavily_key = settings.get_tavily_api_key()
    serpapi_key = settings.get_serpapi_api_key()

    raw_results: List[Dict[str, Any]] = []
    active_provider = provider

    try:
        if provider == "tavily":
            if not tavily_key:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error="Tavily provider configured but TAVILY_API_KEY is not set. Set TAVILY_API_KEY in .env or switch SEARCH_PROVIDER to 'duckduckgo'.",
                )
            raw_results = _search_tavily(clean_query, tavily_key, count)
        elif provider == "serpapi":
            if not serpapi_key:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    error="SerpAPI provider configured but SERPAPI_API_KEY is not set. Set SERPAPI_API_KEY in .env or switch SEARCH_PROVIDER to 'duckduckgo'.",
                )
            raw_results = _search_serpapi(clean_query, serpapi_key, count)
        else:
            # Default to DuckDuckGo search
            active_provider = "duckduckgo"
            raw_results = _search_duckduckgo(clean_query, count)

            # If DuckDuckGo returned 0 results, check if a commercial fallback provider is configured
            if not raw_results:
                if tavily_key:
                    try:
                        logger.info("DuckDuckGo returned 0 results; attempting configured Tavily fallback...")
                        raw_results = _search_tavily(clean_query, tavily_key, count)
                        if raw_results:
                            active_provider = "tavily (fallback)"
                    except Exception as fb_err:
                        logger.warning("Tavily fallback search failed: %s", fb_err)
                elif serpapi_key:
                    try:
                        logger.info("DuckDuckGo returned 0 results; attempting configured SerpAPI fallback...")
                        raw_results = _search_serpapi(clean_query, serpapi_key, count)
                        if raw_results:
                            active_provider = "serpapi (fallback)"
                    except Exception as fb_err:
                        logger.warning("SerpAPI fallback search failed: %s", fb_err)

    except Exception as e:
        err_msg = str(e).split("\n")[0]
        logger.warning("Web search provider '%s' failed for query '%s': %s", active_provider, clean_query, err_msg)
        # Attempt fallback if a primary provider failed with an error
        fallback_done = False
        if active_provider != "duckduckgo":
            try:
                logger.info("Attempting fallback search to DuckDuckGo...")
                raw_results = _search_duckduckgo(clean_query, count)
                active_provider = "duckduckgo (fallback)"
                fallback_done = True
            except Exception:
                pass
        elif tavily_key:
            try:
                logger.info("DuckDuckGo failed with error; attempting fallback to Tavily...")
                raw_results = _search_tavily(clean_query, tavily_key, count)
                active_provider = "tavily (fallback)"
                fallback_done = True
            except Exception:
                pass
        elif serpapi_key:
            try:
                logger.info("DuckDuckGo failed with error; attempting fallback to SerpAPI...")
                raw_results = _search_serpapi(clean_query, serpapi_key, count)
                active_provider = "serpapi (fallback)"
                fallback_done = True
            except Exception:
                pass

        if not fallback_done and not raw_results:
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Web search failed for query '{clean_query}': {err_msg}",
                evidence={"query": clean_query},
            )

    # Filter all discovered URLs with strict SSRF defense
    safe_results: List[Dict[str, Any]] = []
    blocked_count = 0
    seen_urls = set()

    for raw_item in raw_results:
        item = dict(raw_item)
        url = (item.get("url") or item.get("href") or item.get("link") or "").strip()
        if not url or url.lower() in seen_urls:
            continue
        seen_urls.add(url.lower())
        title = (item.get("title") or "Untitled").strip()
        body = (item.get("snippet") or item.get("body") or item.get("content") or "").strip()
        date = item.get("date") or _extract_date_snippet(body)

        item["url"] = url
        item["title"] = title
        item["snippet"] = body
        item["date"] = date

        is_safe, reason, _ = validate_url_for_ssrf(url)
        if is_safe:
            safe_results.append(item)
        else:
            blocked_count += 1
            logger.info("Filtered internal/unsafe search result destination '%s': %s", url, reason)


    if not safe_results:
        return ToolResult(
            status=ToolStatus.NO_RESULTS,
            data=[],
            message=(
                f"No public search results found for query: '{clean_query}'. "
                "Current market prices or specifications could not be confirmed online. "
                "Suggested query refinements: try specifying a specific component capacity (e.g. '16GB DDR4'), "
                "manufacturer (e.g. Crucial, Corsair), or an online retailer (e.g. Amazon, Flipkart)."
            ),
            evidence={"query": clean_query, "provider": active_provider, "results": []},
        )

    # Format human/LLM-readable summary listing verified sources and snippets
    summary_lines = [f"Found {len(safe_results)} verified search result(s) via {active_provider}:"]
    for i, res in enumerate(safe_results, start=1):
        date_str = f" [Date: {res['date']}]" if res.get("date") else ""
        summary_lines.append(f"{i}. [{res['title']}]({res['url']}){date_str}\n   Snippet: {res['snippet']}")

    return ToolResult(
        status=ToolStatus.SUCCESS,
        data=safe_results,
        message="\n".join(summary_lines),
        evidence={
            "query": clean_query,
            "provider": active_provider,
            "result_count": len(safe_results),
            "results": safe_results,
            "blocked_unsafe_count": blocked_count,
        },
    )
