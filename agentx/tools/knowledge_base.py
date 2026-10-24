"""Local JSON knowledge-base search tool for AgentX.

Provides transparent, deterministic retrieval over sample records with stable
source IDs without requiring complex vector databases or external embeddings.
"""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from agentx.config import settings
from agentx.schemas import ToolResult, ToolStatus


DOMAIN_ALIASES = {
    "ssrf": ["server-side request forgery", "server side request forgery", "request forgery"],
    "server-side request forgery": ["ssrf"],
    "server side request forgery": ["ssrf"],
    "ast": ["abstract syntax tree"],
    "abstract syntax tree": ["ast"],
    "eval": ["evaluation", "evaluating", "evaluate"],
    "calculator": ["math", "mathematical", "calculation"],
    "deduplication": ["duplicate", "duplicate response", "rephrase"],
    "playwright": ["browser", "automation"],
}

KB_STOPWORDS = {
    "how", "does", "do", "what", "is", "are", "the", "a", "an", "and", "or",
    "to", "in", "of", "for", "on", "with", "at", "by", "from", "about", "agentx",
    "can", "you", "tell", "me", "show", "policy", "guidelines", "protocol"
}


def _stem(word: str) -> str:
    """Basic English suffix normalization for token matching."""
    w = word.lower().strip()
    for suffix in ("ing", "tion", "tions", "ies", "es", "ed", "s"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            return w[:-len(suffix)]
    return w


def _tokenize(text: str) -> List[str]:
    """Extract lowercase alphanumeric tokens."""
    return re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())


def search_knowledge_base(query: str, kb_path: Optional[Path] = None) -> ToolResult:
    """Search the local knowledge-base dataset for matching records.

    Args:
        query: Search string or keywords.
        kb_path: Optional override path to the JSON dataset file.

    Returns:
        ToolResult with status SUCCESS, NO_RESULTS, INVALID_INPUT, or ERROR.
    """
    if not query or not isinstance(query, str):
        return ToolResult(
            status=ToolStatus.INVALID_INPUT,
            error="Search query must be a non-empty string.",
        )

    clean_query = query.strip()
    if not clean_query:
        return ToolResult(
            status=ToolStatus.INVALID_INPUT,
            error="Search query cannot be whitespace only.",
        )

    target_path = Path(kb_path) if kb_path else settings.KNOWLEDGE_BASE_PATH

    # Check file existence
    if not target_path.exists():
        return ToolResult(
            status=ToolStatus.ERROR,
            error=f"Knowledge base file not found at path: {target_path}",
        )

    # Read and parse JSON dataset
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            error=f"Knowledge base file is malformed JSON: {str(e)}",
        )
    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            error=f"Failed to read knowledge base file: {str(e)}",
        )

    if not isinstance(raw_data, list):
        return ToolResult(
            status=ToolStatus.ERROR,
            error="Knowledge base dataset must be a JSON array of record objects.",
        )

    query_tokens = _tokenize(clean_query)
    query_lower = clean_query.lower()

    # Expand query with domain aliases for acronyms and terms (e.g. ssrf <-> server-side request forgery)
    expanded_phrases = [query_lower]
    for alias_key, alias_vals in DOMAIN_ALIASES.items():
        if alias_key in query_lower:
            expanded_phrases.extend(alias_vals)
        for val in alias_vals:
            if val in query_lower:
                expanded_phrases.append(alias_key)

    core_query_tokens = [tok for tok in query_tokens if tok not in KB_STOPWORDS]
    tokens_to_evaluate = core_query_tokens if core_query_tokens else query_tokens

    scored_records: List[Dict[str, Any]] = []

    for record in raw_data:
        if not isinstance(record, dict) or "id" not in record:
            continue

        rec_id = str(record.get("id", ""))
        rec_title = str(record.get("title", ""))
        rec_category = str(record.get("category", ""))
        rec_tags = record.get("tags", [])
        if isinstance(rec_tags, list):
            rec_tags_str = " ".join(str(t) for t in rec_tags)
        else:
            rec_tags_str = str(rec_tags)
        rec_content = str(record.get("content", ""))

        score = 0.0

        # Exact phrase and alias match bonuses
        has_phrase_match = False
        for phrase in expanded_phrases:
            if phrase in rec_title.lower():
                score += 15.0
                has_phrase_match = True
            if phrase in rec_tags_str.lower():
                score += 12.0
                has_phrase_match = True
            if phrase in rec_content.lower():
                score += 8.0
                has_phrase_match = True

        # Token matching with field weights and stemming
        title_tokens = set(_tokenize(rec_title))
        title_stems = {_stem(t) for t in title_tokens}
        tags_tokens = set(_tokenize(rec_tags_str))
        tags_stems = {_stem(t) for t in tags_tokens}
        content_tokens = set(_tokenize(rec_content))
        content_stems = {_stem(t) for t in content_tokens}
        category_tokens = set(_tokenize(rec_category))
        category_stems = {_stem(t) for t in category_tokens}

        matched_tokens = 0
        for tok in tokens_to_evaluate:
            tok_stem = _stem(tok)
            tok_matched = False
            if tok in rec_id.lower():
                score += 20.0
                tok_matched = True
            if tok in title_tokens or tok_stem in title_stems:
                score += 6.0
                tok_matched = True
            if tok in tags_tokens or tok_stem in tags_stems:
                score += 5.0
                tok_matched = True
            if tok in category_tokens or tok_stem in category_stems:
                score += 3.0
                tok_matched = True
            if tok in content_tokens or tok_stem in content_stems:
                score += 2.0
                tok_matched = True
            if tok_matched:
                matched_tokens += 1

        # Relevance filtering
        is_confident_match = False
        if has_phrase_match:
            is_confident_match = True
        elif len(tokens_to_evaluate) == 1:
            is_confident_match = score >= 3.0
        elif len(tokens_to_evaluate) > 1:
            overlap_ratio = matched_tokens / len(tokens_to_evaluate)
            is_confident_match = (overlap_ratio >= 0.35) or (score >= 6.0)

        if is_confident_match and score > 0:
            scored_records.append({
                "id": rec_id,
                "title": rec_title,
                "category": rec_category,
                "tags": rec_tags,
                "content": rec_content,
                "score": round(score, 2),
            })

    # Sort deterministically: highest score first, stable ID as tiebreaker
    scored_records.sort(key=lambda x: (-x["score"], x["id"]))

    if not scored_records:
        return ToolResult(
            status=ToolStatus.NO_RESULTS,
            data=[],
            message=f"No matching records found in knowledge base for query: '{clean_query}'.",
            evidence={"query": clean_query, "matched_ids": []},
        )

    # Return top matching records (up to 3)
    top_records = scored_records[:3]
    matched_ids = [r["id"] for r in top_records]

    summary_lines = [
        f"Found {len(top_records)} matching record(s) for query '{clean_query}' (IDs: {', '.join(matched_ids)}):"
    ]
    for r in top_records:
        summary_lines.append(f"- [{r['id']}] {r['title']} ({r['category']}): {r['content']}")

    return ToolResult(
        status=ToolStatus.SUCCESS,
        data=top_records,
        message="\n".join(summary_lines),
        evidence={
            "query": clean_query,
            "matched_ids": matched_ids,
            "records": top_records,
        },
    )
