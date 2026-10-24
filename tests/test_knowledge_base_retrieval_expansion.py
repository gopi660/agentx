"""Regression tests for Knowledge Base retrieval enhancements.

Tests:
- Matching query: 'AgentX prevent server-side request forgery SSRF' matches KB-002.
- Alternate wordings: 'server-side request forgery', 'how does agentx defend against ssrf', 'safe AST evaluation'.
- Empty knowledge base handling.
- Genuine no-match cases returning NO_RESULTS without inventing security implementations.
"""

from pathlib import Path
import pytest

from agentx.schemas import ToolStatus
from agentx.tools.knowledge_base import search_knowledge_base


def test_ssrf_query_matches_kb002():
    """Verify 'AgentX prevent server-side request forgery SSRF' retrieves KB-002."""
    res = search_knowledge_base(query="AgentX prevent server-side request forgery SSRF")
    assert res.status == ToolStatus.SUCCESS
    assert "KB-002" in res.evidence["matched_ids"]
    assert any(r["id"] == "KB-002" for r in res.data)
    assert "SSRF Defense Policy" in res.message
    assert len(res.data) >= 1


def test_ssrf_alternate_wordings():
    """Verify alternate natural language wordings for SSRF retrieval."""
    queries = [
        "how does agentx defend against ssrf attacks?",
        "server-side request forgery prevention policy",
        "ssrf allowlist private ip filtering",
    ]
    for q in queries:
        res = search_knowledge_base(query=q)
        assert res.status == ToolStatus.SUCCESS, f"Failed for query: {q}"
        assert "KB-002" in res.evidence["matched_ids"]


def test_ast_evaluation_alternate_wordings():
    """Verify retrieval of AST evaluation documentation via aliases and terms."""
    queries = [
        "safe AST arithmetic evaluation",
        "abstract syntax tree calculator parser",
        "how does calculator prevent code injection",
    ]
    for q in queries:
        res = search_knowledge_base(query=q)
        assert res.status == ToolStatus.SUCCESS, f"Failed for query: {q}"
        assert "KB-003" in res.evidence["matched_ids"]


def test_empty_knowledge_base_handling(tmp_path):
    """Verify search returns NO_RESULTS cleanly when knowledge base is empty."""
    empty_kb_file = tmp_path / "empty_kb.json"
    empty_kb_file.write_text("[]", encoding="utf-8")

    res = search_knowledge_base(
        query="AgentX prevent server-side request forgery SSRF",
        kb_path=empty_kb_file,
    )
    assert res.status == ToolStatus.NO_RESULTS
    assert "No matching records found in knowledge base" in res.message
    assert res.data == []
    assert res.evidence["matched_ids"] == []


def test_genuine_no_match_returns_no_results_and_does_not_hallucinate():
    """Verify completely unrelated query returns NO_RESULTS without fabrication."""
    query = "quantum hyperdimensional teleportation engine warp drive"
    res = search_knowledge_base(query=query)
    assert res.status == ToolStatus.NO_RESULTS
    assert "No matching records found in knowledge base" in res.message
    assert res.data == []
    assert res.evidence["matched_ids"] == []
