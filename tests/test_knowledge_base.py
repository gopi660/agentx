"""Unit tests for the local JSON knowledge-base search tool."""

import json
from pathlib import Path
import pytest
from agentx.schemas import ToolStatus
from agentx.tools.knowledge_base import search_knowledge_base


def test_successful_search_returns_matches():
    """Verify that relevant queries return expected records and stable IDs."""
    res = search_knowledge_base("SSRF defense")
    assert res.status == ToolStatus.SUCCESS
    assert res.data is not None
    assert len(res.data) > 0
    # KB-002 is the SSRF Defense Policy
    matched_ids = [r["id"] for r in res.data]
    assert "KB-002" in matched_ids
    assert res.evidence["matched_ids"] == matched_ids


def test_search_by_tag_or_category():
    """Verify searching by category or tag name."""
    res = search_knowledge_base("architecture")
    assert res.status == ToolStatus.SUCCESS
    assert any(r["id"] == "KB-001" for r in res.data)

    res_rel = search_knowledge_base("release schedule 2026")
    assert res_rel.status == ToolStatus.SUCCESS
    assert any(r["id"] == "KB-005" for r in res_rel.data)


def test_no_results_status():
    """Verify that completely unmatched queries return clear NO_RESULTS status."""
    res = search_knowledge_base("quantum teleportation interstellar propulsion")
    assert res.status == ToolStatus.NO_RESULTS
    assert res.data == []
    assert "No matching records" in res.message
    assert res.evidence["matched_ids"] == []


def test_empty_or_whitespace_query():
    """Verify rejection of empty query strings."""
    res1 = search_knowledge_base("")
    assert res1.status == ToolStatus.INVALID_INPUT

    res2 = search_knowledge_base("     ")
    assert res2.status == ToolStatus.INVALID_INPUT


def test_missing_file_error(tmp_path: Path):
    """Verify that an unavailable dataset file returns ERROR status."""
    non_existent = tmp_path / "does_not_exist.json"
    res = search_knowledge_base("test query", kb_path=non_existent)
    assert res.status == ToolStatus.ERROR
    assert "not found" in res.error.lower()


def test_malformed_json_error(tmp_path: Path):
    """Verify that invalid JSON returns an ERROR status with parse details."""
    broken_file = tmp_path / "broken.json"
    broken_file.write_text("{ this is not valid json }", encoding="utf-8")

    res = search_knowledge_base("test query", kb_path=broken_file)
    assert res.status == ToolStatus.ERROR
    assert "malformed JSON" in res.error


def test_non_list_json_structure(tmp_path: Path):
    """Verify that a JSON object instead of an array returns ERROR status."""
    dict_file = tmp_path / "dict.json"
    dict_file.write_text(json.dumps({"key": "value"}), encoding="utf-8")

    res = search_knowledge_base("test query", kb_path=dict_file)
    assert res.status == ToolStatus.ERROR
    assert "JSON array" in res.error
