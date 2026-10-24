"""Unit tests for the safe AST calculator tool."""

import pytest
from agentx.schemas import ToolStatus
from agentx.tools.calculator import safe_calculate


def test_basic_arithmetic():
    """Verify standard addition, subtraction, multiplication, division."""
    res = safe_calculate("12 + 8")
    assert res.status == ToolStatus.SUCCESS
    assert res.data == 20

    res = safe_calculate("100 - 35")
    assert res.status == ToolStatus.SUCCESS
    assert res.data == 65

    res = safe_calculate("7 * 8")
    assert res.status == ToolStatus.SUCCESS
    assert res.data == 56

    res = safe_calculate("50 / 4")
    assert res.status == ToolStatus.SUCCESS
    assert res.data == 12.5


def test_parentheses_and_precedence():
    """Verify operator precedence and parenthesis grouping."""
    res1 = safe_calculate("2 + 3 * 4")
    assert res1.status == ToolStatus.SUCCESS
    assert res1.data == 14

    res2 = safe_calculate("(2 + 3) * 4")
    assert res2.status == ToolStatus.SUCCESS
    assert res2.data == 20

    res3 = safe_calculate("((10 + 5) * 2) / (3 + 2)")
    assert res3.status == ToolStatus.SUCCESS
    assert res3.data == 6


def test_unary_and_negative_numbers():
    """Verify unary plus and minus operations."""
    res = safe_calculate("-15 + 20")
    assert res.status == ToolStatus.SUCCESS
    assert res.data == 5

    res2 = safe_calculate("-(4 * 5)")
    assert res2.status == ToolStatus.SUCCESS
    assert res2.data == -20

    res3 = safe_calculate("+42")
    assert res3.status == ToolStatus.SUCCESS
    assert res3.data == 42


def test_exponentiation_and_limits():
    """Verify safe exponentiation and rejection of unbounded powers."""
    res = safe_calculate("2 ** 8")
    assert res.status == ToolStatus.SUCCESS
    assert res.data == 256

    # Exponent exceeding safe limit (100) must be rejected
    res_large_exp = safe_calculate("2 ** 101")
    assert res_large_exp.status == ToolStatus.INVALID_INPUT
    assert "safe calculation limit" in res_large_exp.error


def test_division_by_zero():
    """Verify that division by zero returns an ERROR status, not an unhandled crash."""
    res_div = safe_calculate("10 / 0")
    assert res_div.status == ToolStatus.ERROR
    assert "Division by zero" in res_div.error

    res_floor = safe_calculate("10 // 0")
    assert res_floor.status == ToolStatus.ERROR
    assert "Division by zero" in res_floor.error

    res_mod = safe_calculate("10 % 0")
    assert res_mod.status == ToolStatus.ERROR
    assert "Division by zero" in res_mod.error


def test_code_injection_prevention():
    """Verify that arbitrary Python statements, function calls, and modules are rejected."""
    payloads = [
        "__import__('os').system('dir')",
        "eval('2+2')",
        "open('README.md')",
        "lambda x: x+1",
        "[x for x in range(10)]",
        "import sys",
        "print('pwned')",
        "globals()",
        "__builtins__",
    ]
    for payload in payloads:
        res = safe_calculate(payload)
        assert res.status == ToolStatus.INVALID_INPUT, f"Failed to reject injection: {payload}"
        assert res.error is not None


def test_disallow_non_numeric_constants():
    """Verify that booleans, strings, and non-numbers are rejected."""
    assert safe_calculate("'hello' + 'world'").status == ToolStatus.INVALID_INPUT
    assert safe_calculate("True + 1").status == ToolStatus.INVALID_INPUT
    assert safe_calculate("None").status == ToolStatus.INVALID_INPUT


def test_empty_and_oversized_inputs():
    """Verify empty expressions and inputs exceeding length limits are rejected."""
    assert safe_calculate("").status == ToolStatus.INVALID_INPUT
    assert safe_calculate("   ").status == ToolStatus.INVALID_INPUT

    long_expr = "1 + " * 100 + "1"
    res = safe_calculate(long_expr)
    assert res.status == ToolStatus.INVALID_INPUT
    assert "exceeds limit" in res.error
