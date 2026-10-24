"""Safe AST-based arithmetic calculator tool for AgentX.

Evaluates mathematical expressions without using unrestricted eval() or exec().
Only whitelisted AST arithmetic nodes and operations are permitted.
"""

import ast
import math
from typing import Union
from agentx.config import settings
from agentx.schemas import ToolResult, ToolStatus


class CalculationError(Exception):
    """Custom exception raised when an arithmetic operation fails."""
    pass


class SafeASTEvaluator(ast.NodeVisitor):
    """AST visitor that evaluates only strictly whitelisted arithmetic expressions."""

    ALLOWED_BINARY_OPS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
    }

    ALLOWED_UNARY_OPS = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    def visit_Expression(self, node: ast.Expression) -> Union[int, float]:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Union[int, float]:
        # Disallow strings, booleans, None, etc.
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculationError(
                f"Unsupported constant type: {type(node.value).__name__}. Only numbers are allowed."
            )
        val = node.value
        if not math.isfinite(val):
            raise CalculationError("Non-finite numbers (NaN, Inf) are not permitted.")
        if abs(val) > settings.CALCULATOR_MAX_MAGNITUDE:
            raise CalculationError(
                f"Numeric literal exceeds maximum allowed magnitude ({settings.CALCULATOR_MAX_MAGNITUDE})."
            )
        return val

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Union[int, float]:
        op_type = type(node.op)
        if op_type not in self.ALLOWED_UNARY_OPS:
            raise CalculationError(f"Unsupported unary operator: {op_type.__name__}")
        operand = self.visit(node.operand)
        res = self.ALLOWED_UNARY_OPS[op_type](operand)
        if not math.isfinite(res):
            raise CalculationError("Unary operation resulted in a non-finite number.")
        return res

    def visit_BinOp(self, node: ast.BinOp) -> Union[int, float]:
        op_type = type(node.op)
        if op_type not in self.ALLOWED_BINARY_OPS:
            raise CalculationError(f"Unsupported binary operator: {op_type.__name__}")

        left = self.visit(node.left)
        right = self.visit(node.right)

        # Division by zero check
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod):
            if right == 0:
                raise CalculationError("Division by zero is undefined.")

        # Power exponent limit check to prevent CPU exhaustion
        if op_type is ast.Pow:
            if abs(right) > settings.CALCULATOR_MAX_EXPONENT:
                raise CalculationError(
                    f"Exponent {right} exceeds safe calculation limit ({settings.CALCULATOR_MAX_EXPONENT})."
                )
            if left == 0 and right < 0:
                raise CalculationError("Zero cannot be raised to a negative power.")

        try:
            result = self.ALLOWED_BINARY_OPS[op_type](left, right)
        except OverflowError:
            raise CalculationError("Arithmetic operation resulted in numeric overflow.")

        # In Python, complex numbers can be generated (e.g. (-4)**0.5)
        if isinstance(result, complex):
            raise CalculationError("Complex numbers are not supported in real arithmetic.")

        if not math.isfinite(result):
            raise CalculationError("Operation resulted in a non-finite number (NaN or Inf).")

        if abs(result) > settings.CALCULATOR_MAX_MAGNITUDE:
            raise CalculationError(
                f"Result exceeds maximum allowed magnitude ({settings.CALCULATOR_MAX_MAGNITUDE})."
            )

        return result

    def generic_visit(self, node: ast.AST):
        # Reject any node type not explicitly implemented above
        raise CalculationError(
            f"Unsupported syntax or operation: {type(node).__name__}. "
            "Only arithmetic expressions (numbers, +, -, *, /, //, %, **, parentheses) are allowed."
        )


def safe_calculate(expression: str) -> ToolResult:
    """Safely calculate the value of an arithmetic expression.

    Args:
        expression: Mathematical string (e.g., '(12 * 4) + 10 / 2')

    Returns:
        ToolResult with status, computed data, evidence, or error message.
    """
    if not expression or not isinstance(expression, str):
        return ToolResult(
            status=ToolStatus.INVALID_INPUT,
            error="Expression must be a non-empty string.",
        )

    expr_clean = expression.strip()

    # 1. Length check
    if len(expr_clean) > settings.CALCULATOR_MAX_EXPR_LENGTH:
        return ToolResult(
            status=ToolStatus.INVALID_INPUT,
            error=f"Expression length ({len(expr_clean)}) exceeds limit of {settings.CALCULATOR_MAX_EXPR_LENGTH} characters.",
        )

    # 2. Parse AST in 'eval' mode
    try:
        parsed_tree = ast.parse(expr_clean, mode="eval")
    except SyntaxError as e:
        return ToolResult(
            status=ToolStatus.INVALID_INPUT,
            error=f"Invalid mathematical syntax: {e.msg}",
        )
    except Exception as e:
        return ToolResult(
            status=ToolStatus.INVALID_INPUT,
            error=f"Failed to parse expression: {str(e)}",
        )

    # 3. Evaluate AST safely
    evaluator = SafeASTEvaluator()
    try:
        raw_result = evaluator.visit(parsed_tree)
        # Format clean integer if exact
        if isinstance(raw_result, float) and raw_result.is_integer():
            final_result: Union[int, float] = int(raw_result)
        else:
            final_result = raw_result

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data=final_result,
            message=f"Calculated: {expr_clean} = {final_result}",
            evidence={"expression": expr_clean, "result": final_result},
        )
    except CalculationError as e:
        # Determine if this is invalid input vs runtime arithmetic error
        err_msg = str(e)
        if "Division by zero" in err_msg or "overflow" in err_msg:
            return ToolResult(status=ToolStatus.ERROR, error=err_msg)
        return ToolResult(status=ToolStatus.INVALID_INPUT, error=err_msg)
    except Exception as e:
        return ToolResult(
            status=ToolStatus.ERROR,
            error=f"Unexpected calculation error: {str(e)}",
        )
