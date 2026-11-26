"""Built-in tools provided with the agent."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from datetime import datetime, timezone

from .tools import ToolRegistry

_BIN_OPS: dict[type, Callable[[object, object], object]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type, Callable[[object], object]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_CONSTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}

_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}

# The calculator is exposed to model-generated input, so bound the work an
# expression can request. Without these caps, 9**9**9 alone can exhaust memory.
MAX_EXPRESSION_CHARS = 1_000
MAX_EXPONENT = 4_096
MAX_INT_BITS = 8_192


def _bounded_int(value: object) -> object:
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise ValueError("integer result is too large")
    return value


def _safe_eval(node: ast.AST) -> object:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and isinstance(right, int):
            if right < 0 or right > MAX_EXPONENT:
                raise ValueError("exponent is out of range")
            if isinstance(left, int) and left.bit_length() * right > MAX_INT_BITS:
                raise ValueError("integer result is too large")
        return _bounded_int(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _bounded_int(_UNARY_OPS[type(node.op)](_safe_eval(node.operand)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise ValueError(f"unknown function: {node.func.id}")
        return _bounded_int(func(*[_safe_eval(arg) for arg in node.args]))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    """Safely evaluate an arithmetic expression within bounded resource limits."""
    if not isinstance(expression, str):
        return "Error: expression must be a string"
    if len(expression) > MAX_EXPRESSION_CHARS:
        return f"Error: expression is longer than {MAX_EXPRESSION_CHARS} characters"
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        return f"Error: invalid expression: {exc}"
    try:
        return repr(_safe_eval(tree))
    except (ValueError, ZeroDivisionError, OverflowError, TypeError, RecursionError) as exc:
        return f"Error: {exc}"


def now_utc(timezone_name: str = "UTC") -> str:
    """Return the current UTC time in ISO-8601 format."""
    if timezone_name != "UTC":
        return "Error: only the UTC timezone is supported"
    return datetime.now(timezone.utc).isoformat()


def register_defaults(registry: ToolRegistry) -> None:
    registry.register(
        name="calculate",
        description=(
            "Evaluate an arithmetic expression. Supports +, -, *, /, //, %, **, "
            "parentheses, constants pi/e/tau, and functions abs, round, min, max, "
            "sqrt, log, exp, sin, cos, tan. Very large exponents, integer results, "
            "and long expressions are rejected."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate.",
                }
            },
            "required": ["expression"],
        },
        func=calculate,
    )
    registry.register(
        name="now_utc",
        description="Return the current UTC time in ISO-8601 format.",
        parameters={
            "type": "object",
            "properties": {
                "timezone_name": {
                    "type": "string",
                    "description": "Only 'UTC' is supported.",
                    "default": "UTC",
                }
            },
            "required": [],
        },
        func=now_utc,
    )
