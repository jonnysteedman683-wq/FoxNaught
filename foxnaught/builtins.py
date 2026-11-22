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


def _safe_eval(node: ast.AST) -> object:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise ValueError(f"unknown function: {node.func.id}")
        return func(*[_safe_eval(arg) for arg in node.args])
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    """Safely evaluate an arithmetic expression."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return f"Error: invalid expression: {exc.msg}"
    try:
        return repr(_safe_eval(tree))
    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
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
            "sqrt, log, exp, sin, cos, tan."
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
