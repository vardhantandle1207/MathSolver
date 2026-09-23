"""
A quick calculator for one-off numeric expressions.

This is deliberately narrower than the Python tool: no variables, no loops, just
"evaluate this arithmetic expression exactly". It exists so the agent isn't
forced to spin up a whole subprocess just to divide two numbers, and because
routing simple sums here keeps the traces readable.
"""

import ast
import operator

# Only these node/operator types are allowed through. Anything else (attribute
# access, function calls, names, etc.) is rejected — that's the whole safety
# story for this tool.
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):  # numeric literal
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Disallowed expression element: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Evaluate a plain arithmetic expression, e.g. '2**10 + 5*3'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as exc:  # noqa: BLE001 - surface any parse/eval issue to the model
        return f"[error] Could not evaluate {expression!r}: {exc}"
