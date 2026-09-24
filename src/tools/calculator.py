"""
A quick calculator for one-off numeric expressions.

This is deliberately narrower than the Python tool: no variables, no loops, just
"evaluate this arithmetic expression exactly". It exists so the agent isn't
forced to spin up a whole subprocess just to divide two numbers, and because
routing simple sums here keeps the traces readable.
"""

import ast
import math
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

# Named maths functions, resolved from this table rather than from any runtime
# namespace. Rejecting every Call made this tool useless for a maths model:
# `log(16)/log(2)` was refused 240 times in one benchmark run, and the agent
# fell back to doing it in its head. An allowlist keeps the safety property
# (nothing outside this dict can be called) while making the tool usable.
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
    "factorial": math.factorial, "gcd": math.gcd, "comb": math.comb,
    "perm": math.perm, "degrees": math.degrees, "radians": math.radians,
    "min": min, "max": max, "pow": pow,
}
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):  # numeric literal
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _ALLOWED_NAMES:
        return _ALLOWED_NAMES[node.id]
    if isinstance(node, ast.Call):
        # Only a plain name from the allowlist may be called — no attributes,
        # no keywords, no calling the result of another expression.
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            name = getattr(node.func, "id", type(node.func).__name__)
            raise ValueError(f"Unknown function: {name}")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported")
        return _ALLOWED_FUNCS[node.func.id](*[_eval_node(a) for a in node.args])
    raise ValueError(f"Disallowed expression element: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Evaluate a plain arithmetic expression, e.g. '2**10 + 5*3'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as exc:  # noqa: BLE001 - surface any parse/eval issue to the model
        return f"[error] Could not evaluate {expression!r}: {exc}"
