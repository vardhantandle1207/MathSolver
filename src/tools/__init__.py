"""
Tool registry.

Two things live here:
  1. TOOL_SCHEMAS  - the JSON schema list we hand to the model so it knows what
                     it can call and with what arguments.
  2. TOOL_FUNCS    - maps a tool name back to the Python function that runs it.

The `final_answer` tool is a bit of a trick: instead of trying to guess when the
model is "done" from its free text, we make it call an explicit tool to submit
the answer. That gives the agent loop a clean, unambiguous stop condition.
"""

from .calculator import calculate
from .python_repl import run_python

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute Python code and return its printed output. SymPy (as sp / "
                "solve, integrate, diff, Matrix, ...), NumPy (np) and math are already "
                "imported. Use this for any real computation — solving equations, "
                "integrals, derivatives, simplification. A bare expression on the "
                "last line is printed automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source to run. Must print the result.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a single arithmetic expression exactly, e.g. "
                "'(3+4)**2 / 7'. Faster than run_python for plain number crunching."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression using + - * / ** % //",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": (
                "Submit the final answer once you are confident. For a single-answer "
                "MCQ give one option letter (e.g. 'B'); if MORE THAN ONE option is "
                "correct give every correct letter joined together (e.g. 'AC', "
                "'BCD'); for numerical problems give just the number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to the problem.",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]

# final_answer isn't a real callable — the loop intercepts it — so it's not here.
TOOL_FUNCS = {
    "run_python": run_python,
    "calculate": calculate,
}
