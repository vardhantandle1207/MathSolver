"""
The 'before' picture: the same model, same problem, but NO tools.

This is the control we measure the agent against. Whatever accuracy gap opens up
between this and agent.solve() is the actual contribution of the tool layer —
that delta is the headline number for the whole project.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import get_chat_model, message_text
from .config import settings

_BASELINE_SYSTEM = (
    "You are solving a JEE-level maths problem. Think step by step, then end your "
    "response with a line of the form 'FINAL: <answer>' where <answer> is the "
    "option letter for an MCQ (all correct letters joined, e.g. 'AC', if more "
    "than one option is correct) or the numeric value otherwise."
)


def solve_baseline(problem: str, model: Any = None) -> str:
    """Return just the extracted final answer, so scoring is apples-to-apples
    with the agent's output. `model` is injectable for tests."""
    if model is None:
        model = get_chat_model()
    # Same retry policy as the agent, so a transient blip doesn't skew the
    # baseline number vs the agent number.
    if hasattr(model, "with_retry"):
        model = model.with_retry(stop_after_attempt=settings.max_retries)

    try:
        reply = model.invoke([
            SystemMessage(content=_BASELINE_SYSTEM),
            HumanMessage(content=problem),
        ])
    except Exception:  # noqa: BLE001 - same reasoning as the agent: score it a miss
        return ""
    text = message_text(reply)

    # Pull out the FINAL: line if present, otherwise fall back to the last line.
    for line in reversed(text.strip().splitlines()):
        if "FINAL:" in line:
            return line.split("FINAL:", 1)[1].strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""
