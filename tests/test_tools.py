"""
Tests for the pieces that don't need the LLM.

I can't cheaply unit-test the agent loop (it needs a live model), but the tools
and the grader are pure functions, and those are exactly the bits where a silent
bug would quietly wreck the eval numbers. So they get tests.

Run: pytest -q
"""

from src.tools.calculator import calculate
from src.tools.python_repl import run_python
from src.scoring import is_correct


# ---- calculator -----------------------------------------------------------

def test_calculator_basic():
    assert calculate("2 + 3 * 4") == "14"

def test_calculator_power():
    assert calculate("2 ** 10") == "1024"

def test_calculator_rejects_code():
    # No function calls / names allowed — should error, not execute.
    assert "[error]" in calculate("__import__('os').system('ls')")


# ---- python repl ----------------------------------------------------------

def test_python_repl_sympy_solve():
    code = "x = sp.Symbol('x'); print(sorted(solve(x**2 - 4, x)))"
    out = run_python(code)
    assert "-2" in out and "2" in out

def test_python_repl_integral():
    code = "x = sp.Symbol('x'); print(integrate(x**2, (x, 0, 3)))"
    assert run_python(code).strip() == "9"

def test_python_repl_reports_error():
    assert "[error]" in run_python("print(1/0)")

def test_python_repl_reminds_to_print():
    assert "print()" in run_python("1 + 1")


# ---- scoring --------------------------------------------------------------

def test_scoring_numeric_tolerance():
    assert is_correct("0.5000", "0.5", "numeric")

def test_scoring_fraction():
    assert is_correct("1/2", "0.5", "numeric")

def test_scoring_mcq_with_noise():
    assert is_correct("The answer is (B).", "B", "mcq")

def test_scoring_wrong_answer():
    assert not is_correct("7", "5", "numeric")


# ---- scoring: multiple-correct MCQs ---------------------------------------

def test_scoring_mcq_multiple_match():
    assert is_correct("A and D", "AD", "mcq_multiple")

def test_scoring_mcq_multiple_order_insensitive():
    assert is_correct("D, A", "AD", "mcq_multiple")

def test_scoring_mcq_multiple_partial_is_wrong():
    assert not is_correct("A", "AD", "mcq_multiple")


# ---- agent loop over LangGraph (fake model, no network) -------------------

from langchain_core.messages import AIMessage
from src.agent import solve
from src.config import settings


class _FakeChatModel:
    """Minimal stand-in for a LangChain chat model. bind_tools is a no-op that
    returns self; invoke replays a scripted list of AIMessages. This lets us
    drive the real LangGraph loop end-to-end with zero network calls."""

    def __init__(self, scripted):
        self._scripted = scripted
        self._i = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        msg = self._scripted[self._i]
        self._i += 1
        return msg


def test_agent_loop_runs_tool_then_answers():
    scripted = [
        AIMessage(content="", tool_calls=[
            {"name": "run_python",
             "args": {"code": "x=sp.Symbol('x'); print(integrate(x**2,(x,0,3)))"},
             "id": "t1"},
        ]),
        AIMessage(content="", tool_calls=[
            {"name": "final_answer", "args": {"answer": "9"}, "id": "t2"},
        ]),
    ]
    result = solve("Integral of x^2 from 0 to 3?", model=_FakeChatModel(scripted))
    assert result.answer == "9"
    assert result.stopped_reason == "final_answer"
    # The SymPy tool must actually have run and returned 9.
    tool_steps = [t for t in result.trace if t["type"] == "tool_call"]
    assert tool_steps and tool_steps[0]["observation"].strip() == "9"


# ---- security: sandbox denylist -------------------------------------------

def test_repl_blocks_os_import():
    out = run_python("import os; print(os.listdir('/'))")
    assert "[rejected]" in out

def test_repl_blocks_file_open():
    out = run_python("print(open('/etc/passwd').read())")
    assert "[rejected]" in out

def test_repl_allows_legit_math():
    out = run_python("x = sp.Symbol('x'); print(diff(x**3, x))")
    assert out.strip() == "3*x**2"


# ---- safety: input validation, loop detection -----------------------------

def test_solve_rejects_empty_input():
    res = solve("   ", model=_FakeChatModel([]))
    assert res.stopped_reason == "empty_input"

def test_solve_rejects_oversized_input():
    huge = "x" * 99999
    res = solve(huge, model=_FakeChatModel([]))
    assert res.stopped_reason == "input_too_long"

def test_loop_detection_stops_repeated_calls():
    # Model stubbornly issues the SAME tool call every step. Loop detection must
    # start refusing it, and max_steps must guarantee we still terminate.
    # (Fresh AIMessage objects with unique ids — real models never reuse one.)
    same_calls = [
        AIMessage(
            content="",
            id=f"m{i}",
            tool_calls=[{"name": "calculate", "args": {"expression": "1+1"}, "id": f"loop{i}"}],
        )
        for i in range(20)
    ]
    res = solve("stuck?", model=_FakeChatModel(same_calls))
    loop_stops = [t for t in res.trace if t["type"] == "loop_stop"]
    assert loop_stops, "expected loop detection to trigger"
    # Against the configured cap, not a hard-coded number: the cap is an env
    # setting, and a machine with a different .env was failing this test for
    # having a *higher* limit rather than for running away.
    assert res.steps_taken <= settings.max_steps


# ---- scoring: fixes for rounded gold answers and stray letters -------------

def test_scoring_accepts_precision_beyond_rounded_gold():
    # Answer keys round to 2 decimals; SymPy does not. Both are the same answer.
    assert is_correct("0.3333", "0.33", "numeric")

def test_scoring_still_rejects_a_real_miss():
    assert not is_correct("0.45", "0.33", "numeric")

def test_scoring_ignores_lowercase_words_as_options():
    # 'a' and 'bad' are English, not option letters.
    assert is_correct("This is a bad guess: C", "C", "mcq_multiple")


# ---- security: import-form bypasses ---------------------------------------

def test_repl_blocks_from_import_form():
    assert "[rejected]" in run_python("from os import listdir; print(listdir('/'))")

def test_repl_blocks_dunder_traversal():
    assert "[rejected]" in run_python("print(().__class__.__bases__)")


# ---- memory: conversation threads -----------------------------------------

def test_memory_carries_context_across_calls():
    """Second call on the same thread must see the first question's messages."""
    seen = []

    class _RecordingModel(_FakeChatModel):
        def invoke(self, messages):
            seen.append(list(messages))
            return super().invoke(messages)

    first = _RecordingModel([
        AIMessage(content="", tool_calls=[
            {"name": "final_answer", "args": {"answer": "9"}, "id": "a1"}]),
    ])
    solve("Integral of x^2 from 0 to 3?", model=first, thread_id="t-demo")

    second = _RecordingModel([
        AIMessage(content="", tool_calls=[
            {"name": "final_answer", "args": {"answer": "64"}, "id": "a2"}]),
    ])
    res = solve("Now do the same for x^3 from 0 to 4.", model=second, thread_id="t-demo")

    assert res.answer == "64"
    # The second turn was prompted with the earlier exchange, not just the new question.
    assert any("x^2" in str(m.content) for m in seen[-1])
    # Per-question counters reset even though the messages carried over.
    assert res.steps_taken == 1

def test_threads_are_isolated():
    def _answering(value, call_id):
        return _FakeChatModel([AIMessage(content="", tool_calls=[
            {"name": "final_answer", "args": {"answer": value}, "id": call_id}])])

    solve("first thread question", model=_answering("1", "x1"), thread_id="thread-a")
    res = solve("second thread question", model=_answering("2", "x2"), thread_id="thread-b")
    assert res.answer == "2" and res.steps_taken == 1


# ---- scoring: LaTeX-wrapped and prose answers ------------------------------

def test_scoring_unwraps_boxed_latex():
    assert is_correct(r"\(\boxed{\text{B}}\)", "B", "mcq")

def test_scoring_takes_the_concluding_letter():
    # The model rules out A before choosing C — the verdict is the last letter.
    assert is_correct("A is false, so the answer is C", "C", "mcq")

def test_scoring_multiple_from_latex():
    assert is_correct(r"\text{A} and \text{C}", "AC", "mcq_multiple")


# ---- scoring: a bare value that matches an option --------------------------

_OPTS = ["441", "398", "312", "409"]

def test_value_credited_as_its_option():
    # The model computed 441 and submitted the number instead of the letter.
    assert is_correct("441", "A", "mcq", _OPTS)

def test_value_matching_wrong_option_still_fails():
    assert not is_correct("398", "A", "mcq", _OPTS)

def test_value_matching_no_option_fails():
    assert not is_correct("999", "A", "mcq", _OPTS)

def test_letter_still_wins_when_present():
    assert is_correct("The answer is A", "A", "mcq", _OPTS)


# ---- scoring: LaTeX values are evaluated, not scraped ----------------------
# Before this, '\frac{47}{3}' was read as 473 (digits concatenated) and
# '2\sqrt{3}' as 23, so every fraction/surd option matched against nonsense.

from src.scoring import _to_number

def test_latex_fraction_is_evaluated():
    assert abs(_to_number(r"\frac{47}{3}") - 47 / 3) < 1e-9

def test_nested_sqrt_inside_fraction():
    assert abs(_to_number(r"\frac{\sqrt{3}}{2}") - 0.8660254) < 1e-6

def test_surd_is_evaluated_not_truncated():
    assert abs(_to_number(r"2\sqrt{3}") - 3.4641016) < 1e-6

def test_whole_expression_beats_first_number():
    # '10*sqrt(5)' is 22.36, not 10.
    assert abs(_to_number("10*sqrt(5)") - 22.3606797) < 1e-6

def test_prose_fallback_still_works():
    assert _to_number("The answer is 54") == 54.0

def test_unparseable_is_none():
    assert _to_number("no numbers here") is None

def test_option_matching_uses_evaluated_latex():
    opts = [r"\frac{47}{3}", r"\frac{46}{3}", "18", "13"]
    assert is_correct("15.3333", "B", "mcq", opts)      # 46/3
    assert not is_correct("15.6667", "B", "mcq", opts)  # that's 47/3, option A

def test_expression_parser_rejects_non_math():
    # Only maths characters reach SymPy's parser.
    assert _to_number("__import__('os')") is None
