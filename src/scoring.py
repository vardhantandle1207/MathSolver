"""
Grading a math answer is fiddlier than it looks.

Models return '0.5' vs '1/2' vs 'x = 0.5', MCQ answers as 'B' vs '(B)' vs
'Option B', and numeric answers with trailing text. This module normalises both
sides and compares numerically when it can, falling back to a cleaned string
match otherwise.
"""

import re
from fractions import Fraction

from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# Floor for float comparison. The real tolerance is derived from how precisely
# the gold answer is written (see _tolerance_for): an answer key rounded to 2
# decimals must still accept an exact-but-longer '0.3333' against '0.33'.
_NUM_TOL = 1e-3


def _tolerance_for(gold_text: str) -> float:
    """Half a unit in the gold answer's last decimal place — i.e. what rounding
    to that many decimals could have thrown away. Gold '0.33' -> 0.005, so
    '0.3333' passes; gold '9' -> the 1e-3 floor, so '9.5' still fails."""
    match = re.search(r"\.(\d+)", gold_text)
    if not match:
        return _NUM_TOL
    return max(_NUM_TOL, 0.5 * 10 ** -len(match.group(1)))


def _close_enough(pred: float, gold: float, gold_text: str) -> bool:
    """Absolute tolerance from the gold's precision, plus a small relative slack
    so a surd gold (10*sqrt(5) = 22.360679...) accepts a rounded 22.3607."""
    return abs(pred - gold) <= max(_tolerance_for(gold_text), 1e-4 * abs(gold))


def _strip_latex(text: str) -> str:
    r"""Unwrap the LaTeX a maths model reaches for by habit, so
    '\(\boxed{\text{B}}\)' grades as plain 'B'. Command names are dropped
    (they're lower-case, so they can't be mistaken for options anyway) and the
    braces become spaces to keep word boundaries intact."""
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    return re.sub(r"[{}$()\[\]\\]", " ", text)


# Option letters are matched WITHOUT upper-casing the text first: an uppercase
# A/B/C/D is a deliberate option reference, whereas lower-case 'a', 'bad' or
# 'cab' are ordinary English that would otherwise be read as options.
def _extract_mcq_letter(text: str) -> str | None:
    """Find a standalone A/B/C/D, ignoring surrounding punctuation/words.

    Takes the LAST such letter, not the first: when the model answers in prose
    it discusses options in order and states its choice at the end ('A is
    false, so the answer is C'), so the first letter is usually the one it
    rejected. A clean `final_answer` submission is a single letter either way."""
    matches = re.findall(r"\b([ABCD])\b", _strip_latex(text))
    return matches[-1] if matches else None


def _extract_mcq_set(text: str) -> set[str]:
    """All distinct A-D options mentioned. Handles both spaced ('A and D') and
    compact ('AD', 'ABD') forms, since answer keys are usually compact but model
    output usually isn't. We match whole tokens made only of A-D letters, then
    explode them into individual options."""
    letters: set[str] = set()
    for token in re.findall(r"\b[ABCD]+\b", _strip_latex(text)):
        letters.update(token)
    return letters


def _latex_to_expr(text: str) -> str:
    r"""Turn exam/model LaTeX into something SymPy can parse.

    This matters more than it looks. Option values are written like
    '\frac{47}{3}' and '2\sqrt{3}', and a model answers with '10*sqrt(5)'.
    Stripping the backslashes and grabbing the first number reads '\frac{47}{3}'
    as *473* — so before this, every fraction or surd option was matched against
    a nonsense value."""
    text = text.strip()
    for junk in (r"\left", r"\right", r"\displaystyle", r"\,", r"\!", r"\;", "$"):
        text = text.replace(junk, "")
    text = re.sub(r"\\[()\[\]]", "", text)                     # \( \) \[ \]
    # \sqrt first: it turns its braces into parens, so a \frac wrapped around it
    # ('\frac{\sqrt{3}}{2}') is left with brace-free arguments for the rule below.
    while re.search(r"\\sqrt\s*\{([^{}]*)\}", text):
        text = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", text)
    # \frac{a}{b} -> (a)/(b), repeated so nested fractions unwind
    frac = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
    while frac.search(text):
        text = frac.sub(r"((\1)/(\2))", text)
    text = re.sub(r"\\(times|cdot)", "*", text)
    text = re.sub(r"\\(pi|sqrt|log|ln|sin|cos|tan|exp)", r"\1", text)
    text = text.replace("^", "**").replace("{", "(").replace("}", ")")
    return text


# Only these characters reach SymPy's parser. parse_expr() evaluates what it is
# given, and the strings here come from model output, so the input is narrowed
# to maths before it gets there.
_SAFE_EXPR = re.compile(r"^[0-9a-zA-Z+\-*/^().,\s]*$")

_SYMPY_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _sympy_value(text: str) -> float | None:
    """Evaluate a whole expression numerically, or None if it isn't one."""
    if not text or len(text) > 200 or not _SAFE_EXPR.match(text):
        return None
    try:
        expr = parse_expr(text, transformations=_SYMPY_TRANSFORMS, evaluate=True)
        if expr.free_symbols:                 # still has an unknown in it
            return None
        value = complex(expr.evalf())
        if abs(value.imag) > 1e-9:            # complex answers aren't comparable
            return None
        return float(value.real)
    except Exception:                         # noqa: BLE001 - unparseable is just "not a number"
        return None


def _to_number(text: str) -> float | None:
    """Best-effort parse of a numeric answer.

    Tries to evaluate the whole thing first — '10*sqrt(5)' is 22.36, not 10 —
    and only falls back to picking a number out of prose when that fails."""
    text = text.strip()
    # Strip a leading 'x=' style prefix if the model added one.
    text = re.sub(r"^\s*[a-zA-Z]\s*=\s*", "", text)

    compact = text.replace(" ", "")

    # A tuple or list is not a scalar. Without this the prose fallback reads
    # '(7/12, 4/3, 1/4)' as 7.0 — which could score a coordinate triple as a
    # correct single number.
    if re.match(r"^[\(\[\{].*,.*[\)\]\}]$", compact):
        return None

    if re.fullmatch(r"-?\d+/\d+", compact):
        try:
            return float(Fraction(compact))
        except (ValueError, ZeroDivisionError):
            return None

    value = _sympy_value(_latex_to_expr(text))
    if value is not None:
        return value

    # Fallback: the answer is buried in a sentence ('The answer is 54').
    m = re.search(r"-?\d+\.?\d*", compact)
    if m:
        try:
            return float(m.group())
        except ValueError:
            return None
    return None


def _letter_for_value(predicted: str, options: list[str] | None) -> str | None:
    """Map a bare value onto the option it equals.

    A model often computes the right quantity and submits '441' instead of 'A'.
    Marking that wrong measures formatting, not mathematics — and the project's
    claim is about computation. So when no letter is present, we match the
    predicted number against the option values and use whichever it equals.
    Applied identically to the baseline and the agent, so neither is favoured."""
    if not options:
        return None
    pred_num = _to_number(predicted)
    if pred_num is None:
        return None
    for i, option in enumerate(options[:4]):
        opt_num = _to_number(str(option))
        if opt_num is not None and _close_enough(pred_num, opt_num, str(option)):
            return "ABCD"[i]
    return None


def is_correct(predicted: str, gold: str, answer_type: str,
               options: list[str] | None = None) -> bool:
    """answer_type is 'mcq' or 'numeric' (from the dataset)."""
    if predicted is None:
        return False

    if answer_type == "mcq":
        pred_letter = _extract_mcq_letter(predicted)
        if pred_letter is None:
            pred_letter = _letter_for_value(predicted, options)
        gold_letter = _extract_mcq_letter(gold)
        return pred_letter is not None and pred_letter == gold_letter

    if answer_type == "mcq_multiple":
        # 'multiple correct' questions — the selected set must match exactly.
        pred_set = _extract_mcq_set(predicted)
        gold_set = _extract_mcq_set(gold)
        return bool(gold_set) and pred_set == gold_set

    # numeric
    pred_num = _to_number(predicted)
    gold_num = _to_number(gold)
    if pred_num is not None and gold_num is not None:
        return _close_enough(pred_num, gold_num, gold)

    # last resort: normalised string equality
    return predicted.strip().lower() == gold.strip().lower()
