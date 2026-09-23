"""
Grading a math answer is fiddlier than it looks.

Models return '0.5' vs '1/2' vs 'x = 0.5', MCQ answers as 'B' vs '(B)' vs
'Option B', and numeric answers with trailing text. This module normalises both
sides and compares numerically when it can, falling back to a cleaned string
match otherwise.
"""

import re
from fractions import Fraction

# Floor for float comparison. The real tolerance is derived from how precisely
# the gold answer is written (see _tolerance_for): JEEBench rounds to 2 decimals,
# so a gold of '0.33' must still accept an exact-but-longer '0.3333'.
_NUM_TOL = 1e-3


def _tolerance_for(gold_text: str) -> float:
    """Half a unit in the gold answer's last decimal place — i.e. what rounding
    to that many decimals could have thrown away. Gold '0.33' -> 0.005, so
    '0.3333' passes; gold '9' -> the 1e-3 floor, so '9.5' still fails."""
    match = re.search(r"\.(\d+)", gold_text)
    if not match:
        return _NUM_TOL
    return max(_NUM_TOL, 0.5 * 10 ** -len(match.group(1)))


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
    compact ('AD', 'ABD') forms, since JEEBench gold keys are compact but model
    output usually isn't. We match whole tokens made only of A-D letters, then
    explode them into individual options."""
    letters: set[str] = set()
    for token in re.findall(r"\b[ABCD]+\b", _strip_latex(text)):
        letters.update(token)
    return letters


def _to_number(text: str) -> float | None:
    """Best-effort parse of a numeric answer, including fractions like '3/4'."""
    text = text.strip().replace(" ", "")
    # Strip a leading 'x=' style prefix if the model added one.
    text = re.sub(r"^[a-zA-Z]\s*=\s*", "", text)

    if re.fullmatch(r"-?\d+/\d+", text):
        try:
            return float(Fraction(text))
        except (ValueError, ZeroDivisionError):
            return None
    # Grab the first number-looking token from whatever is left.
    m = re.search(r"-?\d+\.?\d*", text)
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
        opt_num = _to_number(_strip_latex(str(option)))
        if opt_num is not None and abs(pred_num - opt_num) <= _tolerance_for(str(option)):
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
        # JEEBench 'multiple correct' — all selected options must match exactly.
        pred_set = _extract_mcq_set(predicted)
        gold_set = _extract_mcq_set(gold)
        return bool(gold_set) and pred_set == gold_set

    # numeric
    pred_num = _to_number(predicted)
    gold_num = _to_number(gold)
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) <= _tolerance_for(gold)

    # last resort: normalised string equality
    return predicted.strip().lower() == gold.strip().lower()
