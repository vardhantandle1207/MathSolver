"""
Failure analysis over a saved eval run.

An accuracy number says the agent lost; it doesn't say why, and "why" is the
only part that tells you what to fix. This reads the records written by
`evaluate.py --out` and sorts every agent failure into one of a few buckets:

    no_submission     hit a limit, or narrated instead of calling final_answer
    unevaluated       submitted symbolic output ('10*sqrt(5)') instead of a value
    unmatched_value   produced a number that matches no option (MCQ only)
    tool_error        every tool call it made came back an error
    no_tool_use       never called a tool — it did the maths in its head
    wrong_maths       submitted a clean, well-formed, wrong answer

Only the last one is a genuine reasoning failure. The rest are plumbing, and
plumbing is fixable — which is the distinction that matters when deciding
whether a bad result is a finding or a bug.

Usage:
    python -m src.analyse results/jeemains_records.json
    python -m src.analyse results/jeemains_records.json --show unevaluated
"""

import argparse
import json
from collections import Counter

from .scoring import _to_number, _extract_mcq_letter, _letter_for_value

_BUCKETS = [
    "no_submission", "unevaluated", "unmatched_value",
    "tool_error", "no_tool_use", "wrong_maths",
]


def classify(rec: dict) -> str:
    """Bucket one failed agent run. Order matters: the earliest matching cause
    is the one to report, since a run that never submitted can't also be a
    reasoning failure."""
    answer = (rec.get("agent_answer") or "").strip()
    trace = rec.get("trace") or []

    if not answer or rec.get("stopped_reason") in {"step_limit", "token_budget",
                                                   "llm_error", "empty_input",
                                                   "input_too_long"}:
        return "no_submission"

    tool_calls = [t for t in trace if t.get("type") == "tool_call"]
    if not tool_calls:
        return "no_tool_use"
    if all(str(t.get("observation", "")).startswith(("[error]", "[rejected]", "[timeout]"))
           for t in tool_calls):
        return "tool_error"

    # A value we cannot evaluate at all: symbolic or malformed output.
    if _to_number(answer) is None:
        return "unevaluated"

    if rec.get("answer_type") == "mcq":
        # It produced a number, but no option letter and no option it matches.
        if _extract_mcq_letter(answer) is None and \
                _letter_for_value(answer, rec.get("options")) is None:
            return "unmatched_value"

    return "wrong_maths"


def analyse(records: list[dict]) -> dict:
    failures = [r for r in records if not r.get("agent_correct")]
    buckets = Counter(classify(r) for r in failures)

    n = len(records)
    agent = sum(1 for r in records if r.get("agent_correct"))
    base = sum(1 for r in records if r.get("baseline_correct"))

    print(f"n = {n}   baseline {base}/{n} ({100*base/n:.1f}%)   "
          f"agent {agent}/{n} ({100*agent/n:.1f}%)\n")

    print(f"agent failures: {len(failures)}")
    print(f"{'bucket':<18}{'count':>7}{'% of failures':>15}")
    for name in _BUCKETS:
        c = buckets.get(name, 0)
        if c:
            print(f"{name:<18}{c:>7}{100*c/len(failures):>14.1f}%")

    fixable = sum(buckets.get(b, 0) for b in _BUCKETS if b != "wrong_maths")
    print(f"\n{fixable}/{len(failures)} failures are plumbing, not mathematics "
          f"({100*fixable/len(failures):.0f}%).")
    print(f"Ceiling if every plumbing failure were fixed: "
          f"{100*(agent+fixable)/n:.1f}%")

    # Cost, because accuracy alone hides that the agent is far more expensive.
    secs = [r.get("seconds", 0) for r in records]
    toks = [r.get("agent_tokens", 0) for r in records]
    if any(secs):
        print(f"\nmedian {sorted(secs)[len(secs)//2]:.0f}s per problem, "
              f"{sum(toks)/max(len(toks),1):.0f} tokens per agent run")

    return dict(buckets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Failure analysis over eval records")
    parser.add_argument("records")
    parser.add_argument("--show", choices=_BUCKETS,
                        help="Print the questions in one bucket.")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    with open(args.records, encoding="utf-8") as f:
        if args.records.endswith(".jsonl"):
            # Streamed copy: one record per line, readable while a run is still
            # going (and all that survives if a run is cut short).
            records = [json.loads(line) for line in f if line.strip()]
        else:
            records = json.load(f)

    analyse(records)

    if args.show:
        print(f"\n--- examples: {args.show} ---")
        shown = 0
        for r in records:
            if r.get("agent_correct") or classify(r) != args.show:
                continue
            print(f"\ngold={r['gold']}  agent={r['agent_answer'][:60]!r}  "
                  f"stop={r['stopped_reason']}")
            print(f"  {r['question'][:150]}...")
            shown += 1
            if shown >= args.limit:
                break


if __name__ == "__main__":
    main()
