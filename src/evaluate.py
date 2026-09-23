"""
Evaluation harness.

Runs both the baseline and the agent over a problem set, scores them with the
same grader, and prints an accuracy table plus a per-topic breakdown. Run it as:

    python -m src.evaluate --data data/sample_problems.json --limit 20

The per-topic split is worth the extra code: it's what tells you *where* the
tools help most (usually calculus/algebra, less so on tricky word problems).
"""

import argparse
import json
import random
import time
from collections import defaultdict

from .agent import solve
from .baseline import solve_baseline
from .scoring import is_correct
from .config import check_config


def load_problems(path: str, limit: int | None, sample: int | None = None,
                  seed: int = 0) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    if sample:
        # Datasets are often stored grouped by question type, so a plain
        # head-slice can miss a whole type (one set's first 40 problems contained
        # no numeric questions at all). Sample proportionally within each type
        # instead, with a fixed seed so the reported numbers are reproducible.
        by_type: dict[str, list[dict]] = defaultdict(list)
        for p in problems:
            by_type[p.get("topic", "misc")].append(p)

        rng = random.Random(seed)
        picked: list[dict] = []
        for topic, group in sorted(by_type.items()):
            share = max(1, round(sample * len(group) / len(problems)))
            picked.extend(rng.sample(group, min(share, len(group))))
        rng.shuffle(picked)
        return picked[:sample]

    return problems[:limit] if limit else problems


# Give up after this many LLM failures in a row rather than reporting a 0%
# that only measures a broken connection.
_ERROR_ABORT_AFTER = 3


def _accuracy(hits: int, total: int) -> str:
    return f"{100 * hits / total:.1f}%" if total else "n/a"


def run_eval(path: str, limit: int | None, sample: int | None = None,
             max_minutes: float | None = None) -> dict:
    check_config()
    problems = load_problems(path, limit, sample)
    deadline = time.time() + max_minutes * 60 if max_minutes else None

    baseline_hits = agent_hits = 0
    # topic -> [baseline_correct, agent_correct, count]
    by_topic: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    # Why each run ended. A table of 0% is ambiguous on its own — it looks the
    # same whether the model was wrong or the provider was refusing every call.
    stops: dict[str, int] = defaultdict(int)
    consecutive_errors = 0

    for i, item in enumerate(problems, start=1):
        # Wall-clock budget: an unattended run on a slow local model should hand
        # back the problems it finished rather than be killed mid-question and
        # lose the summary entirely.
        if deadline and time.time() > deadline:
            print(f"\n[time budget reached — stopping after {i - 1} problems]")
            problems = problems[:i - 1]
            break

        question = item["question"]
        gold = str(item["answer"])
        atype = item.get("answer_type", "numeric")
        topic = item.get("topic", "misc")

        base_pred = solve_baseline(question)
        options = item.get("options")
        base_ok = is_correct(base_pred, gold, atype, options)

        agent_res = solve(question)
        agent_ok = is_correct(agent_res.answer, gold, atype, options)
        stops[agent_res.stopped_reason] += 1

        # Fail loudly. If the provider is refusing every call (quota, outage),
        # every question scores as a miss and the run quietly reports 0% — a
        # number that looks like a result but measures nothing.
        consecutive_errors = consecutive_errors + 1 if agent_res.stopped_reason == "llm_error" else 0
        if consecutive_errors >= _ERROR_ABORT_AFTER:
            detail = next((t.get("content", "") for t in agent_res.trace
                           if str(t.get("content", "")).startswith("[error]")), "")
            raise RuntimeError(
                f"Aborting: {consecutive_errors} consecutive LLM failures — the "
                f"results so far are not a measurement of the agent.\n{detail}"
            )

        baseline_hits += base_ok
        agent_hits += agent_ok
        by_topic[topic][0] += base_ok
        by_topic[topic][1] += agent_ok
        by_topic[topic][2] += 1

        # A one-line progress log so long runs aren't a black box.
        flag = lambda ok: "OK " if ok else "  X"
        print(f"[{i:>3}/{len(problems)}] {topic:<12} "
              f"base:{flag(base_ok)}({base_pred[:15]:<15}) "
              f"agent:{flag(agent_ok)}({agent_res.answer[:15]:<15}) gold={gold}")

    total = len(problems)
    print("\n" + "=" * 60)
    print(f"Baseline (LLM only): {_accuracy(baseline_hits, total)}  "
          f"({baseline_hits}/{total})")
    print(f"Agent (with tools) : {_accuracy(agent_hits, total)}  "
          f"({agent_hits}/{total})")
    print("-" * 60)
    print(f"{'topic':<14}{'baseline':>10}{'agent':>10}{'n':>6}")
    for topic, (b, a, n) in sorted(by_topic.items()):
        print(f"{topic:<14}{_accuracy(b, n):>10}{_accuracy(a, n):>10}{n:>6}")
    print("-" * 60)
    print("agent stop reasons:", dict(sorted(stops.items(), key=lambda kv: -kv[1])))
    print("=" * 60)

    return {
        "baseline_accuracy": baseline_hits / total if total else 0,
        "agent_accuracy": agent_hits / total if total else 0,
        "n": total,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline vs agent evaluation")
    parser.add_argument("--data", default="data/sample_problems.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N problems (handy for a quick check).")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="Stop cleanly once this many minutes have elapsed "
                             "and report on what finished.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Run a stratified random sample of N problems "
                             "(proportional per question type, fixed seed).")
    args = parser.parse_args()

    start = time.time()
    run_eval(args.data, args.limit, args.sample, args.max_minutes)
    print(f"Done in {time.time() - start:.1f}s")
