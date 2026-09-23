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
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

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
             max_minutes: float | None = None, out_path: str | None = None,
             workers: int = 1) -> dict:
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
    # Full per-problem records. A model run is expensive and a grader change is
    # cheap, so every prediction is kept: re-scoring and error analysis then run
    # offline instead of costing another pass over the benchmark.
    records: list[dict] = []

    def _one(item: dict) -> dict:
        """Solve one problem twice — once without tools, once with — and score
        both. Pure with respect to shared state, so the pool can run several at
        once: each call builds its own graph and its own model client."""
        t0 = time.time()
        question = item["question"]
        gold = str(item["answer"])
        atype = item.get("answer_type", "numeric")
        options = item.get("options")

        base_pred = solve_baseline(question)
        agent_res = solve(question)
        return {
            "id": item.get("id"),
            "topic": item.get("topic", "misc"),
            "answer_type": atype,
            "question": question,
            "gold": gold,
            "options": options,
            "baseline_answer": base_pred,
            "baseline_correct": bool(is_correct(base_pred, gold, atype, options)),
            "agent_answer": agent_res.answer,
            "agent_correct": bool(is_correct(agent_res.answer, gold, atype, options)),
            "agent_steps": agent_res.steps_taken,
            "agent_tokens": agent_res.tokens_used,
            "stopped_reason": agent_res.stopped_reason,
            "seconds": round(time.time() - t0, 1),
            "trace": agent_res.trace,
        }

    # A local model is idle much of the time a single problem runs (the agent is
    # parsing, running SymPy, waiting on a subprocess), so a few problems in
    # flight raises throughput well before the GPU saturates.
    flag = lambda ok: "OK " if ok else "  X"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for item in problems:
            if deadline and time.time() > deadline:
                print(f"\n[time budget reached — submitted {len(futures)} problems]")
                break
            futures.append(pool.submit(_one, item))

        # as_completed, not submission order: with N problems in flight, waiting
        # on the first one means a long run prints nothing for many minutes and
        # looks hung. Each line is numbered as it lands instead.
        done = 0
        for future in as_completed(futures):
            rec = future.result()
            done += 1
            i = done
            records.append(rec)
            stops[rec["stopped_reason"]] += 1

            base_ok, agent_ok = rec["baseline_correct"], rec["agent_correct"]
            baseline_hits += base_ok
            agent_hits += agent_ok
            by_topic[rec["topic"]][0] += base_ok
            by_topic[rec["topic"]][1] += agent_ok
            by_topic[rec["topic"]][2] += 1

            consecutive_errors = (consecutive_errors + 1
                                  if rec["stopped_reason"] == "llm_error" else 0)
            if consecutive_errors >= _ERROR_ABORT_AFTER:
                raise RuntimeError(
                    f"Aborting: {consecutive_errors} consecutive LLM failures — "
                    "the results so far are not a measurement of the agent."
                )

            print(f"[{i:>3}/{len(futures)}] {rec['topic']:<18} "
                  f"base:{flag(base_ok)}({str(rec['baseline_answer'])[:15]:<15}) "
                  f"agent:{flag(agent_ok)}({str(rec['agent_answer'])[:15]:<15}) "
                  f"gold={rec['gold']}")

    problems = [r for r in records]
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

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"wrote {len(records)} records -> {out_path}")

    return {
        "baseline_accuracy": baseline_hits / total if total else 0,
        "agent_accuracy": agent_hits / total if total else 0,
        "n": total,
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline vs agent evaluation")
    parser.add_argument("--data", default="data/sample_problems.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N problems (handy for a quick check).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Solve this many problems concurrently. A local "
                             "model has idle time per problem, so 2-4 raises "
                             "throughput on one GPU.")
    parser.add_argument("--out", default=None,
                        help="Write full per-problem records (incl. traces) to "
                             "this JSON file, for offline re-scoring/analysis.")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="Stop cleanly once this many minutes have elapsed "
                             "and report on what finished.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Run a stratified random sample of N problems "
                             "(proportional per question type, fixed seed).")
    args = parser.parse_args()

    start = time.time()
    run_eval(args.data, args.limit, args.sample, args.max_minutes, args.out,
             args.workers)
    print(f"Done in {time.time() - start:.1f}s")
