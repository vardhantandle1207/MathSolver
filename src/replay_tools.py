"""
Replay the tool calls from a finished run against the current tool layer.

A benchmark run costs hours of GPU time, but the *code the model wrote* is
recorded in every trace. So the question "did fixing the tools actually fix
anything?" can be answered without the model at all: take the exact calls it
made, run them through the tools as they stand now, and compare what comes back.

It measures the tool layer in isolation. It cannot tell you the new accuracy —
a working tool changes what the model does next, and that needs a real run.

Usage:
    python -m src.replay_tools results/jeemains_14b.json
"""

import argparse
import json
from collections import Counter

from .tools import TOOL_FUNCS


def _is_failure(observation: str) -> bool:
    return str(observation).startswith(("[error", "[rejected", "[timeout", "[no output"))


def _kind(observation: str) -> str:
    obs = str(observation)
    if obs.startswith("[no output"):
        return "no output (forgot to print)"
    if obs.startswith("[rejected"):
        return "rejected by sandbox"
    if obs.startswith("[timeout"):
        return "timeout"
    if obs.startswith("[error"):
        if "Disallowed expression element" in obs or "Unknown function" in obs:
            return "calculator refused the expression"
        for err in ("NameError", "SyntaxError", "TypeError", "AttributeError",
                    "KeyError", "ValueError", "ZeroDivisionError"):
            if err in obs:
                return f"python {err}"
        return "python error (other)"
    return "ok"


def replay(records: list[dict], limit: int | None = None) -> None:
    before = Counter()
    after = Counter()
    fixed_examples: list[tuple[str, str]] = []
    n = 0

    for rec in records:
        for step in rec.get("trace", []):
            if step.get("type") != "tool_call":
                continue
            name, args = step.get("name"), step.get("args") or {}
            func = TOOL_FUNCS.get(name)
            if func is None:
                continue

            old_obs = step.get("observation", "")
            try:
                new_obs = func(**args)
            except TypeError as exc:
                new_obs = f"[error] Bad arguments for {name}: {exc}"

            before[_kind(old_obs)] += 1
            after[_kind(new_obs)] += 1
            n += 1

            if _is_failure(old_obs) and not _is_failure(new_obs) and len(fixed_examples) < 4:
                arg_text = str(args)[:90]
                fixed_examples.append((arg_text, str(new_obs)[:60]))

            if limit and n >= limit:
                break
        if limit and n >= limit:
            break

    old_fail = sum(c for k, c in before.items() if k != "ok")
    new_fail = sum(c for k, c in after.items() if k != "ok")

    print(f"replayed {n} tool calls from {len(records)} problems\n")
    print(f"{'failure mode':<36}{'before':>8}{'after':>8}")
    for kind in sorted(set(before) | set(after), key=lambda k: -before.get(k, 0)):
        if kind == "ok":
            continue
        print(f"{kind:<36}{before.get(kind,0):>8}{after.get(kind,0):>8}")
    print(f"{'-'*52}")
    print(f"{'TOTAL FAILING':<36}{old_fail:>8}{new_fail:>8}")
    print(f"{'failure rate':<36}{100*old_fail/n:>7.0f}%{100*new_fail/n:>7.0f}%")

    if fixed_examples:
        print("\nexamples now working:")
        for args, out in fixed_examples:
            print(f"  {args}\n    -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay recorded tool calls")
    parser.add_argument("records")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.records, encoding="utf-8") as f:
        if args.records.endswith(".jsonl"):
            records = [json.loads(line) for line in f if line.strip()]
        else:
            records = json.load(f)

    replay(records, args.limit)


if __name__ == "__main__":
    main()
