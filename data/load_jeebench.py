"""
Load the maths split of JEEBench and reshape it to our evaluation schema.

JEEBench (Arora et al., "Have LLMs Advanced Enough?") ships on HuggingFace with
columns: subject, description, gold, index, type, question. We keep the 236 maths
problems and map them to the same {question, answer, answer_type, topic} shape as
sample_problems.json, so evaluate.py works on it unchanged.

We bucket by question `type` as the "topic" — it's the informative axis here,
since the tool layer tends to help numeric/integer answers far more than it helps
multiple-correct MCQs.

Usage:
    python -m data.load_jeebench                     # writes data/jeebench_math.json
    python -m data.load_jeebench --limit 50          # smaller slice
    python -m src.evaluate --data data/jeebench_math.json --limit 50
"""

import argparse
import json
import os

from datasets import load_dataset

# JEEBench 'type' -> our answer_type. Integer answers grade like numerics.
_TYPE_MAP = {
    "MCQ": "mcq",
    "MCQ(multiple)": "mcq_multiple",
    "Numeric": "numeric",
    "Integer": "numeric",
}


def build(limit: int | None, out_path: str) -> None:
    ds = load_dataset("daman1209arora/jeebench", split="test")
    math = ds.filter(lambda r: r["subject"] == "math")

    problems = []
    for row in math:
        problems.append({
            "id": row["index"],
            "topic": row["type"],                 # per-type breakdown in the eval
            "answer_type": _TYPE_MAP.get(row["type"], "numeric"),
            "question": row["question"].strip(),
            "answer": str(row["gold"]).strip(),
        })

    if limit:
        problems = problems[:limit]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(problems)} maths problems -> {out_path}")
    # Quick breakdown so you know what you're about to evaluate on.
    from collections import Counter
    print("By type:", dict(Counter(p["topic"] for p in problems)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the JEEBench maths eval set")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="data/jeebench_math.json")
    args = parser.parse_args()
    build(args.limit, args.out)
