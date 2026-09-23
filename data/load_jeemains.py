"""
Load real JEE Main 2025 maths questions and reshape them to our eval schema.

Why this set sits between the other two:

  CalcBench  — my own generator. Method trivial, arithmetic brutal. Isolates
               what the tool layer contributes, but I wrote it.
  JEE Main   — a real public exam, and the one most students actually sit.
               Multi-step reasoning, but each step is a computation a tool can
               do. This is the honest middle ground.
  JEEBench   — JEE Advanced. So hard that a 7B model fails with or without
               tools, which tells you nothing about the tools.

Source: PhysicsWallahAI/JEE-Main-2025-Math on HuggingFace — the January and
April 2025 sittings, 475 maths questions with official answer keys.

Question types in the source:
    0 -> numeric answer, no options   -> answer_type "numeric"
    1 -> standard 4-option MCQ        -> answer_type "mcq"
    2 -> statement-based 4-option MCQ -> answer_type "mcq"

For MCQs the options are appended to the question as (A)-(D) and the gold
answer becomes the option letter, so the model has to pick a letter exactly as
it would in the exam.

Usage:
    python -m data.load_jeemains                  # writes data/jeemains_math.json
    python -m src.evaluate --data data/jeemains_math.json --sample 50
"""

import argparse
import json
import os
from collections import Counter

from datasets import load_dataset

_REPO = "PhysicsWallahAI/JEE-Main-2025-Math"
_LETTERS = "ABCD"


def _render(row: dict, sitting: str, idx: int) -> dict | None:
    """Map one source row to our schema, or None if it isn't usable."""
    question = (row.get("question") or "").strip()
    if not question:
        return None

    qtype = row.get("question_type")
    options = row.get("options") or []
    correct = row.get("correct_options") or []

    if qtype == 0:                      # numeric answer, no options
        answer = str(row.get("answer", "")).strip()
        if not answer:
            return None
        return {
            "id": f"{sitting}-{idx}",
            "topic": "JEE-Main numeric",
            "answer_type": "numeric",
            "question": question,
            "answer": answer,
        }

    # MCQ. Needs exactly four options and a single valid answer index.
    if len(options) != 4 or len(correct) != 1 or not 0 <= correct[0] < 4:
        return None

    lettered = "\n".join(f"({_LETTERS[i]}) {opt}" for i, opt in enumerate(options))
    return {
        "id": f"{sitting}-{idx}",
        "topic": "JEE-Main MCQ",
        "answer_type": "mcq",
        "question": f"{question}\n\n{lettered}",
        "answer": _LETTERS[correct[0]],
        # Kept so the grader can credit a bare value that equals an option.
        "options": list(options),
    }


def build(out_path: str) -> None:
    problems = []
    for sitting in ("jan", "apr"):
        ds = load_dataset(_REPO, sitting)
        split = list(ds.keys())[0]
        for i, row in enumerate(ds[split]):
            item = _render(row, sitting, i)
            if item is not None:
                problems.append(item)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(problems)} JEE Main maths problems -> {out_path}")
    print("By type:", dict(Counter(p["topic"] for p in problems)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the JEE Main eval set")
    parser.add_argument("--out", default="data/jeemains_math.json")
    args = parser.parse_args()
    build(args.out)
