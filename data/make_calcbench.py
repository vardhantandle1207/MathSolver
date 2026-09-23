"""
Build CalcBench — a computation-heavy eval set, generated with SymPy.

Why this exists alongside the real exam set:

A real exam measures whether the *model* can find a method. When it can't, the
baseline and the agent both score near zero and the tool layer's contribution is
buried in noise — you learn nothing about the thing this project is testing.

This set isolates the other variable. Every problem has a method a model states
correctly in one line (integrate a polynomial, take a determinant, apply Vieta)
but numbers ugly enough that doing it mentally is a coin flip. That is exactly
the gap tools are supposed to close, so the baseline→agent delta here is the
measurement, not the accuracy.

Gold answers are computed by SymPy, so they're exact by construction — no
hand-written key to get wrong. Fixed seed, so the set is reproducible.

Usage:
    python -m data.make_calcbench                  # writes data/calcbench.json
    python -m data.make_calcbench --n 40
    python -m src.evaluate --data data/calcbench.json
"""

import argparse
import json
import os
import random

import sympy as sp

x = sp.Symbol("x")


def _poly_str(a: int, b: int, c: int, d: int) -> str:
    """Render a cubic with real signs — '7x^3 - 28x^2 + 30x - 32', not '+ -28'."""
    terms = [f"{a}x^3"]
    for coef, suffix in ((b, "x^2"), (c, "x"), (d, "")):
        terms.append(f"{'-' if coef < 0 else '+'} {abs(coef)}{suffix}")
    return " ".join(terms)


def _poly_integral(rng: random.Random) -> tuple[str, str]:
    """Definite integral of a cubic with awkward coefficients."""
    a, b, c, d = (rng.randint(2, 19) for _ in range(4))
    lo, hi = sorted(rng.sample(range(1, 12), 2))
    expr = a * x**3 + b * x**2 + c * x + d
    gold = sp.integrate(expr, (x, lo, hi))
    return (f"Evaluate the definite integral of {sp.printing.sstr(expr)} "
            f"with respect to x from {lo} to {hi}.", str(gold))


def _determinant(rng: random.Random) -> tuple[str, str]:
    """Determinant of a 4x4 integer matrix — trivial method, brutal by hand."""
    rows = [[rng.randint(-9, 9) for _ in range(4)] for _ in range(4)]
    gold = sp.Matrix(rows).det()
    return (f"Find the determinant of the 4x4 matrix {rows}.", str(gold))


def _binomial(rng: random.Random) -> tuple[str, str]:
    n = rng.randint(18, 30)
    k = rng.randint(6, n // 2)
    return (f"Compute the binomial coefficient C({n}, {k}).", str(sp.binomial(n, k)))


def _linear_system(rng: random.Random) -> tuple[str, str]:
    """3x3 system with a guaranteed integer solution; answer is x+y+z."""
    sol = [rng.randint(-9, 9) for _ in range(3)]
    rows, rhs = [], []
    for _ in range(3):
        coef = [rng.randint(1, 9) for _ in range(3)]
        rows.append(coef)
        rhs.append(sum(c * s for c, s in zip(coef, sol)))
    if sp.Matrix(rows).det() == 0:          # degenerate draw, try again
        return _linear_system(rng)
    eqs = "; ".join(
        f"{r[0]}x + {r[1]}y + {r[2]}z = {v}" for r, v in zip(rows, rhs)
    )
    return (f"Solve the system {eqs}. Give the value of x + y + z.", str(sum(sol)))


def _derivative_at(rng: random.Random) -> tuple[str, str]:
    a, b, c = (rng.randint(2, 13) for _ in range(3))
    point = rng.randint(1, 6)
    expr = a * x**4 - b * x**3 + c * x
    gold = sp.diff(expr, x).subs(x, point)
    # Phrased WITHOUT prime notation on purpose. With tools bound, qwen2.5:7b
    # returns a completely empty response to "Find f'(3)" on a quartic — no text,
    # no tool call — while the same question as "the derivative of f at x = 3"
    # works every time. A quirk of the model's tool template, not of the maths.
    return (f"Let f(x) = {sp.printing.sstr(expr)}. "
            f"Find the derivative of f at x = {point}.", str(gold))


def _gcd_lcm(rng: random.Random) -> tuple[str, str]:
    # Two random numbers are usually coprime, which makes the answer 1 and the
    # question pointless. Plant a shared factor so the algorithm has work to do.
    common = rng.randint(12, 400)
    a, b = common * rng.randint(20, 900), common * rng.randint(20, 900)
    return (f"Find the greatest common divisor of {a} and {b}.", str(sp.gcd(a, b)))


def _modular(rng: random.Random) -> tuple[str, str]:
    base, exp = rng.randint(3, 19), rng.randint(40, 300)
    mod = rng.choice([97, 101, 1009, 10007])
    return (f"Compute {base}^{exp} mod {mod}.", str(pow(base, exp, mod)))


def _polynomial_roots(rng: random.Random) -> tuple[str, str]:
    """Vieta's is one line; the arithmetic on ugly coefficients is not."""
    a = rng.randint(2, 15)
    b, c, d = (rng.randint(-40, 40) for _ in range(3))
    gold = round(float(sp.Rational(-b, a)), 4)
    return (f"Find the sum of the roots of {_poly_str(a, b, c, d)} = 0. "
            f"Give your answer as a decimal to 4 places.", f"{gold:.4f}")


def _matrix_trace(rng: random.Random) -> tuple[str, str]:
    m1 = [[rng.randint(-8, 8) for _ in range(3)] for _ in range(3)]
    m2 = [[rng.randint(-8, 8) for _ in range(3)] for _ in range(3)]
    gold = (sp.Matrix(m1) * sp.Matrix(m2)).trace()
    return (f"Let A = {m1} and B = {m2}. Find the trace of the matrix product AB.",
            str(gold))


def _big_arithmetic(rng: random.Random) -> tuple[str, str]:
    a, b = rng.randint(200, 9999), rng.randint(20, 99)
    c, d, e = rng.randint(3, 9), rng.randint(2, 4), rng.randint(2, 9)
    gold = a * b - c**d * e
    return (f"Evaluate {a} * {b} - {c}^{d} * {e}.", str(gold))


_GENERATORS = {
    "integral": _poly_integral,
    "determinant": _determinant,
    "combinatorics": _binomial,
    "linear_system": _linear_system,
    "derivative": _derivative_at,
    "number_theory": _gcd_lcm,
    "modular": _modular,
    "roots": _polynomial_roots,
    "matrices": _matrix_trace,
    "arithmetic": _big_arithmetic,
}


def build(n: int, out_path: str, seed: int = 7) -> None:
    rng = random.Random(seed)
    names = list(_GENERATORS)
    problems = []

    for i in range(n):
        topic = names[i % len(names)]        # even spread across topics
        question, answer = _GENERATORS[topic](rng)
        problems.append({
            "id": i + 1,
            "topic": topic,
            "answer_type": "numeric",        # every answer is an exact number
            "question": question,
            "answer": answer,
        })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2)

    print(f"Wrote {len(problems)} problems -> {out_path}")
    from collections import Counter
    print("By topic:", dict(Counter(p["topic"] for p in problems)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the CalcBench eval set")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--out", default="data/calcbench.json")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    build(args.n, args.out, args.seed)
