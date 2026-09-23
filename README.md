# 🧮 MathSolver-Agent

A **LangGraph** tool-using agent that solves JEE-level mathematics — and
**quantifies how much the tools actually help**, across three benchmarks.

A raw LLM is confident but sloppy at multi-step math: it "knows" the method but
fumbles the algebra. This project wraps a small/free LLM in a **ReAct agent**
(built as an explicit LangGraph `StateGraph`) that offloads every real
computation to **SymPy**, verifies its own results, and reports its accuracy
against a plain-LLM baseline.

The headline artefact isn't "it solves math" — it's the **baseline → agent
accuracy delta**, measured per question type. It is not a uniform win:
**46 % → 98 %** on computation-heavy problems, and a *loss* on real JEE Main
questions where planning, not arithmetic, is the bottleneck. See
[Results](#results).

---

## What it does

- Takes a maths problem in plain English.
- Runs a LangGraph **reason → act → observe** loop: the model plans, writes SymPy
  code, sees the result, and either continues or commits an answer.
- Remembers a conversation when given a `thread_id`, so follow-ups work.
- Grades itself with numeric-tolerance + MCQ-aware scoring (incl. JEEBench's
  multiple-correct questions).
- Ships a Streamlit UI that exposes the **full reasoning trace** — every tool
  call and what it returned.

## Architecture

Built as a custom LangGraph `StateGraph` with two nodes I wrote myself, rather
than the prebuilt `create_react_agent` — that's what lets me keep a readable
trace, intercept a `final_answer` tool for a clean stop condition, and enforce a
step cap.

```
                    ┌──────────────── LangGraph StateGraph ────────────────┐
   problem ───────▶ │                                                      │
                    │   ┌───────┐   has tool calls   ┌───────┐             │
                    │   │ agent │ ─────────────────▶ │ tools │             │
                    │   └───────┘                    └───────┘             │
                    │       ▲  │ no tool calls           │                 │
                    │       │  └──────────▶ END          │ final_answer    │
                    │       │                            │ or step cap ─▶ END
                    │       └────────── loop back ───────┘                 │
                    └──────────────────────────────────────────────────────┘
                        agent = ask the model (tools bound)
                        tools = run SymPy / calculator, feed observations back
```

Three design choices worth calling out:

1. **Custom StateGraph, not the prebuilt agent.** The nodes, routing, and state
   are all explicit in `agent.py` (~90 lines) — I can walk through exactly how a
   step is taken and why the loop terminates.
2. **`final_answer` as a tool.** Instead of guessing when the model is "done"
   from its prose, the model submits its answer via an explicit tool call — an
   unambiguous stop condition the router keys off.

3. **Memory is the graph's, not mine.** Pass a `thread_id` to `solve()` and the
   graph is compiled with a LangGraph checkpointer, so the thread's messages are
   restored on the next call and follow-ups ("now do the same for x³") work. The
   per-question counters (steps, tokens, trace) are reset on each call — only the
   message history accumulates. No `thread_id` means a clean one-shot solve,
   which is what the eval harness wants.

## Project structure

```
mathsolver-agent/
├── app.py                     # Streamlit demo (shows the reasoning trace)
├── requirements.txt
├── .env.example
├── data/
│   ├── sample_problems.json   # 10 verified problems (quick smoke set)
│   ├── make_calcbench.py      # generates the computation-heavy set (SymPy gold)
│   ├── load_jeemains.py       # pulls 475 real JEE Main 2025 questions
│   └── load_jeebench.py       # pulls the 236 JEEBench maths problems
├── src/
│   ├── config.py              # provider + model + agent knobs (env-driven)
│   ├── llm.py                 # chat-model factory (Groq or Ollama)
│   ├── prompts.py             # system prompts
│   ├── tools/
│   │   ├── python_repl.py     # sandboxed subprocess exec w/ timeout
│   │   ├── calculator.py      # AST-restricted safe arithmetic
│   │   └── __init__.py        # tool schemas + dispatch registry
│   ├── agent.py               # the LangGraph StateGraph + ReAct loop
│   ├── baseline.py            # LLM-only solver (the control)
│   ├── scoring.py             # numeric-tolerance + MCQ grader
│   └── evaluate.py            # baseline-vs-agent harness + per-type table
└── tests/
    └── test_tools.py          # offline tests incl. a full-loop test (fake model)
```

## Setup (all free)

```bash
git clone <your-repo-url> && cd mathsolver-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your key
```

You need **one** backend:

**Option A — Groq (hosted, free tier, recommended)**
Free key at <https://console.groq.com>, then in `.env`:
```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
```

**Option B — Ollama (fully local, no key)**
```bash
# install from https://ollama.com
ollama pull qwen2.5:7b-instruct
```
then set `LLM_PROVIDER=ollama` in `.env`.

## Usage

**Interactive UI**
```bash
streamlit run app.py
```

**Solve one problem**
```python
from src import solve
res = solve("Evaluate the integral of x^2 from 0 to 3.")
print(res.answer)     # 9
print(res.trace)      # every tool call + observation

# With a thread_id, the agent remembers the conversation:
solve("Evaluate the integral of x^2 from 0 to 3.", thread_id="chat-1")
solve("Now do the same for x^3 from 0 to 4.",      thread_id="chat-1")
```

**Reproduce the benchmark numbers**
```bash
# computation-heavy set (generated, exact SymPy answers)
python -m data.make_calcbench
python -m src.evaluate --data data/calcbench.json

# real JEE Main 2025 questions, stratified sample, with a wall-clock budget
python -m data.load_jeemains
python -m src.evaluate --data data/jeemains_math.json --sample 50 --max-minutes 120
```

`--sample` draws proportionally per question type with a fixed seed (a plain
head-slice misses whole types); `--max-minutes` stops cleanly and still reports.

**Quick smoke run on the bundled set**
```bash
python -m src.evaluate --data data/sample_problems.json
```

**Tests** (no API key needed — includes a full-loop test with a fake model)
```bash
pytest -q
```

## Results

Model: `qwen2.5:7b-instruct` running locally via Ollama. Baseline and agent use
the same model, the same grader and the same problems — the only difference is
whether the tools are available.

**The short version: tools help enormously when the bottleneck is computation,
and they hurt when the bottleneck is reasoning.** Both results are below.

### CalcBench — computation-heavy (n = 50, generated, seed 7)

| Setup                | Accuracy  |
|----------------------|:---------:|
| Baseline (LLM only)  |   46 %    |
| **Agent (w/ tools)** | **98 %**  |

| Topic         | Baseline | Agent |     | Topic         | Baseline | Agent |
|---------------|:--------:|:-----:|-----|---------------|:--------:|:-----:|
| determinant   |    0 %   | 100 % |     | integral      |   40 %   | 100 % |
| linear system |   20 %   | 100 % |     | combinatorics |   40 %   | 100 % |
| modular arith |   20 %   | 100 % |     | arithmetic    |   60 %   | 100 % |
| number theory |    0 %   |  80 % |     | derivative    |   80 %   | 100 % |
| matrices      |  100 %   | 100 % |     | roots         |  100 %   | 100 % |

The baseline scores **0 % on 4x4 determinants** and 20 % on 3x3 systems: it
knows the method and fumbles the arithmetic, which is exactly the failure the
tool layer removes. Where the numbers are small (matrices, Vieta's) both score
100 % and the tools add nothing — as they should.

<sub>Honest footnote: the raw run scored 48 % / 88 %, with the agent at 0/5 on
derivatives. That was not a maths failure — with tools bound, `qwen2.5:7b`
returns a *completely empty* response to "Find f'(3)" on a quartic, while the
same question as "the derivative of f at x = 3" works every time. After
rephrasing, a re-run of those 5 gave agent 5/5 and baseline 4/5; the table above
substitutes them. Raw log: `results/calcbench_50.log`.</sub>

### JEE Main 2025 — a real exam (n = 27, stratified sample)

| Setup                | Accuracy  |
|----------------------|:---------:|
| Baseline (LLM only)  |  44.4 %   |
| **Agent (w/ tools)** |  29.6 %   |

| Type            | Baseline | Agent | n  |
|-----------------|:--------:|:-----:|:--:|
| MCQ             |  57.9 %  | 42.1 %| 19 |
| Numeric         |  12.5 %  |  0.0 %|  8 |

**The agent loses here, and that is the more interesting result.** JEE Main
questions need a multi-step plan before any arithmetic happens. The tool layer
fixes computation, not planning — and it adds a failure mode: the model has to
translate its plan into correct SymPy, and a subtly wrong translation produces a
confidently wrong number. The baseline, reasoning in prose, keeps more of the
problem in view. 5 of 27 agent runs also hit the step cap without answering.

Two caveats I'd rather state than bury:

- **n = 27 is small** (a wall-clock budget cut the run short), so treat the gap
  as directional, not precise.
- **The MCQ split slightly favours the baseline.** It answers in prose, and the
  grader takes the last option letter it finds, which occasionally lands on the
  right one by luck. The numeric split (12.5 % vs 0 %) has no such escape hatch
  and shows the same direction.

### JEEBench (JEE Advanced) — attempted, not reportable

Runs on this set were abandoned: a 7B model solves almost nothing with or
without tools, so the comparison measures noise. Kept in the repo
(`data/load_jeebench.py`) because ruling a benchmark out is part of the work.

### What this actually shows

Tool-augmentation is not a uniform win. It is a large win (+52 points) exactly
where the model's arithmetic is the weak link, and a net loss where the model's
*planning* is the weak link and SymPy translation becomes one more thing to get
wrong. A bigger base model would likely narrow the second gap; that is the
obvious next experiment.

## Safety & reliability

These are the "what happens when it goes wrong" concerns, handled explicitly
rather than left to chance:

**Error handling**
- Tool crash → caught and returned to the model as a readable `[error]` string,
  so it can self-correct on the next step (implicit retry via re-prompting).
- Bad tool arguments (wrong key from the model) → caught, named, fed back.
- Transient LLM failure (rate limit / network) → retried with exponential
  backoff via LangChain's `.with_retry()` (`MAX_RETRIES`), instead of crashing.

**Termination guarantees** (an agent that can't stop is a liability)
- `max_steps` — hard iteration cap on the loop.
- `max_tokens_budget` — cumulative-token cost cap; the run stops when exceeded.
- **Loop detection** — an identical `(tool, args)` call is refused after
  `max_repeats` tries and the model is told to change approach.
- LangGraph `recursion_limit` — a final backstop below the framework default.
- Every stop path sets an explicit `stopped_reason` (`final_answer`,
  `step_limit`, `token_budget`, `input_too_long`, …) so failures are legible.

**Security** (honest about the tier this is at)
- Input validation: problem text is length-checked; the calculator tool is
  AST-allowlisted (only arithmetic nodes execute).
- The code denylist covers both import spellings (`import os` *and*
  `from os import ...`) plus dunder traversal (`__class__` / `__subclasses__`),
  which is how a sandbox escape usually starts.
- Sandboxed execution: the Python tool runs in an isolated subprocess (`python -I`)
  with a hard timeout, a stripped environment, and a temp working dir.
- Static denylist: code touching the OS, filesystem, network, or `eval`/`exec`
  is refused before it runs.
- **This is defense-in-depth, not a true jail.** A denylist is bypassable; for
  untrusted internet input you'd want gVisor / Firejail / a container with
  seccomp. Access control (auth, per-user rate limits) is out of scope for a
  local demo and would be the first add for a hosted deployment.

## How scoring works

Math grading is fiddly (`0.5` vs `1/2` vs `x = 0.5`; `B` vs `(B)`; multi-answer
`AD` vs `A and D`). `scoring.py` normalises both sides, compares numbers within a
tolerance, matches MCQ letters, and treats multiple-correct answers as sets.

Two details that decide whether the numbers mean anything:

- **Tolerance follows the gold answer's precision.** JEEBench rounds to 2
  decimals but SymPy doesn't, so a fixed `1e-3` would mark a correct `0.3333`
  wrong against a gold of `0.33`. The tolerance is half a unit in the gold's last
  decimal place instead.
- **Option letters are matched case-sensitively.** Upper-casing the text first
  turns ordinary words (`a`, `bad`, `cab`) into options — which would penalise
  the prose-heavy baseline and inflate the agent's apparent lead.

## Limitations & honest notes

- Bounded by the base model's reasoning — tools fix *computation*, not a wrong
  plan. Hard JEE-Advanced problems still fail when the approach is wrong.
- The Python sandbox is subprocess + timeout, not a hardened jail. Fine for a
  trusted local demo; don't expose it to arbitrary internet input as-is.
- Multiple-correct MCQs are the hardest bucket (partial credit isn't given).

## License

MIT.
