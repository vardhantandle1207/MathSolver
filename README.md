# 🧮 MathSolver-Agent

A **LangGraph** tool-using agent that solves JEE-level mathematics — and measures
**how much the tools actually help**, against the same model with no tools.

A language model is confident but unreliable at multi-step arithmetic: it knows
the method and fumbles the algebra. This project wraps one in a **ReAct agent**
(an explicit LangGraph `StateGraph`) that hands every calculation to **SymPy**,
and then tests whether that helps — on generated computation-heavy problems and
on real JEE Main 2025 questions.

**The headline is not an accuracy number, it's a boundary:**

| Benchmark | Baseline | Agent | Gain | significance |
|---|---:|---:|---:|---|
| **CalcBench** — computation-heavy, n=50 | 58.0 % | **86.0 %** | **+28.0** | p = 0.007 |
| **JEE Main 2025** — real exam, n=150 | 37.3 % | 41.3 % | +4.0 | p = 0.48 (not significant) |

Tools produce a large, statistically solid gain where the bottleneck is
*computation*, and no measurable gain where the bottleneck is *choosing a
method*. Both results come from the same model, the same grader and the same
code — only the tools differ. Details, including why the JEE Main result is
reported as a null, are in [Results](#results).

Model: `qwen2.5:14b-instruct` via Ollama, on one RTX A5000.

---

## What it does

- Takes a maths problem in plain English.
- Runs a LangGraph **reason → act → observe** loop: the model plans, writes
  SymPy, sees the result, and either continues or commits an answer.
- Remembers a conversation when given a `thread_id`, so follow-ups work.
- Grades itself with a SymPy-backed grader that evaluates LaTeX, fractions and
  surds rather than pattern-matching them.
- Ships a Streamlit UI that exposes the **full reasoning trace** — every tool
  call and what it returned.

## Architecture

A custom `StateGraph` with three nodes, rather than the prebuilt
`create_react_agent` — that is what makes the trace readable, lets a
`final_answer` tool serve as the stop condition, and leaves room for real
safety limits.

```
                    ┌──────────────── LangGraph StateGraph ────────────────┐
   problem ───────▶ │                                                      │
                    │   ┌───────┐   has tool calls   ┌───────┐             │
                    │   │ agent │ ─────────────────▶ │ tools │             │
                    │   └───────┘                    └───────┘             │
                    │     ▲   │ answered in prose        │                 │
                    │     │   └──────▶ ┌───────┐         │ final_answer    │
                    │     │            │ nudge │         │ or a limit ─▶ END
                    │     │            └───────┘         │                 │
                    │     └────────── loop back ─────────┘                 │
                    └──────────────────────────────────────────────────────┘
        agent = ask the model (tools bound)
        tools = run SymPy / calculator, feed observations back
        nudge = the model narrated an answer instead of submitting one; re-ask
```

Four design choices worth explaining:

1. **Custom StateGraph, not the prebuilt agent.** Nodes, routing and state are
   explicit in `agent.py`, so every step and every exit path is inspectable.
2. **`final_answer` as a tool.** Rather than guessing when the model is done
   from its prose, the model submits through an explicit tool call — an
   unambiguous stop condition the router keys off.
3. **A nudge node.** Small models often narrate the answer and forget to submit
   it. Instead of scraping a number out of a sentence, the graph re-asks once.
   (Scraping is worse than it sounds: the first number in `x^2 = 4` is 2.)
4. **Memory is the graph's, not mine.** Pass a `thread_id` and the graph is
   compiled with a LangGraph checkpointer, so the thread's messages return on
   the next call. Per-question counters reset each call; only messages
   accumulate. No `thread_id` means a clean one-shot solve, which is what the
   eval harness wants.

## Project structure

```
mathsolver-agent/
├── app.py                     # Streamlit demo (shows the reasoning trace)
├── data/
│   ├── make_calcbench.py      # generates the computation-heavy set (SymPy gold)
│   ├── load_jeemains.py       # pulls 475 real JEE Main 2025 questions
│   └── sample_problems.json   # 10 problems for a quick smoke run
├── src/
│   ├── agent.py               # the StateGraph: agent / tools / nudge + routing
│   ├── baseline.py            # same model, no tools (the control)
│   ├── tools/
│   │   ├── python_repl.py     # sandboxed subprocess exec w/ timeout
│   │   ├── calculator.py      # AST-allowlisted arithmetic + maths functions
│   │   └── __init__.py        # tool schemas + dispatch registry
│   ├── scoring.py             # SymPy-backed grader (LaTeX, surds, MCQ, tolerance)
│   ├── evaluate.py            # baseline-vs-agent harness, concurrent, resumable
│   ├── analyse.py             # sorts failures into plumbing vs wrong maths
│   ├── replay_tools.py        # re-runs recorded tool calls against new tools
│   ├── config.py              # model + agent knobs (env-driven)
│   ├── llm.py                 # chat-model factory
│   └── prompts.py             # system prompts
├── scripts/
│   ├── setup_gpu_server.sh    # one-shot setup on a no-sudo GPU box
│   ├── probe_server.sh        # read-only survey of an unfamiliar machine
│   └── watch_run.sh           # mirror a remote run and show live progress
├── results/                   # the two runs the tables above are computed from
└── tests/test_tools.py        # 60 tests, no network and no model required
```

## Setup

```bash
git clone https://github.com/vardhantandle1207/MathSolver.git && cd MathSolver
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The model runs locally through **Ollama** — no API key, no rate limits:

```bash
# install from https://ollama.com, then:
ollama pull qwen2.5:14b-instruct
```

Any tool-calling Ollama model works; set `OLLAMA_MODEL` in `.env` to switch.
A 14B model needs roughly 10 GB of VRAM.

**On a shared GPU server** (no sudo, old Python, old driver), one script handles
it:

```bash
GPU=0 bash scripts/setup_gpu_server.sh
```

It installs Ollama and Python 3.12 under `$HOME`, pins one GPU, writes `.env`,
pulls the model and runs the tests. Nothing is installed system-wide. See
[Running on a shared GPU box](#running-on-a-shared-gpu-box) for the two traps
that cost me hours.

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

**Reproduce the numbers**
```bash
# computation-heavy set (generated, exact SymPy answers)
python -m data.make_calcbench
python -m src.evaluate --data data/calcbench.json --workers 4 \
    --out results/calcbench_14b.json

# real JEE Main 2025 questions, stratified sample
python -m data.load_jeemains
python -m src.evaluate --data data/jeemains_math.json --sample 150 \
    --workers 4 --out results/jeemains_150_fixed.json
```

`--sample` draws proportionally per question type with a fixed seed (a plain
head-slice misses whole types); `--workers` solves several problems at once;
`--max-minutes` stops cleanly and still reports; `--out` saves every prediction
and trace, streaming to `.jsonl` as it goes so a killed run keeps its results.

**Analyse and re-measure**
```bash
python -m src.analyse results/jeemains_150_fixed.json          # failure buckets
python -m src.analyse results/jeemains_150_fixed.json --show unevaluated
python -m src.replay_tools results/jeemains_150_fixed.json     # tool health
```

**Tests** (no API key, no model, no network)
```bash
pytest -q
```

## Results

Both benchmarks use `qwen2.5:14b-instruct`, the same grader and the same
problems for each arm. The only difference between baseline and agent is whether
the tools exist. Significance is **McNemar's exact test** on the paired
outcomes, which is the right test here because both systems answer the same
questions.

### CalcBench — computation-heavy (n = 50, generated, seed 7)

| Setup                | Accuracy  |
|----------------------|:---------:|
| Baseline (LLM only)  |  58.0 %   |
| **Agent (w/ tools)** | **86.0 %** |

19 questions only the agent got right, 5 only the baseline — **p = 0.0066**.

| Topic         | Baseline | Agent |     | Topic         | Baseline | Agent |
|---------------|:--------:|:-----:|-----|---------------|:--------:|:-----:|
| determinant   |    0 %   | 100 % |     | integral      |   60 %   |  80 % |
| number theory |    0 %   | 100 % |     | arithmetic    |   40 %   |  40 % |
| modular arith |   20 %   | 100 % |     | derivative    |  100 %   | 100 % |
| combinatorics |   60 %   | 100 % |     | linear system |  100 %   | 100 % |
| roots         |  100 %   | 100 % |     | matrices      |  100 %   |  40 % |

The baseline scores **0 % on 4×4 determinants** and 20 % on modular arithmetic:
it knows the method and fumbles the numbers, which is exactly what the tool
layer removes. Where the numbers are small (Vieta's, small matrices) both are at
100 % and the tools add nothing — as they should.

One honest anomaly: the agent scores **40 % on matrices where the baseline gets
100 %**. Five problems, so it is two questions' worth of difference, but it is a
real regression and not noise-free — the records are in `results/` and it is the
first thing I would dig into next.

### JEE Main 2025 — a real exam (n = 150, stratified sample of 475)

| Setup                | Accuracy  |
|----------------------|:---------:|
| Baseline (LLM only)  |  37.3 %   |
| Agent (w/ tools)     |  41.3 %   |

| Type            | Baseline | Agent |  n  |
|-----------------|:--------:|:-----:|:---:|
| MCQ             |  41.7 %  | 46.7 %| 120 |
| Numeric         |  20.0 %  | 20.0 %|  30 |

**This +4 points is not statistically significant.** 28 questions only the agent
got, 22 only the baseline; McNemar gives **p = 0.48**, meaning a gap this size
arises by chance about half the time. The honest reading is that on real exam
questions the tool layer is **not measurably better than no tools** — not that
it is 4 points better.

Why? The failure analysis answers it directly:

```
$ python -m src.analyse results/jeemains_150_fixed.json
agent failures: 88
  wrong_maths        64   72.7%      <- clean, well-formed, wrong answers
  no_submission       7    8.0%
  tool_error          6    6.8%
  no_tool_use         6    6.8%
  unmatched_value     4    4.5%
  unevaluated         1    1.1%
24/88 failures are plumbing, not mathematics (27%)
Ceiling if every plumbing failure were fixed: 57.3%
```

**73 % of the agent's failures are clean but wrong answers** — it picked the
wrong method, then executed it perfectly. A tool cannot fix a wrong plan. On
CalcBench, by contrast, **100 %** of the few remaining failures are wrong maths
too, but there the method was never the hard part, so tools had room to help.

### The tool layer was broken, and fixing it did not move accuracy

The first JEE Main run exposed something worth reporting: of 1,893 tool calls,
**1,051 (56 %) returned an error or nothing.** A failed tool call means the agent
falls back to mental arithmetic — at which point it *is* the baseline.

| Cause | count | fix |
|---|---:|---|
| `[no output]` — model never called `print()` | 387 | echo a bare trailing expression, as a REPL does |
| calculator refused every function call | 240 | allowlist named maths functions |
| `NameError` on `exp`, `log`, `binomial` … | 168 | import SymPy's full namespace |
| `SyntaxError` | 81 | the model's own fault |

`src/replay_tools.py` re-runs recorded calls against the current tools, so a
tool fix is measurable in minutes instead of GPU-hours:

```
$ python -m src.replay_tools <old run>
TOTAL FAILING   1051 -> 572
failure rate     56% -> 30%
```

**And yet accuracy barely moved.** Re-running the same 150 questions with the
repaired tools: 26 answers became correct, 22 became wrong (p = 0.67), with the
tool failure rate down from 53 % to 31 %. That is a genuine negative result:
**tool reliability was not the binding constraint on this benchmark.** It is
also the strongest evidence for the conclusion above — we halved tool failures
and the exam score stayed put, because the exam is bottlenecked on reasoning.

### Cost

| | median time | tokens per agent run |
|---|---:|---:|
| CalcBench | 53 s | 2,131 |
| JEE Main | 161 s | 6,366 |

The agent is several times more expensive than the baseline. On CalcBench that
buys 28 points. On JEE Main it buys nothing measurable — worth stating plainly,
because "add tools" is not free.

## Safety & reliability

**Termination** (an agent that cannot stop is a liability)
- `max_steps` — hard iteration cap.
- `max_tokens_budget` — cumulative-token cost cap.
- **Loop detection** — an identical `(tool, args)` call is refused after
  `max_repeats` tries and the model is told to change approach.
- `max_nudges` — how many times the agent is re-asked to submit properly.
- Every exit path sets an explicit `stopped_reason` (`final_answer`,
  `step_limit`, `token_budget`, `no_final_answer_call`, `llm_error`, …), so the
  eval can report *why* runs ended, not just that they did.

**Error handling**
- A tool crash is caught and returned to the model as a readable `[error]`
  string, so it can self-correct next step.
- A provider failure that survives retries becomes one scored miss with
  `stopped_reason="llm_error"`, so one bad response cannot abort a long run.
- The harness **aborts after 3 consecutive LLM failures**. A dead connection
  otherwise produces a full run of zeros, which looks like a result and measures
  nothing. (This happened; hence the guard.)

**Security**
- The Python tool runs in an isolated subprocess (`python -I`) with a hard
  timeout, a stripped environment and a temp working directory.
- A static denylist refuses OS/filesystem/network access, both import spellings
  (`import os` and `from os import …`) and dunder traversal (`__subclasses__`).
- The calculator is AST-allowlisted: only arithmetic nodes and named functions
  from one dict execute, so nothing else can be called.
- **This is defence in depth, not a jail.** A denylist is bypassable. For
  untrusted input you would want gVisor, Firejail or a seccomp container.

## How scoring works

Grading maths output is where a benchmark quietly goes wrong. Four details that
decide whether the numbers mean anything:

- **Expressions are evaluated, not pattern-matched.** `\frac{47}{3}` is 15.67,
  not 473, and `10*sqrt(5)` is 22.36, not 10. An earlier regex version read the
  digits out of LaTeX and matched every fraction option against nonsense.
- **Tolerance follows the gold answer's precision.** Keys round to 2 decimals
  but SymPy does not, so the tolerance is half a unit in the gold's last decimal
  place rather than a fixed epsilon.
- **A bare value counts as the option it equals.** A model that computes 441 and
  submits the number rather than `A` has done the mathematics; marking that wrong
  measures formatting. Applied identically to both arms.
- **Option letters are matched case-sensitively, and tuples are not scalars.**
  Upper-casing turns `a`, `bad` and `cab` into options; reading `(7/12, 4/3, 1/4)`
  as `7.0` would score a coordinate triple as a correct number.

## Running on a shared GPU box

Two traps, both of which cost hours and neither of which announces itself:

- **Ollama ≥ 0.13 requires NVIDIA driver 550+.** On an older driver it does not
  fail — it silently falls back to a Vulkan backend that segfaults. The setup
  script pins v0.12.11 (CUDA 12.2) and prints the selected backend, because
  "Vulkan instead of CUDA" otherwise looks like "working, just slow".
- **More concurrency is faster only while it fits in VRAM.** Eight slots on a
  24 GB card hit `cudaMalloc: out of memory`; the runner then died and reloaded
  an 8 GB model in a loop, and throughput fell from 63 problems/hour to 12. Four
  slots was the right answer.

Everything the script installs lives under `$HOME`, and one GPU is pinned with
`CUDA_VISIBLE_DEVICES` so the rest of a shared machine stays free.

## Limitations & honest notes

- **The JEE Main result is a null, not a win.** n=150, p=0.48. A larger sample
  would tighten the interval; it would not turn +4 points into a finding.
- **Bounded by the base model's reasoning.** Tools fix computation, not a wrong
  plan — measured, not assumed: 73 % of agent failures are clean wrong answers,
  and repairing the tool layer did not move the score.
- **One model, one seed.** Everything here is `qwen2.5:14b-instruct` at
  temperature 0. A stronger model would plausibly change the JEE Main picture,
  and that is the obvious next experiment.
- **CalcBench is my own generator.** Gold answers are exact by construction
  (SymPy computes them), which removes key errors but not the fact that I chose
  the problem types. It is a diagnostic, not a public benchmark — which is why
  the real exam is reported alongside it.
- **The matrices regression on CalcBench is unexplained** (agent 40 % vs
  baseline 100 %, n=5). Flagged rather than smoothed over.
- The sandbox is subprocess + timeout, not a hardened jail.

## License

MIT — see [LICENSE](LICENSE).
