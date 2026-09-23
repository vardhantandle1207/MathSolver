"""
Sandboxed Python execution — the workhorse tool.

The LLM is good at *deciding* what to compute but bad at actually doing the
arithmetic/algebra reliably, so we let it write SymPy/NumPy code and run it in a
throwaway subprocess. Running it out-of-process (instead of exec() inline) buys
us a real timeout and crash isolation cheaply.

Security note — read this honestly:
This is *defense in depth*, not a true jail. We do three things:
  1. Static denylist — reject code that references OS/filesystem/network/eval
     before it ever runs. This blocks the obvious stuff a hallucinated snippet
     might emit; it is NOT bypass-proof (a determined attacker can obfuscate).
  2. Subprocess isolation + hard timeout — a crash or runaway loop can't take
     the agent down.
  3. Stripped environment + temp working dir — no inherited secrets, nothing
     useful to write to.
For a trusted local math demo this is appropriate. For untrusted internet input
you'd want a real sandbox (gVisor / Firejail / a container / seccomp). That
tradeoff is called out in the README rather than hidden.
"""

import os
import re
import subprocess
import sys
import tempfile
import textwrap

from ..config import settings

# Patterns we refuse to run. A math tool never needs any of these, so blocking
# them costs nothing and removes the scariest failure modes.
# One pattern covers both spellings of an import — `import os` AND the
# `from os import ...` form, which a module-name-only list would let straight
# through.
_DENY_MODULES = (
    "os|sys|subprocess|socket|shutil|requests|urllib|pathlib|importlib|"
    "ctypes|pickle|multiprocessing|http|ftplib|glob|tempfile"
)
_DENY_PATTERNS = [
    rf"\b(?:import|from)\s+(?:{_DENY_MODULES})\b",
    r"\b__import__\b", r"\beval\s*\(", r"\bexec\s*\(", r"\bopen\s*\(",
    r"\bos\.", r"\bsys\.", r"\bsocket\.",
    # Dunder attribute access is how a sandbox escape usually starts
    # (e.g. ().__class__.__bases__[0].__subclasses__()).
    r"__subclasses__|__bases__|__globals__|__builtins__",
]
_DENY_RE = re.compile("|".join(_DENY_PATTERNS))

_PREAMBLE = textwrap.dedent(
    """
    import math
    import sympy as sp
    from sympy import (
        symbols, solve, simplify, expand, factor, diff, integrate,
        limit, Matrix, Rational, sqrt, sin, cos, tan, pi, oo, Eq, S,
    )
    import numpy as np
    """
).strip()


def _validate(code: str) -> str | None:
    """Return an error string if the code should be refused, else None."""
    if len(code) > 5000:
        return "[rejected] Code too long — keep the snippet focused."
    hit = _DENY_RE.search(code)
    if hit:
        return (f"[rejected] Disallowed operation: {hit.group().strip()!r}. "
                "This tool is for pure math (SymPy/NumPy) only — no OS, file, "
                "or network access.")
    return None


def run_python(code: str) -> str:
    """Execute `code` and return whatever it printed (or the error)."""
    refusal = _validate(code)
    if refusal:
        return refusal

    full_source = _PREAMBLE + "\n\n" + code

    # Run in a temp dir with a stripped environment: nothing inherited that a
    # snippet could read or leak, and PATH kept minimal so Python still starts.
    with tempfile.TemporaryDirectory() as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", full_source],  # -I: isolated mode
                capture_output=True,
                text=True,
                timeout=settings.exec_timeout,
                cwd=workdir,
                env={"PATH": os.environ.get("PATH", "/usr/bin")},
            )
        except subprocess.TimeoutExpired:
            return (f"[timeout] Execution exceeded {settings.exec_timeout}s. "
                    "Try a more direct approach or a closed-form solve().")

    if proc.returncode != 0:
        # Hand the traceback back — the model is usually good at fixing its own
        # SyntaxError / undefined-symbol mistakes on the next turn.
        return f"[error]\n{proc.stderr.strip()}"

    out = proc.stdout.strip()
    return out if out else "[no output] Did you forget to print() the result?"
