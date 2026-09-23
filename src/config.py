"""
Central config for the math agent.

Everything is env-driven, so swapping the model or loosening a safety rail is a
one-line change in .env rather than a code edit.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    provider: str = os.getenv("LLM_PROVIDER", "ollama")

    # Runs locally through Ollama. Qwen2.5 7B is a good small maths model and
    # calls tools reliably, which a lot of small models don't.
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Agent knobs. 8 steps is usually enough; anything more and it's probably
    # stuck in a loop, so we bail out.
    max_steps: int = int(os.getenv("MAX_STEPS", "8"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))

    # Hard timeout (seconds) for any single code execution.
    exec_timeout: int = int(os.getenv("EXEC_TIMEOUT", "10"))

    # Retry transient LLM failures (rate limits, network blips) with backoff.
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))

    # Loop detection: allow the same (tool, args) call at most this many times
    # before we stop re-running it and nudge the model to change approach.
    max_repeats: int = int(os.getenv("MAX_REPEATS", "3"))

    # If the model narrates an answer instead of calling final_answer, re-ask
    # this many times before giving up and scraping its text.
    max_nudges: int = int(os.getenv("MAX_NUDGES", "2"))

    # Cost budget: hard stop once cumulative tokens exceed this. 0 = no limit.
    max_tokens_budget: int = int(os.getenv("MAX_TOKENS_BUDGET", "20000"))

    # Input validation: reject absurdly long problem strings up front.
    max_problem_chars: int = int(os.getenv("MAX_PROBLEM_CHARS", "4000"))


settings = Settings()


def check_config() -> None:
    """Fail early with a clear message instead of a confusing error mid-run."""
    if settings.provider != "ollama":
        raise RuntimeError(
            f"Unknown LLM_PROVIDER {settings.provider!r}. This project runs on "
            "Ollama: install it from https://ollama.com, then "
            f"`ollama pull {settings.ollama_model}`."
        )
