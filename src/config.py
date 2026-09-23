"""
Central config for the math agent.

I keep everything provider-agnostic here so switching between Groq (default,
free tier) and a local Ollama model is a one-line change in the .env file.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # "groq" for the hosted free tier, "ollama" if you're running locally.
    provider: str = os.getenv("LLM_PROVIDER", "groq")

    # Groq gives a generous free tier and Llama 3.3 70B is plenty for this.
    groq_model: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # Local fallback. Qwen2.5 is a good small math model if you have the RAM.
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # Agent knobs. 8 steps is usually enough; anything more and it's probably
    # stuck in a loop, so we bail out.
    max_steps: int = int(os.getenv("MAX_STEPS", "8"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))

    # Cap on a single response. Groq's free tier enforces an output-tokens-per-
    # minute limit and rejects a request whose *expected* output exceeds it, so
    # this has to be set explicitly rather than left to the model's default.
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "900"))

    # Hard timeout (seconds) for any single code execution.
    exec_timeout: int = int(os.getenv("EXEC_TIMEOUT", "10"))

    # Retry transient LLM failures (rate limits, network blips) with backoff.
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))

    # Provider-level retries. The Groq SDK honours the server's Retry-After on a
    # 429, which is what actually paces a long eval against the free tier's
    # output-tokens-per-minute cap — a fixed sleep would be guesswork.
    provider_retries: int = int(os.getenv("PROVIDER_RETRIES", "8"))

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
    """Fail early with a clear message instead of a cryptic 401 later."""
    if settings.provider == "groq" and not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Grab a free key at https://console.groq.com "
            "and put it in your .env file, or switch LLM_PROVIDER=ollama."
        )
