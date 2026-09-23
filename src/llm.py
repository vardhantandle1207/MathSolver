"""
LLM factory.

Returns a LangChain chat model so the LangGraph agent can bind tools to it and
invoke it uniformly. The import is done lazily inside the factory so importing
this module stays cheap.
"""

from typing import Any

from .config import settings


def get_chat_model() -> Any:
    """Build the chat model for the configured provider, at temperature 0
    (we want deterministic math, not creativity)."""
    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_host,
            temperature=settings.temperature,
        )

    raise ValueError(f"Unknown provider: {settings.provider!r}")


def message_text(msg: Any) -> str:
    """The model's visible text. A reasoning model can finish a turn with an
    empty `content` and the actual sentence in `reasoning_content` — without
    this we'd throw a correct answer away. Lives here so the agent and the
    baseline extract answers identically."""
    text = (msg.content or "").strip()
    if text:
        return text
    return str(getattr(msg, "additional_kwargs", {}).get("reasoning_content", "")).strip()
