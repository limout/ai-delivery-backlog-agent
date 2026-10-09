"""Select an LLM provider from the environment. Same convention as Copilot."""

from __future__ import annotations

import os

from backlog_agent.llm.exceptions import AIProviderConfigError
from backlog_agent.llm.provider import AIProvider


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def get_provider(name: str | None = None) -> AIProvider:
    """Return the provider named by ``name`` or ``AI_PROVIDER`` (default ``gemini``)."""

    _load_dotenv()
    selected = (name if name is not None else os.getenv("AI_PROVIDER") or "gemini").strip().lower()
    if selected == "gemini":
        from backlog_agent.llm.gemini import GeminiProvider

        return GeminiProvider()
    if selected == "mock":
        from backlog_agent.llm.mock import MockProvider

        return MockProvider()
    raise AIProviderConfigError(
        f"Unsupported AI_PROVIDER: {selected!r}. Expected 'gemini' or 'mock'."
    )
