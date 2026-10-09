from __future__ import annotations

import pytest

from backlog_agent.llm.exceptions import AIProviderConfigError
from backlog_agent.llm.factory import get_provider
from backlog_agent.llm.mock import MockProvider


def test_get_provider_defaults_to_gemini_and_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(AIProviderConfigError, match="GEMINI_API_KEY"):
        get_provider()


def test_get_provider_selects_gemini_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-for-network")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")

    class StubProvider:
        def __init__(self) -> None:
            self.model = "gemini-test-model"

    monkeypatch.setattr("backlog_agent.llm.gemini.GeminiProvider", StubProvider)

    provider = get_provider()

    assert isinstance(provider, StubProvider)
    assert provider.model == "gemini-test-model"


def test_get_provider_selects_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_PROVIDER", "mock")

    provider = get_provider()

    assert isinstance(provider, MockProvider)


def test_get_provider_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_PROVIDER", "openai")

    with pytest.raises(AIProviderConfigError, match="Unsupported AI_PROVIDER"):
        get_provider()


def test_get_provider_name_argument_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_PROVIDER", "gemini")

    provider = get_provider("mock")

    assert isinstance(provider, MockProvider)
