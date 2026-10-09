from __future__ import annotations

from typing import Any

import pytest

from backlog_agent.llm.exceptions import (
    AIProviderConfigError,
    AIProviderError,
    AIProviderQuotaError,
    AIProviderTimeoutError,
)
from backlog_agent.llm.gemini import GeminiProvider


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = None


class _FakeModels:
    def __init__(
        self,
        *,
        text: str | None = None,
        errors: list[BaseException] | None = None,
    ) -> None:
        self.text = text
        self.errors = list(errors or [])
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return _FakeResponse(self.text or "")


class _FakeClient:
    def __init__(self, models: _FakeModels) -> None:
        self.models = models


class ServerError(Exception):
    pass


class ClientError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _provider(models: _FakeModels, **kwargs: Any) -> GeminiProvider:
    return GeminiProvider(client=_FakeClient(models), retry_delay=0, **kwargs)


def test_gemini_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(AIProviderConfigError, match="GEMINI_API_KEY"):
        GeminiProvider()


def test_gemini_parses_structured_json() -> None:
    models = _FakeModels(text='{"items": [{"local_id": "e1"}]}')
    provider = _provider(models)

    parsed = provider.generate_json("prompt", {"type": "object"})

    assert parsed == {"items": [{"local_id": "e1"}]}
    assert models.calls[0]["config"]["response_mime_type"] == "application/json"
    assert models.calls[0]["config"]["response_schema"] == {"type": "object"}
    assert provider.last_usage is not None
    assert provider.last_usage["provider"] == "gemini"


def test_gemini_parses_fenced_json() -> None:
    models = _FakeModels(text="```json\n{\"ok\": true}\n```")
    parsed = _provider(models).generate_json("prompt", {})
    assert parsed == {"ok": True}


def test_gemini_rejects_malformed_response() -> None:
    models = _FakeModels(text="totally not json")
    with pytest.raises(AIProviderError, match="invalid JSON"):
        _provider(models).generate_json("prompt", {})


def test_gemini_retries_server_error_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("backlog_agent.llm.gemini.time.sleep", slept.append)
    models = _FakeModels(text='{"ok": true}', errors=[ServerError("blip")])
    parsed = _provider(models, max_retries=2).generate_json("prompt", {})
    assert parsed == {"ok": True}
    assert slept == [0]


def test_gemini_quota_is_not_retried() -> None:
    models = _FakeModels(errors=[ClientError("rate", status_code=429)])
    with pytest.raises(AIProviderQuotaError, match="quota"):
        _provider(models).generate_json("prompt", {})
    assert len(models.calls) == 1


def test_gemini_maps_timeout() -> None:
    models = _FakeModels(errors=[TimeoutError("deadline")])
    with pytest.raises(AIProviderTimeoutError, match="timed out"):
        _provider(models).generate_json("prompt", {})


def test_gemini_uses_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "env-model")
    provider = GeminiProvider(client=_FakeClient(_FakeModels(text="{}")), api_key="x")
    assert provider.model == "env-model"
