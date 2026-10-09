"""In-process provider for tests. Never performs network I/O."""

from __future__ import annotations

from typing import Any, Mapping

from backlog_agent.llm.exceptions import AIProviderError
from backlog_agent.llm.provider import AIProvider


class MockProvider(AIProvider):
    """Scriptable JSON provider. Default factory mock has no canned Copilot mapping."""

    def __init__(
        self,
        response: Mapping[str, Any] | None = None,
        *,
        responses: list[Mapping[str, Any] | BaseException] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._queue: list[Mapping[str, Any] | BaseException] = []
        if responses is not None:
            self._queue.extend(responses)
        elif error is not None:
            self._queue.append(error)
        elif response is not None:
            self._queue.append(response)
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((prompt, schema))
        self.last_usage = {
            "provider": "mock",
            "model": "mock",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        if not self._queue:
            raise AIProviderError("MockProvider has no scripted response.")
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return dict(item)
