"""LLM backend interface. Business logic depends on this ABC only."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class AIProvider(ABC):
    """JSON LLM backend used by backlog decomposition."""

    last_usage: dict[str, Any] | None = None

    @abstractmethod
    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        """Return a JSON object matching ``schema`` as closely as the vendor allows.

        Application code must still validate the dict against the Pydantic schema.
        """
