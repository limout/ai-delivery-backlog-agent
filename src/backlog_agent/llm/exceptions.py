"""Provider and decomposition errors. No vendor SDK types leak from here."""

from __future__ import annotations

from backlog_agent.findings.models import GenerationFindings


class AIProviderError(RuntimeError):
    """Base exception for LLM provider failures."""


class AIProviderConfigError(AIProviderError):
    """Missing API key, unknown provider name, or missing optional SDK."""


class AIProviderQuotaError(AIProviderError):
    """Provider quota or rate limit (for example HTTP 429)."""


class AIProviderTimeoutError(AIProviderError):
    """Provider call exceeded the configured timeout."""


class DecompositionValidationError(ValueError):
    """LLM proposal remained invalid after schema and deterministic checks."""

    def __init__(self, errors: list[str], findings: GenerationFindings) -> None:
        detail = "\n".join(f"- {item}" for item in errors) or "- (no details)"
        super().__init__(f"LLM backlog proposal failed validation:\n{detail}")
        self.errors = list(errors)
        self.findings = findings
