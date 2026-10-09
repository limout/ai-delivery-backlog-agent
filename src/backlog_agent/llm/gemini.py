"""Gemini connector. Vendor SDK usage stays in this module."""

from __future__ import annotations

import os
import time
from typing import Any, Mapping

from backlog_agent.llm.exceptions import (
    AIProviderConfigError,
    AIProviderError,
    AIProviderQuotaError,
    AIProviderTimeoutError,
)
from backlog_agent.llm.gemini_schema import to_gemini_response_schema
from backlog_agent.llm.parse import parse_json_object
from backlog_agent.llm.provider import AIProvider

_DEFAULT_MODEL = "gemini-3.5-flash-lite"
_DEFAULT_TIMEOUT_SECONDS = 60.0


def _load_genai() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import errors
    except ImportError as exc:
        raise AIProviderConfigError(
            "google-genai is not installed. Install the gemini extra: "
            "pip install 'limout-ai-delivery-backlog-agent[gemini]'"
        ) from exc
    return genai, errors


class GeminiProvider(AIProvider):
    """Google Gemini JSON generation via the Gen AI SDK."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        client: Any | None = None,
    ) -> None:
        key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        if not key and client is None:
            raise AIProviderConfigError("GEMINI_API_KEY environment variable is not configured.")

        timeout = timeout_seconds
        if timeout is None:
            raw_timeout = os.getenv("GEMINI_TIMEOUT_SECONDS")
            timeout = float(raw_timeout) if raw_timeout else _DEFAULT_TIMEOUT_SECONDS

        self.model = model or os.getenv("GEMINI_MODEL") or _DEFAULT_MODEL
        self.timeout_seconds = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.api_key = key

        if client is not None:
            self.client = client
        else:
            genai, _errors = _load_genai()
            timeout_ms = int(self.timeout_seconds * 1000)
            self.client = genai.Client(
                api_key=key,
                http_options={"timeout": timeout_ms},
            )

    def generate_json(self, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        errors_mod = _try_errors_module()
        started_at = time.perf_counter()
        schema_dict = to_gemini_response_schema(schema)

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": schema_dict,
                    },
                )
                raw_text = getattr(response, "text", None)
                parsed = parse_json_object(raw_text if raw_text is not None else "", "Gemini")
                usage = getattr(response, "usage_metadata", None)
                self.last_usage = {
                    "provider": "gemini",
                    "model": self.model,
                    "duration_ms": max(0, int(round((time.perf_counter() - started_at) * 1000))),
                    "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
                    "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
                    "total_tokens": getattr(usage, "total_token_count", None) if usage else None,
                }
                return parsed
            except Exception as exc:
                mapped = _map_gemini_exception(
                    exc,
                    errors_mod=errors_mod,
                    attempt=attempt,
                    max_retries=self.max_retries,
                    timeout_seconds=self.timeout_seconds,
                )
                if mapped == "retry":
                    time.sleep(self.retry_delay * (2**attempt))
                    continue
                raise

        raise AIProviderError("Gemini request failed after retries.")


def _try_errors_module() -> Any | None:
    try:
        _genai, errors = _load_genai()
    except AIProviderConfigError:
        return None
    return errors


def _map_gemini_exception(
    exc: Exception,
    *,
    errors_mod: Any | None,
    attempt: int,
    max_retries: int,
    timeout_seconds: float,
) -> str | None:
    if isinstance(exc, AIProviderError):
        raise exc
    if isinstance(exc, TimeoutError):
        raise AIProviderTimeoutError(
            f"Gemini request timed out after {timeout_seconds}s."
        ) from exc

    name = type(exc).__name__
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(exc, "code", None)

    is_server = errors_mod is not None and isinstance(exc, errors_mod.ServerError)
    is_client = errors_mod is not None and isinstance(exc, errors_mod.ClientError)
    if not is_server:
        is_server = name == "ServerError"
    if not is_client:
        is_client = name == "ClientError"

    if is_server:
        if attempt >= max_retries:
            raise AIProviderError(
                "Gemini is temporarily unavailable. "
                f"type={name}, message={exc}"
            ) from exc
        return "retry"

    if is_client:
        if status_code == 429:
            raise AIProviderQuotaError(f"Gemini quota/rate limit reached. message={exc}") from exc
        raise AIProviderError(
            f"Gemini request failed. type={name}, status={status_code}, message={exc}"
        ) from exc

    if "timeout" in name.lower() or "timed out" in str(exc).lower():
        raise AIProviderTimeoutError(
            f"Gemini request timed out after {timeout_seconds}s."
        ) from exc

    raise AIProviderError(f"Gemini request failed. type={name}, message={exc}") from exc
