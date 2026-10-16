"""Standalone HTTP API for LLM-backed backlog generation.

Does not replace the CLI. Does not call the deterministic generator.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from backlog_agent.backlog.decompose import decompose_backlog
from backlog_agent.cli import serialize_preview
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.llm.exceptions import (
    AIProviderConfigError,
    AIProviderError,
    AIProviderQuotaError,
    AIProviderTimeoutError,
    DecompositionValidationError,
)
from backlog_agent.llm.factory import get_provider

_LOG = logging.getLogger("backlog_agent.api")
_TOKEN_ENV = "BACKLOG_API_TOKEN"

app = FastAPI(
    title="AI Delivery Backlog Agent",
    description="Generate a canonical backlog from a COMPLETE Copilot workflow JSON envelope.",
    version="0.1.0",
)


def _configured_token() -> str:
    return os.getenv(_TOKEN_ENV, "").strip()


def _tokens_match(provided: str, expected: str) -> bool:
    left = provided.encode("utf-8")
    right = expected.encode("utf-8")
    if len(left) != len(right):
        hmac.compare_digest(right, right)
        return False
    return hmac.compare_digest(left, right)


def _error(
    status_code: int,
    code: str,
    message: str,
    validation_errors: list[dict[str, str]] | None = None,
) -> JSONResponse:
    content: dict[str, Any] = {"error": code, "message": message}
    if validation_errors:
        content["validation_errors"] = validation_errors
    return JSONResponse(status_code=status_code, content=content)


def _validation_error_items(exc: ValidationError) -> list[dict[str, str]]:
    """Field location, message, and type. Submitted values are omitted."""
    items: list[dict[str, str]] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ()) if not isinstance(part, dict))
        items.append(
            {
                "loc": loc[:180],
                "msg": str(err.get("msg") or "")[:180],
                "type": str(err.get("type") or "")[:80],
            }
        )
        if len(items) >= 8:
            break
    return items


def require_api_token(request: Request) -> None:
    expected = _configured_token()
    if not expected:
        raise _AuthNotConfigured()
    header = request.headers.get("authorization") or ""
    scheme, _, credential = header.partition(" ")
    if scheme.lower() != "bearer" or not credential.strip():
        raise _Unauthorized()
    if not _tokens_match(credential.strip(), expected):
        raise _Unauthorized()


class _AuthNotConfigured(Exception):
    pass


class _Unauthorized(Exception):
    pass


@app.exception_handler(_AuthNotConfigured)
def _handle_auth_not_configured(_request: Request, _exc: _AuthNotConfigured) -> JSONResponse:
    return _error(
        503,
        "AUTH_NOT_CONFIGURED",
        "Server token is not configured.",
    )


@app.exception_handler(_Unauthorized)
def _handle_unauthorized(_request: Request, _exc: _Unauthorized) -> JSONResponse:
    return _error(401, "UNAUTHORIZED", "Valid bearer token required.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "backlog-agent"}


@app.post("/generate", response_model=None)
def generate(
    payload: dict[str, Any],
    _: None = Depends(require_api_token),
) -> dict[str, Any] | JSONResponse:
    try:
        envelope = CopilotWorkflowResponseV1.model_validate(payload)
    except ValidationError as exc:
        errors = _validation_error_items(exc)
        _LOG.info("generate rejected invalid envelope errors=%s", errors)
        return _error(
            422,
            "INVALID_ENVELOPE",
            "Request must be a COMPLETE copilot.workflow_response.v1 JSON envelope.",
            errors,
        )
    except (TypeError, ValueError):
        _LOG.info("generate rejected envelope that could not be validated")
        return _error(
            422,
            "INVALID_ENVELOPE",
            "Request must be a COMPLETE copilot.workflow_response.v1 JSON envelope.",
        )

    try:
        provider = get_provider()
        backlog, findings = decompose_backlog(envelope, provider)
        return serialize_preview(backlog, findings)
    except DecompositionValidationError as exc:
        errors = [
            {"loc": "", "msg": str(item)[:180], "type": "decomposition"}
            for item in exc.errors[:8]
        ]
        _LOG.info("generate rejected decomposition errors=%s", [item["msg"] for item in errors])
        return _error(
            422,
            "DECOMPOSITION_FAILED",
            "LLM backlog proposal failed independent validation.",
            errors,
        )
    except AIProviderConfigError:
        return _error(
            503,
            "AI_PROVIDER_NOT_CONFIGURED",
            "The configured AI provider is unavailable.",
        )
    except AIProviderQuotaError:
        return _error(
            503,
            "AI_PROVIDER_QUOTA_EXCEEDED",
            "The AI provider quota was exceeded.",
        )
    except AIProviderTimeoutError:
        return _error(
            503,
            "AI_PROVIDER_TIMEOUT",
            "The AI provider timed out.",
        )
    except AIProviderError:
        return _error(
            503,
            "AI_PROVIDER_ERROR",
            "The AI provider failed.",
        )
    except Exception:
        _LOG.exception("Unhandled error in POST /generate")
        return _error(500, "INTERNAL_ERROR", "Backlog generation failed.")


def run() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("backlog_agent.api:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    run()
