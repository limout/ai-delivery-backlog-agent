from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from backlog_agent.api import app
from backlog_agent.llm.exceptions import AIProviderError
from backlog_agent.llm.mock import MockProvider
from tests.llm_proposal_fixtures import representative_complete_proposal

FIXTURE = Path(__file__).parent / "fixtures" / "copilot.workflow_response.v1.complete.json"
_TOKEN = "test-backlog-token"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("BACKLOG_API_TOKEN", _TOKEN)
    monkeypatch.setattr(
        "backlog_agent.api.get_provider",
        lambda: MockProvider(representative_complete_proposal()),
    )
    return TestClient(app)


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {_TOKEN}"}


def _envelope() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_health_does_not_call_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def forbidden() -> None:
        calls.append("get_provider")
        raise AssertionError("health must not invoke the LLM provider")

    monkeypatch.setenv("BACKLOG_API_TOKEN", _TOKEN)
    monkeypatch.setattr("backlog_agent.api.get_provider", forbidden)
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "backlog-agent"}
    assert calls == []


def test_generate_complete_envelope_returns_preview(client: TestClient) -> None:
    response = client.post("/generate", json=_envelope(), headers=_auth())
    assert response.status_code == 200
    body = response.json()
    assert "backlog" in body
    assert "findings" in body
    assert body["backlog"]["items"]
    assert body["findings"]["llm_calls"]
    epics = [item for item in body["backlog"]["items"] if item["type"] == "epic"]
    assert epics[0]["title"] == "Customer self-service policy renewal portal"


def test_generate_rejects_missing_authorization(client: TestClient) -> None:
    response = client.post("/generate", json=_envelope())
    assert response.status_code == 401
    assert response.json()["error"] == "UNAUTHORIZED"


def test_generate_rejects_incorrect_token(client: TestClient) -> None:
    response = client.post(
        "/generate",
        json=_envelope(),
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401
    assert response.json()["error"] == "UNAUTHORIZED"


def test_generate_fails_closed_when_server_token_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BACKLOG_API_TOKEN", raising=False)
    monkeypatch.setattr(
        "backlog_agent.api.get_provider",
        lambda: MockProvider(representative_complete_proposal()),
    )
    response = TestClient(app).post("/generate", json=_envelope(), headers=_auth())
    assert response.status_code == 503
    assert response.json()["error"] == "AUTH_NOT_CONFIGURED"


@pytest.mark.parametrize("status", ["NEEDS_INFO", "BLOCKED"])
def test_generate_rejects_incomplete_status(client: TestClient, status: str) -> None:
    payload = _envelope()
    payload["status"] = status
    payload["workflow_status"] = status
    response = client.post("/generate", json=payload, headers=_auth())
    assert response.status_code == 422
    assert response.json()["error"] == "INVALID_ENVELOPE"
    assert "COMPLETE" in response.json()["message"]


def test_generate_rejects_invalid_envelope(client: TestClient) -> None:
    response = client.post("/generate", json={"user_request": "SENTINEL-not-a-contract"}, headers=_auth())
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "INVALID_ENVELOPE"
    errors = body["validation_errors"]
    assert errors
    assert all(set(item) == {"loc", "msg", "type"} for item in errors)
    assert any(item["loc"] and item["type"] for item in errors)
    assert "SENTINEL-not-a-contract" not in response.text


_PREMIUM_PATH = "$.requirements.functional_requirements[2]"
_PREMIUM_TEXT = "The system shall display current premium and coverage before confirmation."
_PREMIUM_DESCRIPTION = (
    "Before the customer confirms, the portal shows the proposed premium and "
    "coverage so they can abort if the terms are unacceptable."
)


def _coverage_log(caplog: pytest.LogCaptureFixture) -> dict:
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "backlog_agent.api"
        and record.getMessage().startswith("generate rejected decomposition coverage=")
    ]
    assert messages == [messages[0]]
    return ast.literal_eval(messages[0].split("coverage=", 1)[1])


def _assert_coverage_log_omits_sensitive_content(message_text: str) -> None:
    forbidden = (
        _PREMIUM_TEXT,
        _PREMIUM_DESCRIPTION,
        "Customer self-service policy renewal portal",
        "Build a customer self-service portal for insurance policy renewals.",
        _TOKEN,
        "Authorization",
        "Bearer",
        "BACKLOG_API_TOKEN",
        "normalized project facts",
    )
    for secret in forbidden:
        assert secret not in message_text


def _proposal_without_items(local_ids: set[str]) -> dict:
    proposal = copy.deepcopy(representative_complete_proposal())
    proposal["items"] = [item for item in proposal["items"] if item["local_id"] not in local_ids]
    return proposal


def _premium_cited_only_by_feature() -> dict:
    proposal = _proposal_without_items({"story-premium"})
    feature = next(item for item in proposal["items"] if item["local_id"] == "feat-renewal")
    feature["provenance"].append(
        {
            "json_path": _PREMIUM_PATH,
            "field_name": "functional_requirements",
            "excerpt": _PREMIUM_TEXT,
        }
    )
    return proposal


def _post_rejected_proposal(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    proposals: list[dict],
) -> tuple[object, dict, str]:
    monkeypatch.setenv("BACKLOG_API_TOKEN", _TOKEN)
    monkeypatch.setattr(
        "backlog_agent.api.get_provider",
        lambda: MockProvider(responses=proposals),
    )
    with caplog.at_level("INFO", logger="backlog_agent.api"):
        response = TestClient(app).post("/generate", json=_envelope(), headers=_auth())
    text = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "backlog_agent.api"
    )
    return response, _coverage_log(caplog), text


def test_decomposition_failure_log_marks_missing_story_citation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    missing = _proposal_without_items({"story-premium"})
    response, diagnostic, logged = _post_rejected_proposal(monkeypatch, caplog, [missing, missing])

    assert response.status_code == 422
    assert response.json()["error"] == "DECOMPOSITION_FAILED"
    assert diagnostic == {
        "uncovered_fr_paths": [_PREMIUM_PATH],
        "non_story_citations": [{"path": _PREMIUM_PATH, "item_types": []}],
        "lost_fr_paths": [],
    }
    _assert_coverage_log_omits_sensitive_content(logged)


def test_decomposition_failure_log_marks_non_story_citation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cited_on_feature = _premium_cited_only_by_feature()
    response, diagnostic, logged = _post_rejected_proposal(
        monkeypatch,
        caplog,
        [cited_on_feature, cited_on_feature],
    )

    assert response.status_code == 422
    assert response.json()["error"] == "DECOMPOSITION_FAILED"
    assert diagnostic["uncovered_fr_paths"] == [_PREMIUM_PATH]
    assert diagnostic["non_story_citations"] == [{"path": _PREMIUM_PATH, "item_types": ["feature"]}]
    assert diagnostic["lost_fr_paths"] == []
    _assert_coverage_log_omits_sensitive_content(logged)


def test_decomposition_failure_log_reports_paths_lost_by_repair(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    covered_then_invalid = _proposal_without_items({"story-confirm", "task-email"})
    premium = next(item for item in covered_then_invalid["items"] if item["local_id"] == "story-premium")
    premium["title"] = "Unrelated quarterly billing export"
    premium["title_origin"] = "source"
    repaired_without_story = _premium_cited_only_by_feature()
    response, diagnostic, logged = _post_rejected_proposal(
        monkeypatch,
        caplog,
        [covered_then_invalid, repaired_without_story],
    )

    assert response.status_code == 422
    assert response.json()["error"] == "DECOMPOSITION_FAILED"
    assert diagnostic["uncovered_fr_paths"] == [_PREMIUM_PATH]
    assert diagnostic["non_story_citations"] == [{"path": _PREMIUM_PATH, "item_types": ["feature"]}]
    assert diagnostic["lost_fr_paths"] == [_PREMIUM_PATH]
    _assert_coverage_log_omits_sensitive_content(logged)


def test_generate_provider_failure_is_controlled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BACKLOG_API_TOKEN", _TOKEN)
    monkeypatch.setattr(
        "backlog_agent.api.get_provider",
        lambda: MockProvider(error=AIProviderError("GEMINI_API_KEY=leaked-secret")),
    )
    response = TestClient(app).post("/generate", json=_envelope(), headers=_auth())
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "AI_PROVIDER_ERROR"
    assert "leaked-secret" not in response.text
    assert "GEMINI_API_KEY" not in response.text
    assert "traceback" not in response.text.lower()
