"""A structurally valid COMPLETE envelope still returns a draft when quality is incomplete."""

from __future__ import annotations

import pytest

from backlog_agent.backlog.decompose import decompose_backlog
from backlog_agent.backlog.models import WorkItemType
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import FindingCode
from backlog_agent.llm.exceptions import DecompositionValidationError
from backlog_agent.llm.mock import MockProvider
from tests.copilot_envelopes import library_holds_envelope
from tests.test_decompose_generic import _library_proposal


def _library() -> CopilotWorkflowResponseV1:
    return CopilotWorkflowResponseV1.model_validate(library_holds_envelope())


def _notify_text() -> str:
    return library_holds_envelope()["requirements"]["functional_requirements"][2]


def _story_cites(backlog, path: str) -> bool:
    return any(
        item.type is WorkItemType.USER_STORY
        and any(reference.json_path == path and reference.excerpt for reference in item.provenance)
        for item in backlog.items
    )


def test_uncovered_requirement_still_returns_a_draft_with_findings():
    backlog, findings = decompose_backlog(
        _library(),
        MockProvider(_library_proposal(cover_notify=False)),
        repair=False,
    )

    assert any(item.type is WorkItemType.EPIC for item in backlog.items)
    assert any(item.type is WorkItemType.USER_STORY for item in backlog.items)
    uncovered = [
        item
        for item in findings.items
        if item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT
    ]
    assert len(uncovered) == 1
    assert uncovered[0].source_references[0].excerpt == _notify_text()
    assert not _story_cites(backlog, "$.requirements.functional_requirements[2]")
    gap = [
        item
        for item in backlog.items
        if item.type is WorkItemType.TASK and "uncovered requirement" in item.title.lower()
    ]
    assert gap
    assert gap[0].provenance[0].json_path == "$.requirements.functional_requirements[2]"
    assert findings.llm_calls[0].validation_passed is True
    assert findings.llm_calls[0].accepted is False


def test_failed_quality_repair_keeps_the_valid_draft():
    provider = MockProvider(
        responses=[
            _library_proposal(cover_notify=False),
            {"items": [{"local_id": "nope", "type": "epic"}]},
        ]
    )

    backlog, findings = decompose_backlog(_library(), provider)

    assert len(provider.calls) == 2
    assert "quality" in provider.calls[1][0].lower()
    assert any(item.type is WorkItemType.USER_STORY for item in backlog.items)
    assert not _story_cites(backlog, "$.requirements.functional_requirements[2]")
    assert any(item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT for item in findings.items)
    assert any(item.code is FindingCode.QUALITY_REPAIR_REJECTED for item in findings.items)
    assert findings.llm_calls[0].accepted is False


def test_blocked_workflow_and_invalid_input_are_still_rejected():
    blocked = library_holds_envelope()
    blocked["status"] = "BLOCKED"
    blocked["workflow_status"] = "BLOCKED"
    with pytest.raises(ValueError, match="BLOCKED"):
        CopilotWorkflowResponseV1.model_validate(blocked)

    with pytest.raises(Exception):
        decompose_backlog({"status": "COMPLETE"}, MockProvider(_library_proposal()), repair=False)


def test_structurally_invalid_proposal_still_fails():
    proposal = _library_proposal()
    proposal["items"] = [item for item in proposal["items"] if item["type"] != "epic"]
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(_library(), MockProvider(proposal), repair=False)
    assert any("epic" in error for error in exc.value.errors)
    assert exc.value.findings.llm_calls[0].validation_passed is False


def test_omission_is_not_reported_as_covered():
    backlog, findings = decompose_backlog(
        _library(),
        MockProvider(_library_proposal(cover_notify=False)),
        repair=False,
    )
    covered = {
        reference.json_path
        for item in backlog.items
        if item.type is WorkItemType.USER_STORY
        for reference in item.provenance
        if reference.json_path.startswith("$.requirements.functional_requirements")
    }
    assert "$.requirements.functional_requirements[2]" not in covered
    assert any(
        item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT
        and item.source_references[0].json_path == "$.requirements.functional_requirements[2]"
        for item in findings.items
    )
