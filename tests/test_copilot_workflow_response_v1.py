from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from backlog_agent.contracts.copilot_workflow_response_v1 import (
    COPILOT_WORKFLOW_RESPONSE_V1,
    CopilotWorkflowResponseV1,
)
from tests.conftest import minimal_complete_envelope


def parse(payload: Any) -> CopilotWorkflowResponseV1:
    return CopilotWorkflowResponseV1.model_validate(payload)


def test_accepts_minimal_complete_envelope(complete_envelope: dict[str, Any]) -> None:
    result = parse(complete_envelope)

    assert result.contract_id == COPILOT_WORKFLOW_RESPONSE_V1
    assert result.envelope.workflow_status == "COMPLETE"
    assert result.envelope.status == "COMPLETE"
    assert result.envelope.user_request == complete_envelope["user_request"]
    assert result.envelope.requirements.functional_requirements == [
        "The system shall allow a customer to renew an existing policy."
    ]
    assert result.original_envelope == complete_envelope


def test_preserves_optional_fields_without_inferring_missing_data(
    complete_envelope: dict[str, Any],
) -> None:
    complete_envelope["executive_summary"] = "Renewals should be self-service."
    complete_envelope["requirements"]["assumptions"] = ["Customers have an account."]
    original = copy.deepcopy(complete_envelope)

    result = parse(complete_envelope)

    assert result.original_envelope == original
    assert result.original_envelope["executive_summary"] == "Renewals should be self-service."
    assert "non_functional_requirements" not in result.original_envelope["requirements"]
    assert not hasattr(result.envelope.requirements, "non_functional_requirements") or (
        "non_functional_requirements" not in result.envelope.requirements.model_dump()
    )


def test_does_not_mutate_caller_payload(complete_envelope: dict[str, Any]) -> None:
    original = copy.deepcopy(complete_envelope)

    parse(complete_envelope)

    assert complete_envelope == original


@pytest.mark.parametrize("status", ["NEEDS_INFO", "BLOCKED", "RUNNING", "INCOMPLETE"])
def test_rejects_non_complete_aligned_statuses(
    complete_envelope: dict[str, Any], status: str
) -> None:
    complete_envelope["workflow_status"] = status
    complete_envelope["status"] = status

    with pytest.raises(ValidationError, match="not COMPLETE"):
        parse(complete_envelope)


def test_rejects_mismatched_status(complete_envelope: dict[str, Any]) -> None:
    complete_envelope["workflow_status"] = "COMPLETE"
    complete_envelope["status"] = "RUNNING"

    with pytest.raises(ValidationError, match="Mismatched status"):
        parse(complete_envelope)


def test_rejects_empty_user_request(complete_envelope: dict[str, Any]) -> None:
    complete_envelope["user_request"] = "   "

    with pytest.raises(ValidationError, match="user_request must be a non-empty string"):
        parse(complete_envelope)


def test_rejects_empty_functional_requirements(complete_envelope: dict[str, Any]) -> None:
    complete_envelope["requirements"]["functional_requirements"] = []

    with pytest.raises(ValidationError):
        parse(complete_envelope)


def test_rejects_blank_functional_requirement(complete_envelope: dict[str, Any]) -> None:
    complete_envelope["requirements"]["functional_requirements"] = ["   "]

    with pytest.raises(ValidationError, match="non-empty list of non-empty strings"):
        parse(complete_envelope)


def test_rejects_missing_acceptance_criteria(complete_envelope: dict[str, Any]) -> None:
    del complete_envelope["requirements"]["acceptance_criteria"]

    with pytest.raises(ValidationError):
        parse(complete_envelope)


def test_accepts_empty_acceptance_criteria(complete_envelope: dict[str, Any]) -> None:
    complete_envelope["requirements"]["acceptance_criteria"] = []

    result = parse(complete_envelope)

    assert result.envelope.requirements.acceptance_criteria == []


@pytest.mark.parametrize(
    "section,field",
    [
        ("delivery_plan", "delivery_phases"),
        ("delivery_plan", "workstreams"),
        ("delivery_plan", "milestones"),
        ("sow", "in_scope"),
        ("sow", "out_of_scope"),
        ("sow", "deliverables"),
        ("solution", "key_capabilities"),
    ],
)
def test_rejects_missing_required_lists(
    complete_envelope: dict[str, Any], section: str, field: str
) -> None:
    del complete_envelope[section][field]

    with pytest.raises(ValidationError):
        parse(complete_envelope)


def test_accepts_empty_optional_required_lists(complete_envelope: dict[str, Any]) -> None:
    complete_envelope["solution"]["key_capabilities"] = []
    complete_envelope["delivery_plan"]["milestones"] = []
    complete_envelope["sow"]["out_of_scope"] = []

    result = parse(complete_envelope)

    assert result.envelope.solution.key_capabilities == []
    assert result.envelope.delivery_plan.milestones == []
    assert result.envelope.sow.out_of_scope == []


def test_rejects_plain_text_export() -> None:
    plain_text = """
    # Delivery Plan
    Status: COMPLETE
    The system shall allow renewals.
    """.strip()

    with pytest.raises(ValidationError, match="plain-text"):
        parse(plain_text)


def test_accepts_json_string_envelope() -> None:
    import json

    payload = minimal_complete_envelope()

    result = parse(json.dumps(payload))

    assert result.envelope.status == "COMPLETE"
    assert result.original_envelope == payload


def test_rejects_jira_payload() -> None:
    jira = {
        "expand": "schema,names",
        "issues": [{"key": "PORTAL-1", "fields": {"summary": "Renew policy"}}],
        "maxResults": 50,
        "total": 1,
    }

    with pytest.raises(ValidationError, match="Jira tracker payload"):
        parse(jira)


def test_rejects_azure_devops_payload() -> None:
    ado = {
        "id": 34821,
        "rev": 4,
        "fields": {
            "System.Title": "Renew policy",
            "System.WorkItemType": "User Story",
            "System.State": "New",
        },
    }

    with pytest.raises(ValidationError, match="Azure DevOps tracker payload"):
        parse(ado)


def test_does_not_rewrite_source_strings(complete_envelope: dict[str, Any]) -> None:
    messy = "  Keep leading space in Copilot text"
    complete_envelope["requirements"]["functional_requirements"] = [messy]
    complete_envelope["user_request"] = "Keep this exact request."

    result = parse(complete_envelope)

    assert result.envelope.requirements.functional_requirements == [messy]
    assert result.original_envelope["requirements"]["functional_requirements"] == [messy]
