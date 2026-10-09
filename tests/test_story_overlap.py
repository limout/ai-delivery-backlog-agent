from __future__ import annotations

from types import SimpleNamespace

from backlog_agent.backlog.granularity import (
    description_leaks_owned_action,
    record_granularity_findings,
)
from backlog_agent.backlog.models import SourceReference, WorkItemType
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.contracts.versions import COPILOT_WORKFLOW_RESPONSE_V1
from backlog_agent.findings.models import FindingCode
from backlog_agent.llm.mock import MockProvider
from backlog_agent.backlog.decompose import decompose_backlog
from tests.copilot_envelopes import library_holds_envelope
from tests.test_decompose_generic import _library_proposal


def _ref(path: str, excerpt: str = "excerpt") -> SourceReference:
    return SourceReference(
        contract_id=COPILOT_WORKFLOW_RESPONSE_V1,
        json_path=path,
        field_name=None,
        excerpt=excerpt,
    )


def _story(
    canonical_id: str,
    title: str,
    *,
    description: str = "",
    acceptance_criteria: list[str] | None = None,
    provenance: list[SourceReference] | None = None,
    dependencies: list[str] | None = None,
    local_id: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        type=WorkItemType.USER_STORY,
        canonical_id=canonical_id,
        local_id=local_id,
        title=title,
        description=description,
        acceptance_criteria=acceptance_criteria or [],
        test_requirements=[],
        provenance=tuple(provenance or ()),
        child_ids=(),
        dependencies=dependencies or [],
        parent_id=None,
    )


class _Sink:
    def __init__(self) -> None:
        self.items: list = []

    def add(self, finding: object) -> None:
        self.items.append(finding)


def _codes(sink: _Sink) -> set[FindingCode]:
    return {item.code for item in sink.items}


def test_submit_story_must_not_own_separate_email_story() -> None:
    submit = _story(
        "cbl_submit",
        "Submit Policy Renewal Request",
        local_id="story-submit",
        description=(
            "As an eligible customer, I want to submit a policy renewal request and trigger "
            "a confirmation email so that my policy is renewed."
        ),
        acceptance_criteria=[
            "An eligible customer can submit a policy renewal request through the web portal "
            "and trigger a confirmation email via the email delivery service."
        ],
        provenance=[_ref("$.requirements.functional_requirements[1]")],
        dependencies=["cbl_review"],
    )
    email = _story(
        "cbl_email",
        "Send Confirmation Email via Delivery Service",
        local_id="story-email",
        description=(
            "As a system, I want to utilize the existing email delivery service to send "
            "confirmation emails so that customers receive written verification."
        ),
        provenance=[_ref("$.requirements.functional_requirements[4]")],
        dependencies=["cbl_submit"],
    )
    assert description_leaks_owned_action(email, submit) is True
    assert description_leaks_owned_action(submit, email) is False

    sink = _Sink()
    record_granularity_findings(SimpleNamespace(), SimpleNamespace(items=[submit, email]), sink)
    overlap = [item for item in sink.items if item.code is FindingCode.OVERLAPPING_STORY_RESPONSIBILITY]
    assert overlap
    assert set(overlap[0].canonical_ids) == {"cbl_submit", "cbl_email"}
    assert set(overlap[0].local_ids) == {"story-submit", "story-email"}
    message = overlap[0].message
    assert "canonical_id=cbl_submit" in message and "canonical_id=cbl_email" in message
    assert "local_id=story-submit" in message and "local_id=story-email" in message
    assert "Submit Policy Renewal Request" in message
    assert "Send Confirmation Email via Delivery Service" in message
    assert "confirmation email" in message
    assert "trigger a confirmation email" in message
    assert "Remediation:" in message
    assert "remove" in message.lower()


def test_login_review_submit_email_sequence_is_not_overlap() -> None:
    login = _story(
        "cbl_login",
        "Customer Login via Identity Provider",
        description=(
            "As an eligible customer, I want to securely log into the web portal using the "
            "existing identity provider so that I can access my account."
        ),
        provenance=[_ref("$.requirements.functional_requirements[3]")],
    )
    review = _story(
        "cbl_review",
        "Review Existing Policies Online",
        description=(
            "As an eligible customer, I want to review my existing personal motor and home "
            "insurance policy details online so that I can inspect current coverage."
        ),
        provenance=[_ref("$.requirements.functional_requirements[0]")],
        dependencies=["cbl_login"],
    )
    submit = _story(
        "cbl_submit",
        "Submit Policy Renewal Request",
        description=(
            "As an eligible customer, I want to submit a policy renewal request through the "
            "web portal so that the policy is renewed."
        ),
        provenance=[_ref("$.requirements.functional_requirements[1]")],
        dependencies=["cbl_review"],
    )
    email = _story(
        "cbl_email",
        "Send Confirmation Email via Delivery Service",
        description=(
            "As an eligible customer, I want to receive a confirmation email after renewal "
            "so that I have written verification of completion."
        ),
        provenance=[_ref("$.requirements.functional_requirements[4]")],
        dependencies=["cbl_submit"],
    )
    sink = _Sink()
    record_granularity_findings(
        SimpleNamespace(),
        SimpleNamespace(items=[login, review, submit, email]),
        sink,
    )
    assert FindingCode.OVERLAPPING_STORY_RESPONSIBILITY not in _codes(sink)


def test_duplicate_acceptance_criterion_text_and_path() -> None:
    shared = "An eligible customer can view policy details within the web portal."
    left = _story(
        "cbl_a",
        "Review Existing Policies Online",
        description="A customer inspects coverage and premium for their own policies.",
        acceptance_criteria=[shared],
        provenance=[
            _ref("$.requirements.functional_requirements[0]"),
            _ref("$.requirements.acceptance_criteria[1]", shared),
        ],
    )
    right = _story(
        "cbl_b",
        "Inspect Coverage After Login",
        description="A customer opens coverage after authenticating.",
        acceptance_criteria=[shared],
        provenance=[
            _ref("$.requirements.functional_requirements[3]"),
            _ref("$.requirements.acceptance_criteria[1]", shared),
        ],
    )
    sink = _Sink()
    record_granularity_findings(SimpleNamespace(), SimpleNamespace(items=[left, right]), sink)
    duplicates = [
        item for item in sink.items if item.code is FindingCode.DUPLICATE_ACCEPTANCE_CRITERION
    ]
    assert duplicates
    assert set(duplicates[0].canonical_ids) == {"cbl_a", "cbl_b"}
    duplicate_message = duplicates[0].message
    assert "cbl_a" in duplicate_message and "cbl_b" in duplicate_message
    assert "Review Existing Policies Online" in duplicate_message
    assert "Inspect Coverage After Login" in duplicate_message
    assert "acceptance_criteria[1]" in duplicate_message or "view policy details" in duplicate_message
    assert "Remediation:" in duplicate_message
    assert FindingCode.OVERLAPPING_STORY_RESPONSIBILITY not in _codes(sink)


def test_compound_fr_distinct_clauses_are_not_overlap() -> None:
    path = "$.requirements.functional_requirements[2]"
    retrieve = _story(
        "cbl_retrieve",
        "Retrieve Policy Data via Integration",
        description=(
            "The portal retrieves policy information from the policy administration system "
            "so listed records stay accurate."
        ),
        provenance=[_ref(path)],
    )
    submit = _story(
        "cbl_pas_submit",
        "Submit Renewal Requests to Administration",
        description=(
            "The portal submits renewal requests to the policy administration system after "
            "the customer confirms."
        ),
        provenance=[_ref(path)],
    )
    sink = _Sink()
    record_granularity_findings(
        SimpleNamespace(), SimpleNamespace(items=[retrieve, submit]), sink
    )
    assert FindingCode.OVERLAPPING_STORY_RESPONSIBILITY not in _codes(sink)


def test_library_fixture_does_not_flag_sequenced_unrelated_domain() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(library_holds_envelope())
    _backlog, findings = decompose_backlog(envelope, MockProvider(_library_proposal()), repair=False)
    assert not any(item.code is FindingCode.OVERLAPPING_STORY_RESPONSIBILITY for item in findings.items)
    assert not any(item.code is FindingCode.DUPLICATE_ACCEPTANCE_CRITERION for item in findings.items)
