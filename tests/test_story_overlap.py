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


def test_related_stories_sharing_a_later_dependency_are_not_overlap() -> None:
    login = _story(
        "cbl_login",
        "Customer Login via Identity Provider",
        description=(
            "As an eligible customer, I want to log in using the existing identity provider "
            "so that I can later review existing policies."
        ),
        provenance=[_ref("$.requirements.functional_requirements[3]")],
    )
    review = _story(
        "cbl_review",
        "Review Existing Policies Online",
        description=(
            "As an eligible customer, I want to review existing policies online so that I can "
            "inspect current coverage."
        ),
        provenance=[_ref("$.requirements.functional_requirements[0]")],
        dependencies=["cbl_login"],
    )
    sink = _Sink()
    record_granularity_findings(
        SimpleNamespace(), SimpleNamespace(items=[login, review]), sink
    )
    assert FindingCode.OVERLAPPING_STORY_RESPONSIBILITY not in _codes(sink)


def test_auth_reusing_policy_view_acceptance_criterion_is_duplicate() -> None:
    shared = "An eligible customer can view policy details within the web portal."
    auth = _story(
        "cbl_auth",
        "Customer Authentication",
        description="A customer authenticates with the existing identity provider.",
        acceptance_criteria=[shared],
        provenance=[
            _ref("$.requirements.functional_requirements[3]"),
            _ref("$.requirements.acceptance_criteria[0]", shared),
        ],
    )
    review = _story(
        "cbl_review",
        "Review Existing Policies Online",
        description="A customer inspects coverage and premium for their own policies.",
        acceptance_criteria=[shared],
        provenance=[
            _ref("$.requirements.functional_requirements[0]"),
            _ref("$.requirements.acceptance_criteria[0]", shared),
        ],
    )
    sink = _Sink()
    record_granularity_findings(SimpleNamespace(), SimpleNamespace(items=[auth, review]), sink)
    duplicates = [
        item for item in sink.items if item.code is FindingCode.DUPLICATE_ACCEPTANCE_CRITERION
    ]
    assert duplicates
    assert set(duplicates[0].canonical_ids) == {"cbl_auth", "cbl_review"}


def test_feature_child_misalignment_when_no_shared_capability_terms() -> None:
    email = _story(
        "cbl_email",
        "Send Pickup Notice via Notice Service",
        description="The notice service sends a pickup notice when a held copy is ready.",
        provenance=[_ref("$.requirements.functional_requirements[2]")],
    )
    feature = SimpleNamespace(
        type=WorkItemType.FEATURE,
        canonical_id="cbl_feat_search",
        title="Catalog search",
        description="",
        child_ids=(email.canonical_id,),
        provenance=(_ref("$.solution.key_capabilities[0]", "Catalog search"),),
        acceptance_criteria=[],
        test_requirements=[],
        parent_id=None,
        dependencies=[],
    )
    sink = _Sink()
    record_granularity_findings(
        SimpleNamespace(), SimpleNamespace(items=[feature, email]), sink
    )
    misaligned = [
        item for item in sink.items if item.code is FindingCode.FEATURE_CHILD_MISALIGNMENT
    ]
    assert misaligned
    assert "Catalog search" in misaligned[0].message


def test_aligned_feature_with_matching_child_is_not_flagged() -> None:
    search = _story(
        "cbl_search",
        "Search the Catalog by Title",
        description="A member searches the catalog and matching holdings are listed.",
        provenance=[_ref("$.requirements.functional_requirements[0]")],
    )
    feature = SimpleNamespace(
        type=WorkItemType.FEATURE,
        canonical_id="cbl_feat_search",
        title="Catalog search",
        description="",
        child_ids=(search.canonical_id,),
        provenance=(_ref("$.solution.key_capabilities[0]", "Catalog search"),),
        acceptance_criteria=[],
        test_requirements=[],
        parent_id=None,
        dependencies=[],
    )
    sink = _Sink()
    record_granularity_findings(
        SimpleNamespace(), SimpleNamespace(items=[feature, search]), sink
    )
    assert FindingCode.FEATURE_CHILD_MISALIGNMENT not in _codes(sink)


def test_compound_listing_and_terms_criterion_is_duplicate_until_scoped() -> None:
    shared = "A member can view holdings and renewal terms in the catalog."
    listing = _story(
        "cbl_list",
        "View Catalog Holdings",
        description="A member views the holdings available on their account.",
        acceptance_criteria=[shared],
        provenance=[
            _ref("$.requirements.functional_requirements[0]"),
            _ref("$.requirements.acceptance_criteria[0]", shared),
        ],
    )
    terms = _story(
        "cbl_terms",
        "Review Renewal Terms",
        description="A member reviews the renewal terms before confirming.",
        acceptance_criteria=[shared],
        provenance=[
            _ref("$.requirements.functional_requirements[1]"),
            _ref("$.requirements.acceptance_criteria[0]", shared),
        ],
    )
    sink = _Sink()
    record_granularity_findings(SimpleNamespace(), SimpleNamespace(items=[listing, terms]), sink)
    duplicates = [
        item for item in sink.items if item.code is FindingCode.DUPLICATE_ACCEPTANCE_CRITERION
    ]
    assert duplicates
    assert "generated" in duplicates[0].message.lower()
    assert FindingCode.OVERLAPPING_STORY_RESPONSIBILITY in _codes(sink)

    listing.acceptance_criteria = ["A member can view the holdings on their account."]
    listing.provenance = [_ref("$.requirements.functional_requirements[0]")]
    terms.acceptance_criteria = ["A member can review the renewal terms before confirming."]
    terms.provenance = [_ref("$.requirements.functional_requirements[1]")]
    scoped = _Sink()
    record_granularity_findings(SimpleNamespace(), SimpleNamespace(items=[listing, terms]), scoped)
    assert FindingCode.DUPLICATE_ACCEPTANCE_CRITERION not in _codes(scoped)
    assert FindingCode.OVERLAPPING_STORY_RESPONSIBILITY not in _codes(scoped)


def test_downstream_system_mention_is_not_integration_ownership() -> None:
    integration = _story(
        "cbl_pas",
        "Administration System Integration",
        description="The portal integrates with the administration system as the source of record.",
        provenance=[_ref("$.requirements.functional_requirements[3]")],
    )
    listing = _story(
        "cbl_list",
        "View Account Holdings",
        description="A member views holdings sourced from the administration system.",
        acceptance_criteria=["Holdings are retrieved from the administration system."],
        provenance=[_ref("$.requirements.functional_requirements[0]")],
    )
    submit = _story(
        "cbl_submit",
        "Submit Renewal Request",
        description="A member submits a renewal request through the portal.",
        acceptance_criteria=[
            "Submitting a request triggers a submission to the administration system."
        ],
        provenance=[_ref("$.requirements.functional_requirements[2]")],
    )
    sink = _Sink()
    record_granularity_findings(
        SimpleNamespace(),
        SimpleNamespace(items=[integration, listing, submit]),
        sink,
    )
    assert FindingCode.OVERLAPPING_STORY_RESPONSIBILITY not in _codes(sink)


def test_broad_feature_is_not_misaligned_when_most_children_match() -> None:
    search = _story(
        "cbl_search",
        "Search the Catalog",
        description="A member searches the catalog and sees matching holdings.",
    )
    hold = _story(
        "cbl_hold",
        "Place a Hold",
        description="A member places a hold on an available copy.",
    )
    notice = _story(
        "cbl_notice",
        "Send Pickup Notice",
        description="The notice service sends a pickup notice.",
    )
    feature = SimpleNamespace(
        type=WorkItemType.FEATURE,
        canonical_id="cbl_feat",
        title="Catalog search and hold placement",
        description="",
        child_ids=(search.canonical_id, hold.canonical_id, notice.canonical_id),
        provenance=(
            _ref(
                "$.solution.key_capabilities[0]",
                "Catalog search and hold placement",
            ),
        ),
        acceptance_criteria=[],
        test_requirements=[],
        parent_id=None,
        dependencies=[],
    )
    sink = _Sink()
    record_granularity_findings(
        SimpleNamespace(),
        SimpleNamespace(items=[feature, search, hold, notice]),
        sink,
    )
    assert FindingCode.FEATURE_CHILD_MISALIGNMENT not in _codes(sink)


def test_feature_provenance_that_covers_only_a_minority_warns() -> None:
    search = _story(
        "cbl_search",
        "Search the Catalog",
        description="A member searches the catalog.",
    )
    auth = _story(
        "cbl_auth",
        "Member Authentication",
        description="A member authenticates with the identity provider.",
    )
    mail = _story(
        "cbl_mail",
        "Send Confirmation Email",
        description="The delivery service sends a confirmation email.",
    )
    audit = _story(
        "cbl_audit",
        "Record an Audit Event",
        description="An operator can inspect an audit event after submission.",
    )
    feature = SimpleNamespace(
        type=WorkItemType.FEATURE,
        canonical_id="cbl_feat_search",
        title="Catalog search",
        description="",
        child_ids=(
            search.canonical_id,
            auth.canonical_id,
            mail.canonical_id,
            audit.canonical_id,
        ),
        provenance=(_ref("$.solution.key_capabilities[0]", "Catalog search"),),
        acceptance_criteria=[],
        test_requirements=[],
        parent_id=None,
        dependencies=[],
    )
    sink = _Sink()
    record_granularity_findings(
        SimpleNamespace(),
        SimpleNamespace(items=[feature, search, auth, mail, audit]),
        sink,
    )
    misaligned = [
        item for item in sink.items if item.code is FindingCode.FEATURE_CHILD_MISALIGNMENT
    ]
    assert misaligned
    assert "heuristic" in misaligned[0].message.lower()
    assert (
        "1 of 4" in misaligned[0].message
        or "distinct capability clusters" in misaligned[0].message
    )


def test_cohesive_feature_with_several_related_stories_is_not_flagged() -> None:
    children = [
        _story(
            "cbl_title",
            "Search Holdings by Title",
            description="A member searches the catalog by title and matching holdings are listed.",
        ),
        _story(
            "cbl_author",
            "Search Holdings by Author",
            description="A member searches the catalog by author and matching holdings are listed.",
        ),
        _story(
            "cbl_filter",
            "Filter Catalog Search Results",
            description="A member filters catalog search results by format so matching holdings remain listed.",
        ),
        _story(
            "cbl_empty",
            "Show Empty Catalog Search Results",
            description="A member sees an empty catalog search result when no holdings match.",
        ),
    ]
    feature = SimpleNamespace(
        type=WorkItemType.FEATURE,
        canonical_id="cbl_feat_search",
        title="Catalog search",
        description="Search, filter, and present catalog holdings, including empty result handling.",
        child_ids=tuple(child.canonical_id for child in children),
        provenance=(_ref("$.solution.key_capabilities[0]", "Catalog search"),),
        acceptance_criteria=[],
        test_requirements=[],
        parent_id=None,
        dependencies=[],
    )
    sink = _Sink()
    record_granularity_findings(
        SimpleNamespace(),
        SimpleNamespace(items=[feature, *children]),
        sink,
    )
    assert FindingCode.FEATURE_CHILD_MISALIGNMENT not in _codes(sink)


def test_broad_feature_spanning_auth_review_submit_integration_and_confirmation() -> None:
    children = [
        _story(
            "cbl_auth",
            "Customer Authentication",
            description="A customer authenticates before opening the account.",
        ),
        _story(
            "cbl_review",
            "Review Personal Policies",
            description="A customer reviews personal policies and current coverage.",
        ),
        _story(
            "cbl_submit",
            "Submit Renewal Request",
            description="A customer submits a renewal request after accepting the terms.",
        ),
        _story(
            "cbl_pas",
            "Administration System Integration",
            description="The portal integrates the submitted renewal with the administration system.",
        ),
        _story(
            "cbl_mail",
            "Confirmation Email Delivery",
            description="The customer receives confirmation after the renewal is accepted.",
        ),
    ]
    feature = SimpleNamespace(
        type=WorkItemType.FEATURE,
        canonical_id="cbl_feat_portal",
        title="Customer renewal",
        description=(
            "Authenticate, review policies, submit a renewal, integrate with the "
            "administration system, and send confirmation."
        ),
        child_ids=tuple(child.canonical_id for child in children),
        provenance=(
            _ref(
                "$.solution.key_capabilities[0]",
                "Online review of personal policies and renewal submission",
            ),
        ),
        acceptance_criteria=[],
        test_requirements=[],
        parent_id=None,
        dependencies=[],
    )
    sink = _Sink()
    record_granularity_findings(
        SimpleNamespace(),
        SimpleNamespace(items=[feature, *children]),
        sink,
    )
    misaligned = [
        item for item in sink.items if item.code is FindingCode.FEATURE_CHILD_MISALIGNMENT
    ]
    assert misaligned
    assert "distinct capability clusters" in misaligned[0].message
    assert "listing those areas" in misaligned[0].message.lower()


def test_coherent_feature_with_narrow_provenance_is_not_split() -> None:
    review = _story(
        "cbl_review",
        "Review Personal Policies",
        description="A customer reviews personal policies and inspects current coverage.",
    )
    submit = _story(
        "cbl_submit",
        "Submit Renewal Request",
        description="A customer submits a renewal after reviewing the terms.",
    )
    notice = _story(
        "cbl_notice",
        "Send Renewal Confirmation",
        description="A customer receives confirmation after the renewal is accepted.",
    )
    feature = SimpleNamespace(
        type=WorkItemType.FEATURE,
        canonical_id="cbl_feat_renewal",
        title="Policy review and renewal submission",
        description="Customers review policies, submit a renewal, and receive renewal confirmation.",
        child_ids=(review.canonical_id, submit.canonical_id, notice.canonical_id),
        provenance=(_ref("$.solution.key_capabilities[0]", "Online review of policies"),),
        acceptance_criteria=[],
        test_requirements=[],
        parent_id=None,
        dependencies=[],
    )
    sink = _Sink()
    record_granularity_findings(
        SimpleNamespace(),
        SimpleNamespace(items=[feature, review, submit, notice]),
        sink,
    )
    assert FindingCode.FEATURE_CHILD_MISALIGNMENT not in _codes(sink)


def test_library_fixture_does_not_flag_sequenced_unrelated_domain() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(library_holds_envelope())
    _backlog, findings = decompose_backlog(envelope, MockProvider(_library_proposal()), repair=False)
    assert not any(item.code is FindingCode.OVERLAPPING_STORY_RESPONSIBILITY for item in findings.items)
    assert not any(item.code is FindingCode.DUPLICATE_ACCEPTANCE_CRITERION for item in findings.items)
