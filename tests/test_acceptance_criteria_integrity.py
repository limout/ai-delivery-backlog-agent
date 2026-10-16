"""Acceptance criteria stay complete when shared compounds are scoped.

The billing, clinic, and warehouse fixtures are unrelated. Production scoping
must not depend on their wording.
"""

from __future__ import annotations

from backlog_agent.backlog.decompose import (
    _scope_shared_story_outcomes,
    decompose_backlog,
)
from backlog_agent.backlog.models import WorkItemType
from backlog_agent.backlog.proposal import (
    BacklogProposal,
    ContentOrigin,
    ProposedItem,
    ProposedProvenance,
)
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import FindingCode
from backlog_agent.llm.mock import MockProvider
from tests.copilot_envelopes import _complete
from tests.test_decompose_generic import _item, _src
from tests.test_story_scope_repair import _invalid, _story, _tree


def _proposal(*stories: ProposedItem) -> BacklogProposal:
    return BacklogProposal(items=list(stories))


def _story_item(
    local_id: str,
    title: str,
    description: str,
    criterion: str,
    excerpt: str,
) -> ProposedItem:
    return ProposedItem(
        local_id=local_id,
        type=WorkItemType.USER_STORY,
        title=title,
        description=description,
        acceptance_criteria=[criterion],
        acceptance_criteria_origins=[ContentOrigin.SOURCE],
        provenance=[
            ProposedProvenance(
                json_path="$.requirements.functional_requirements[0]",
                field_name="text",
                excerpt=excerpt,
            )
        ],
        title_origin=ContentOrigin.GENERATED,
    )


def _scoped(stories: list[ProposedItem]) -> dict[str, ProposedItem]:
    proposal = _proposal(*stories)
    _scope_shared_story_outcomes(proposal)
    _scope_shared_story_outcomes(proposal)
    return {item.local_id: item for item in proposal.items}


def test_single_outcome_criterion_is_unchanged() -> None:
    criterion = "A clerk can record the received quantity."
    stories = _scoped(
        [
            _story_item(
                "receive",
                "Record Received Quantity",
                "A clerk records the received quantity against the shipment.",
                criterion,
                "A clerk can record the received quantity.",
            ),
            _story_item(
                "bin",
                "Confirm Reserve Bin",
                "A clerk confirms the reserve bin for the shipment.",
                "A clerk can confirm the reserve bin.",
                "A clerk can confirm the reserve bin.",
            ),
        ]
    )
    assert stories["receive"].acceptance_criteria == [criterion]


def test_billing_coordinated_verbs_keep_actor_and_object() -> None:
    shared = "Business customers can successfully upgrade or downgrade an active subscription"
    stories = _scoped(
        [
            _story_item(
                "upgrade",
                "Subscription Upgrades",
                "A customer upgrades an active subscription.",
                shared,
                "Customers can upgrade an active subscription.",
            ),
            _story_item(
                "downgrade",
                "Subscription Downgrades",
                "A customer downgrades an active subscription.",
                shared,
                "Customers can downgrade an active subscription.",
            ),
        ]
    )
    upgrade = stories["upgrade"].acceptance_criteria[0]
    downgrade = stories["downgrade"].acceptance_criteria[0]
    assert upgrade == "Business customers can successfully upgrade an active subscription."
    assert downgrade == "Business customers can successfully downgrade an active subscription."
    assert stories["upgrade"].acceptance_criteria_origins == [ContentOrigin.GENERATED]
    assert stories["downgrade"].acceptance_criteria_origins == [ContentOrigin.GENERATED]
    assert "downgrade" not in upgrade.casefold()
    assert "upgrade" not in downgrade.casefold()


def test_billing_actor_phrase_does_not_truncate_other_outcomes() -> None:
    search = (
        "Finance staff can search for customers and view their subscriptions, "
        "invoices, and payment statuses"
    )
    report = "Finance staff can generate reports displaying overdue invoices and failed payments."
    stories = _scoped(
        [
            _story_item(
                "search",
                "Customer Search and Inspection",
                "As a finance staff member, I want to search for customers and view their "
                "subscriptions, invoices, and payment statuses so that I can inspect account states.",
                search,
                search,
            ),
            _story_item(
                "report",
                "Overdue Invoice Report",
                "As a finance staff member, I want to generate reports displaying overdue "
                "invoices and failed payments.",
                report,
                report,
            ),
            _story_item(
                "access",
                "Finance Staff Access Restriction",
                "Finance staff access is restricted to authorized users.",
                "Finance staff access is restricted to authorized users.",
                "Finance staff access must be restricted to authorized users.",
            ),
        ]
    )
    search_text = f"{stories['search'].description} {stories['search'].acceptance_criteria[0]}"
    report_text = f"{stories['report'].description} {stories['report'].acceptance_criteria[0]}"
    assert "invoices, and payment statuses" in search_text
    assert "search for customers" in search_text
    assert "overdue invoices" in report_text
    assert "failed payments" in report_text
    assert "generate reports" in report_text
    assert stories["report"].acceptance_criteria != ["Failed payments."]


def test_migration_phrase_does_not_drop_the_validation_outcome() -> None:
    description = (
        "The data migration import must identify invalid records and potential "
        "duplicates before importing data."
    )
    stories = _scoped(
        [
            _story_item(
                "validate",
                "Pre-Import Validation and Duplicate Detection",
                description,
                "The process identifies invalid records and potential duplicates before importing data.",
                "The process must identify invalid records and potential duplicates before importing data.",
            ),
            _story_item(
                "migrate",
                "CSV Data Migration Import",
                "Existing records are migrated from CSV files.",
                "Existing records are migrated from CSV files.",
                "Existing customer and subscription data must be migrated from CSV files.",
            ),
        ]
    )
    assert "identify invalid records" in stories["validate"].description
    assert "potential duplicates" in stories["validate"].description


def test_clinic_booking_verbs_stay_complete_and_listed_results_stay_whole() -> None:
    shared = "A patient can book or reschedule an upcoming appointment."
    listed = "A nurse can view allergies, current medications, and recent lab results."
    stories = _scoped(
        [
            _story_item(
                "book",
                "Book Appointment",
                "A patient books an upcoming appointment.",
                shared,
                "A patient can book an upcoming appointment.",
            ),
            _story_item(
                "reschedule",
                "Reschedule Appointment",
                "A patient reschedules an upcoming appointment.",
                shared,
                "A patient can reschedule an upcoming appointment.",
            ),
            _story_item(
                "chart",
                "View Clinical Chart",
                "A nurse views allergies, current medications, and recent lab results.",
                listed,
                listed,
            ),
        ]
    )
    assert stories["book"].acceptance_criteria == [
        "A patient can book an upcoming appointment."
    ]
    assert stories["reschedule"].acceptance_criteria == [
        "A patient can reschedule an upcoming appointment."
    ]
    assert stories["chart"].acceptance_criteria == [listed]


def test_warehouse_coordinated_systems_each_keep_the_predicate() -> None:
    shared = "The scanner integrates with the location service and the inventory ledger."
    stories = _scoped(
        [
            _story_item(
                "location",
                "Location Service Integration",
                "The scanner integrates with the location service.",
                shared,
                "The scanner integrates with the location service.",
            ),
            _story_item(
                "ledger",
                "Inventory Ledger Integration",
                "The scanner integrates with the inventory ledger.",
                shared,
                "The scanner integrates with the inventory ledger.",
            ),
        ]
    )
    location = stories["location"].acceptance_criteria[0]
    ledger = stories["ledger"].acceptance_criteria[0]
    assert location == "The scanner integrates with the location service."
    assert ledger == "The scanner integrates with the inventory ledger."
    assert "ledger" not in location.casefold()
    assert "location" not in ledger.casefold()


def test_shared_email_clause_keeps_the_system_subject() -> None:
    shared = (
        "The system automatically generates a new invoice upon subscription renewal "
        "and sends a payment confirmation email after a successful payment."
    )
    stories = _scoped(
        [
            _story_item(
                "renew",
                "Automatic Renewal Invoicing",
                "The system generates a renewal invoice.",
                shared,
                "The system must automatically generate invoices when a subscription renews.",
            ),
            _story_item(
                "email",
                "Payment Confirmation Emails",
                "The system sends a payment confirmation email.",
                shared,
                "The system must send payment confirmation emails after successful payments.",
            ),
        ]
    )
    email = stories["email"].acceptance_criteria[0]
    renewal = stories["renew"].acceptance_criteria[0]
    assert email.startswith("The system sends")
    assert "payment confirmation" in email.casefold()
    assert "renewal" not in email.casefold()
    assert "invoice" in renewal.casefold()
    assert not email.lower().startswith("sends ")


def _clinic_envelope() -> dict:
    return _complete(
        user_request="Patients book or reschedule an upcoming appointment.",
        executive_summary="Patients manage upcoming appointments.",
        requirements={
            "functional_requirements": [
                "A patient can book an upcoming appointment.",
                "A patient can reschedule an upcoming appointment.",
            ],
            "acceptance_criteria": [
                "A patient can book or reschedule an upcoming appointment."
            ],
            "non_functional_requirements": [],
        },
        solution={
            "key_capabilities": ["Book appointment.", "Reschedule appointment."],
        },
        sow={
            "in_scope": ["Appointments"],
            "out_of_scope": ["Billing"],
            "deliverables": ["Booking", "Reschedule"],
        },
    )


def test_canonical_billing_criteria_keep_provenance_and_generated_wording() -> None:
    payload = _complete(
        user_request="Customers upgrade or downgrade an active subscription.",
        executive_summary="Customers change an active subscription.",
        requirements={
            "functional_requirements": [
                "Customers can upgrade an active subscription.",
                "Customers can downgrade an active subscription.",
            ],
            "acceptance_criteria": [
                "Business customers can successfully upgrade or downgrade an active subscription"
            ],
            "non_functional_requirements": [],
        },
        solution={"key_capabilities": ["Subscription upgrades.", "Subscription downgrades."]},
        sow={
            "in_scope": ["Subscription changes"],
            "out_of_scope": ["Tax"],
            "deliverables": ["Upgrade", "Downgrade"],
        },
    )
    shared = payload["requirements"]["acceptance_criteria"][0]
    fr = payload["requirements"]["functional_requirements"]
    stories = [
        _story(
            "story-up",
            "Subscription Upgrades",
            "feat-0",
            0,
            fr[0],
            description="A customer upgrades an active subscription.",
            acceptance=[shared],
            ac_paths=[(0, shared)],
        ),
        _story(
            "story-down",
            "Subscription Downgrades",
            "feat-1",
            1,
            fr[1],
            description="A customer downgrades an active subscription.",
            acceptance=[shared],
            ac_paths=[(0, shared)],
        ),
    ]
    provider = MockProvider(responses=[_invalid(payload), _tree(payload, stories)])
    backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload), provider
    )
    by_title = {item.title: item for item in backlog.items}
    upgrade = by_title["Subscription Upgrades"]
    downgrade = by_title["Subscription Downgrades"]
    assert upgrade.acceptance_criteria == [
        "Business customers can successfully upgrade an active subscription."
    ]
    assert downgrade.acceptance_criteria == [
        "Business customers can successfully downgrade an active subscription."
    ]
    for item, path, excerpt in (
        (upgrade, "$.requirements.functional_requirements[0]", fr[0]),
        (downgrade, "$.requirements.functional_requirements[1]", fr[1]),
    ):
        cited = {(reference.json_path, reference.excerpt) for reference in item.provenance}
        assert (path, excerpt) in cited
        assert all(reference.excerpt != item.acceptance_criteria[0] for reference in item.provenance)
        assert shared not in item.acceptance_criteria
    assert any(
        item.code is FindingCode.GENERATED_CONTENT and "acceptance_criteria" in item.message
        for item in findings.items
    )
    assert findings.llm_calls[1].accepted is True
    assert not any(item.code is FindingCode.DUPLICATE_ACCEPTANCE_CRITERION for item in findings.items)


def test_unsplittable_shared_list_is_not_chopped() -> None:
    payload = _clinic_envelope()
    shared = "A nurse can view allergies, current medications, and recent lab results."
    note = "A nurse can record the visit note."
    payload["requirements"]["acceptance_criteria"][0] = shared
    payload["requirements"]["functional_requirements"] = [shared, note]
    stories = [
        _story(
            "story-chart",
            "View Clinical Chart",
            "feat-0",
            0,
            shared,
            description="A nurse views the clinical chart.",
            acceptance=[shared],
            ac_paths=[(0, shared)],
        ),
        _story(
            "story-note",
            "Record Visit Note",
            "feat-1",
            1,
            note,
            description="A nurse records the visit note.",
            acceptance=[shared],
            ac_paths=[(0, shared)],
        ),
    ]
    provider = MockProvider(responses=[_invalid(payload), _tree(payload, stories)])
    backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload), provider
    )
    chart = next(item for item in backlog.items if item.title == "View Clinical Chart")
    recorded = next(item for item in backlog.items if item.title == "Record Visit Note")
    assert chart.acceptance_criteria == [shared]
    assert recorded.acceptance_criteria == [note]
    assert all(reference.excerpt != recorded.acceptance_criteria[0] or reference.excerpt == note for reference in recorded.provenance)
    assert any(reference.excerpt == shared for reference in chart.provenance)
    assert findings.llm_calls[1].validation_passed is True
    assert not any(item.code is FindingCode.DUPLICATE_ACCEPTANCE_CRITERION for item in findings.items)


def test_unsplittable_duplicate_without_a_substitute_still_blocks() -> None:
    listed = "A nurse can view allergies, current medications, and recent lab results."
    stories = _scoped(
        [
            _story_item(
                "chart",
                "View Clinical Chart",
                "A nurse views the clinical chart.",
                listed,
                listed,
            ),
            _story_item(
                "audit",
                "Clinical Chart Audit",
                "A nurse views the clinical chart.",
                listed,
                listed,
            ),
        ]
    )
    assert stories["chart"].acceptance_criteria == [listed]
    assert stories["audit"].acceptance_criteria == [listed]


def test_fragment_is_not_serialized_as_the_source_criterion() -> None:
    payload = _clinic_envelope()
    fragment = "Reschedule an upcoming appointment."
    full = payload["requirements"]["acceptance_criteria"][0]
    fr = payload["requirements"]["functional_requirements"]
    stories = [
        _story(
            "story-book",
            "Book Appointment",
            "feat-0",
            0,
            fr[0],
            description="A patient books an upcoming appointment.",
            acceptance=["A patient can book an upcoming appointment."],
        ),
        _story(
            "story-move",
            "Reschedule Appointment",
            "feat-1",
            1,
            fr[1],
            description="A patient reschedules an upcoming appointment.",
            acceptance=[fragment],
            ac_paths=[(0, full)],
        ),
    ]
    stories[1]["acceptance_criteria_origins"] = ["source"]
    provider = MockProvider(responses=[_tree(payload, stories)])
    backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload), provider, repair=False
    )
    moved = next(item for item in backlog.items if item.title == "Reschedule Appointment")
    assert moved.acceptance_criteria == [fragment]
    assert all(reference.excerpt != fragment for reference in moved.provenance)
    assert any(reference.excerpt == fr[1] for reference in moved.provenance)
    assert any(reference.excerpt == full for reference in moved.provenance)
    assert any(
        item.code is FindingCode.GENERATED_CONTENT
        and moved.canonical_id in item.canonical_ids
        and "acceptance_criteria" in item.message
        for item in findings.items
    )
