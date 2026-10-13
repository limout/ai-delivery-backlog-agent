"""Regression tests for Feature grouping, Task salvage, and NFR provenance."""

from __future__ import annotations

import pytest

from backlog_agent.backlog.decompose import decompose_backlog
from backlog_agent.llm.exceptions import DecompositionValidationError
from backlog_agent.backlog.models import ApprovalState, WorkItemStatus, WorkItemType
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import FindingCode
from backlog_agent.llm.mock import MockProvider
from backlog_agent.source.model import looks_like_delivery_process_constraint
from tests.copilot_envelopes import customer_portal_nine_outcomes_envelope
from tests.test_decompose_generic import _item, _src


def _envelope():
    return CopilotWorkflowResponseV1.model_validate(customer_portal_nine_outcomes_envelope())


def _facts():
    payload = customer_portal_nine_outcomes_envelope()
    return payload, payload["requirements"]["functional_requirements"], payload["requirements"]["acceptance_criteria"]


def _story(
    local_id: str,
    index: int,
    parent: str,
    fr: list[str],
    ac: list[str],
    *,
    description: str,
    extra_tests: list[str] | None = None,
) -> dict:
    return _item(
        local_id,
        "user_story",
        fr[index],
        parent=parent,
        provenance=[
            _src(
                f"$.requirements.functional_requirements[{index}]",
                "functional_requirements",
                fr[index],
            ),
            _src(
                f"$.requirements.acceptance_criteria[{index}]",
                "acceptance_criteria",
                ac[index],
            ),
        ],
        title_origin="source",
        description=description,
        ac=[ac[index]],
        ac_origins=["source"],
        tests=[f"Assert the expected result for requirement {index + 1}.", *(extra_tests or [])],
    )


def _catch_all_proposal(
    *,
    sap_task: bool = False,
    confirm_access_task: bool = False,
    enumerate_in_description: bool = False,
    cover_encryption: bool = True,
) -> dict:
    payload, fr, ac = _facts()
    nfr = payload["requirements"]["non_functional_requirements"][0]
    stories = [
        _story("story-auth", 0, "feat-core", fr, ac, description="A customer authenticates before opening protected pages."),
        _story("story-profile", 1, "feat-core", fr, ac, description="A customer updates a saved contact detail and sees the new value."),
        _story("story-orders", 2, "feat-core", fr, ac, description="A customer places a product order and sees it listed."),
        _story("story-requests", 3, "feat-core", fr, ac, description="A customer submits a service request and receives an identifier."),
        _story(
            "story-pay",
            4,
            "feat-core",
            fr,
            ac,
            description="A customer pays an outstanding balance and sees the reduced amount.",
            extra_tests=(
                ["Assert payment data in transit is encrypted."] if cover_encryption else None
            ),
        ),
        _story("story-notify", 5, "feat-core", fr, ac, description="A customer is notified when an order or request status changes."),
        _story(
            "story-sap",
            6,
            "feat-core",
            fr,
            ac,
            description="The portal integrates customer and order records with the existing SAP system.",
        ),
        _story(
            "story-stripe",
            7,
            "feat-core",
            fr,
            ac,
            description="The portal integrates card payments with the existing Stripe account.",
        ),
        _story("story-status", 8, "feat-core", fr, ac, description="A customer views the current status of submitted orders and service requests."),
    ]
    for story in stories:
        story["provenance"].append(
            _src(
                "$.requirements.non_functional_requirements[0]",
                "non_functional_requirements",
                nfr,
            )
        )
    items = [
        _item(
            "epic-1",
            "epic",
            "Customer self-service portal",
            parent=None,
            provenance=[_src("$.user_request", "user_request", payload["user_request"])],
            title_origin="generated",
            description="",
        ),
        _item(
            "feat-core",
            "feature",
            "Customer Management and Core Capabilities",
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[0]", "key_capabilities", payload["solution"]["key_capabilities"][0])],
            title_origin="generated",
            description=(
                "Covers authentication, profile contact details, product orders, "
                "service requests, payments, notifications, records-system integration, "
                "card-payment integration, and status visibility."
                if enumerate_in_description
                else ""
            ),
        ),
        *stories,
    ]
    if sap_task:
        items.append(
            _item(
                "task-sap-map",
                "task",
                "Map order identifiers onto the documented SAP records interface",
                parent="story-sap",
                provenance=[
                    _src(
                        "$.requirements.functional_requirements[6]",
                        "functional_requirements",
                        fr[6],
                    )
                ],
                title_origin="generated",
                description="Use the documented SAP records interface to store a placed order without inventing extra fields.",
            )
        )
    if confirm_access_task:
        items.append(
            _item(
                "task-sap-access",
                "task",
                "Investigate whether SAP API access exists",
                parent="story-sap",
                provenance=[
                    _src(
                        "$.delivery_plan.dependencies[0]",
                        "dependencies",
                        payload["delivery_plan"]["dependencies"][0],
                    )
                ],
                title_origin="generated",
                description="Confirm that API access and technical documentation are available.",
            )
        )
    return {"items": items, "prerequisites": []}


def _grouped_proposal(*, sap_task: bool = True, stripe_task: bool = True, confirm_access_task: bool = False) -> dict:
    payload, fr, ac = _facts()
    caps = payload["solution"]["key_capabilities"]
    encrypt = payload["requirements"]["non_functional_requirements"][1]
    items = [
        _item(
            "epic-1",
            "epic",
            "Customer self-service portal",
            parent=None,
            provenance=[
                _src("$.user_request", "user_request", payload["user_request"]),
                _src(
                    "$.requirements.non_functional_requirements[0]",
                    "non_functional_requirements",
                    payload["requirements"]["non_functional_requirements"][0],
                ),
            ],
            title_origin="generated",
            description="",
        ),
        _item(
            "feat-identity",
            "feature",
            caps[0],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[0]", "key_capabilities", caps[0])],
        ),
        _item(
            "feat-orders",
            "feature",
            caps[1],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[1]", "key_capabilities", caps[1])],
        ),
        _item(
            "feat-pay",
            "feature",
            "Card payments",
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[2]", "key_capabilities", caps[2])],
            title_origin="generated",
        ),
        _item(
            "feat-notify",
            "feature",
            "Status change notifications",
            parent="epic-1",
            provenance=[
                _src(
                    "$.requirements.functional_requirements[5]",
                    "functional_requirements",
                    fr[5],
                )
            ],
            title_origin="generated",
        ),
        _story("story-auth", 0, "feat-identity", fr, ac, description="A customer authenticates before opening protected pages."),
        _story("story-profile", 1, "feat-identity", fr, ac, description="A customer updates a saved contact detail and sees the new value."),
        _story("story-orders", 2, "feat-orders", fr, ac, description="A customer places a product order and sees it listed."),
        _story("story-requests", 3, "feat-orders", fr, ac, description="A customer submits a service request and receives an identifier."),
        _story(
            "story-sap",
            6,
            "feat-orders",
            fr,
            ac,
            description="The portal integrates customer and order records with the existing SAP system.",
        ),
        _story("story-status", 8, "feat-orders", fr, ac, description="A customer views the current status of submitted orders and service requests."),
        _story(
            "story-pay",
            4,
            "feat-pay",
            fr,
            ac,
            description=(
                "A customer pays an outstanding balance and sees the reduced amount. "
                "Payment data in transit is encrypted."
            ),
            extra_tests=["Assert payment data in transit is encrypted."],
        ),
        _story("story-notify", 5, "feat-notify", fr, ac, description="A customer is notified when an order or request status changes."),
        _story(
            "story-stripe",
            7,
            "feat-pay",
            fr,
            ac,
            description="The portal integrates card payments with the existing Stripe account.",
        ),
    ]
    pay = next(item for item in items if item["local_id"] == "story-pay")
    pay["provenance"].append(
        _src(
            "$.requirements.non_functional_requirements[1]",
            "non_functional_requirements",
            encrypt,
        )
    )
    if sap_task:
        items.append(
            _item(
                "task-sap-map",
                "task",
                "Map order identifiers onto the documented SAP records interface",
                parent="story-sap",
                provenance=[
                    _src(
                        "$.requirements.functional_requirements[6]",
                        "functional_requirements",
                        fr[6],
                    )
                ],
                title_origin="generated",
                description="Use the documented SAP records interface to store a placed order without inventing extra fields.",
            )
        )
    if stripe_task:
        items.append(
            _item(
                "task-stripe-settle",
                "task",
                "Record a Stripe settlement against the documented payment confirmation",
                parent="story-stripe",
                provenance=[
                    _src(
                        "$.requirements.functional_requirements[7]",
                        "functional_requirements",
                        fr[7],
                    )
                ],
                title_origin="generated",
                description="Record settlement using the documented Stripe confirmation without inventing endpoints.",
            )
        )
    if confirm_access_task:
        items.append(
            _item(
                "task-sap-access",
                "task",
                "Investigate whether SAP API access exists",
                parent="story-sap",
                provenance=[
                    _src(
                        "$.delivery_plan.dependencies[0]",
                        "dependencies",
                        payload["delivery_plan"]["dependencies"][0],
                    )
                ],
                title_origin="generated",
                description="Confirm that API access and technical documentation are available.",
            )
        )
    return {"items": items, "prerequisites": []}


def _counts(backlog) -> dict[str, int]:
    counts = {name: 0 for name in ("epic", "feature", "user_story", "task", "subtask")}
    for item in backlog.items:
        counts[item.type.value] += 1
    return counts


def _fr_paths(backlog) -> set[str]:
    return {
        reference.json_path
        for item in backlog.items
        if item.type is WorkItemType.USER_STORY
        for reference in item.provenance
        if reference.json_path.startswith("$.requirements.functional_requirements")
    }


def _assert_question_task_under_sap(backlog) -> None:
    by_id = {item.canonical_id: item for item in backlog.items}
    questions = [
        item
        for item in backlog.items
        if item.type is WorkItemType.TASK
        and any(ref.json_path == "$.requirements.open_questions[0]" for ref in item.provenance)
    ]
    assert len(questions) == 1
    assert "SAP" in by_id[questions[0].parent_id].title


def test_catch_all_feature_is_quality_repaired_into_capability_groups() -> None:
    provider = MockProvider(responses=[_catch_all_proposal(), _grouped_proposal()])
    backlog, findings = decompose_backlog(_envelope(), provider)

    assert len(provider.calls) == 2
    assert FindingCode.FEATURE_CHILD_MISALIGNMENT.value in provider.calls[1][0]
    counts = _counts(backlog)
    assert counts["epic"] == 1
    assert counts["feature"] >= 3
    assert counts["feature"] <= 6
    assert counts["user_story"] == 9
    # Two implementation tasks plus one clarification task for the open source question.
    assert counts["task"] == 3
    _assert_question_task_under_sap(backlog)
    assert counts["subtask"] == 0
    titles = {item.title for item in backlog.items if item.type is WorkItemType.FEATURE}
    assert "Customer authentication and profile" in titles
    assert "Orders and service requests" in titles
    assert "Customer Management and Core Capabilities" not in titles
    assert len(_fr_paths(backlog)) == 9
    assert all(item.status is WorkItemStatus.TODO for item in backlog.items)
    assert all(item.approval_state is ApprovalState.NOT_APPROVED for item in backlog.items)
    assert findings.llm_calls[1].accepted is True


def test_enumerated_description_does_not_make_a_catch_all_feature_cohesive() -> None:
    enumerated = _catch_all_proposal(enumerate_in_description=True)
    provider = MockProvider(responses=[enumerated, _grouped_proposal()])
    backlog, findings = decompose_backlog(_envelope(), provider)

    assert len(provider.calls) == 2
    assert FindingCode.FEATURE_CHILD_MISALIGNMENT.value in provider.calls[1][0]
    assert "distinct capability clusters" in provider.calls[1][0]
    assert "listing those areas" in provider.calls[1][0].lower()
    titles = {item.title for item in backlog.items if item.type is WorkItemType.FEATURE}
    assert "Customer authentication and profile" in titles
    assert "Orders and service requests" in titles
    assert "Customer Management and Core Capabilities" not in titles
    assert len(_fr_paths(backlog)) == 9
    assert _counts(backlog)["task"] == 3
    _assert_question_task_under_sap(backlog)


def test_grouped_proposal_keeps_nine_outcomes_and_does_not_force_tasks_on_atomic_stories() -> None:
    backlog, findings = decompose_backlog(_envelope(), MockProvider(_grouped_proposal()), repair=False)
    counts = _counts(backlog)
    assert 3 <= counts["feature"] <= 6
    assert counts["user_story"] == 9
    assert counts["task"] == 3  # no task is forced on atomic stories; the question gets one
    _assert_question_task_under_sap(backlog)
    profile = next(item for item in backlog.items if "profile contact details" in item.title.lower())
    assert profile.child_ids == []
    assert any(
        item.code is FindingCode.ATOMIC_STORY and profile.canonical_id in item.canonical_ids
        for item in findings.items
    )
    sap = next(item for item in backlog.items if "SAP" in item.title)
    assert sap.child_ids
    assert not any(
        item.code is FindingCode.MISSING_IMPLEMENTATION_DETAIL and sap.canonical_id in item.canonical_ids
        for item in findings.items
    )


def test_established_access_cannot_become_an_investigation_task() -> None:
    proposal = _grouped_proposal(confirm_access_task=True)
    with pytest.raises(DecompositionValidationError, match="established source fact"):
        decompose_backlog(_envelope(), MockProvider(proposal), repair=False)


def test_invalid_repair_tasks_are_dropped_without_losing_feature_split() -> None:
    repaired = _grouped_proposal(confirm_access_task=True)
    provider = MockProvider(responses=[_catch_all_proposal(), repaired])
    backlog, findings = decompose_backlog(_envelope(), provider)
    assert 3 <= _counts(backlog)["feature"] <= 6
    assert _counts(backlog)["user_story"] == 9
    assert not any("Investigate whether SAP API access exists" in item.title for item in backlog.items)
    assert any("documented SAP records interface" in item.title for item in backlog.items)
    assert any(item.code is FindingCode.QUALITY_REPAIR_REJECTED for item in findings.items)


def test_invalid_repair_that_drops_stories_keeps_the_initial_tree() -> None:
    broken = _grouped_proposal()
    broken["items"] = [item for item in broken["items"] if item["local_id"] != "story-profile"]
    provider = MockProvider(responses=[_catch_all_proposal(), broken])
    backlog, findings = decompose_backlog(_envelope(), provider)
    assert _counts(backlog)["feature"] == 1
    assert _counts(backlog)["user_story"] == 9
    assert any(item.code is FindingCode.QUALITY_REPAIR_REJECTED for item in findings.items)
    assert any(item.code is FindingCode.FEATURE_CHILD_MISALIGNMENT for item in findings.items)
    assert findings.llm_calls[0].accepted is True
    assert findings.llm_calls[1].accepted is False


def test_incremental_release_nfr_is_not_copied_onto_every_story() -> None:
    assert looks_like_delivery_process_constraint(
        "The system shall be released incrementally rather than in one large final delivery."
    )
    backlog, findings = decompose_backlog(
        _envelope(), MockProvider(_catch_all_proposal(cover_encryption=False)), repair=False
    )
    nfr_path = "$.requirements.non_functional_requirements[0]"
    story_hits = [
        item
        for item in backlog.items
        if item.type is WorkItemType.USER_STORY
        and any(reference.json_path == nfr_path for reference in item.provenance)
    ]
    assert story_hits == []
    epic = next(item for item in backlog.items if item.type is WorkItemType.EPIC)
    assert any(reference.json_path == nfr_path for reference in epic.provenance)
    assert any(item.code is FindingCode.PLANNING_CONSTRAINT for item in findings.items)
    encrypt_path = "$.requirements.non_functional_requirements[1]"
    assert any(
        item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT
        and any(reference.json_path == encrypt_path for reference in item.source_references)
        for item in findings.items
    )


def _finding_ids_exist_on_tree(backlog, findings) -> None:
    ids = {item.canonical_id for item in backlog.items}
    for finding in findings.items:
        for canonical_id in finding.canonical_ids:
            assert canonical_id in ids


def test_zero_task_quality_repair_gets_grounded_task_and_matching_findings() -> None:
    repaired = _grouped_proposal(sap_task=False, stripe_task=False)
    provider = MockProvider(responses=[_catch_all_proposal(), repaired])
    backlog, findings = decompose_backlog(_envelope(), provider)
    counts = _counts(backlog)
    assert counts["user_story"] == 9
    assert counts["task"] >= 1
    assert counts["task"] <= 3
    titles = " ".join(item.title.lower() for item in backlog.items if item.type is WorkItemType.TASK)
    assert "unspecified" in titles or "mapping" in titles
    assert "access exists" not in titles
    profile = next(item for item in backlog.items if "profile contact details" in item.title.lower())
    assert profile.child_ids == []
    assert any(
        item.code is FindingCode.ATOMIC_STORY and profile.canonical_id in item.canonical_ids
        for item in findings.items
    )
    assert not any(
        item.code is FindingCode.MISSING_IMPLEMENTATION_DETAIL and profile.canonical_id in item.canonical_ids
        for item in findings.items
    )
    sap = next(item for item in backlog.items if "SAP" in item.title)
    assert sap.child_ids
    assert not any(
        item.code is FindingCode.MISSING_IMPLEMENTATION_DETAIL and sap.canonical_id in item.canonical_ids
        for item in findings.items
    )
    assert not any(item.code is FindingCode.FEATURE_CHILD_MISALIGNMENT for item in findings.items)
    _finding_ids_exist_on_tree(backlog, findings)
    assert findings.llm_calls[-1].accepted is True


def test_invalid_task_only_repair_is_not_accepted_as_an_empty_clean_tree() -> None:
    repaired = _grouped_proposal(sap_task=False, stripe_task=False, confirm_access_task=True)
    provider = MockProvider(responses=[_catch_all_proposal(), repaired])
    backlog, findings = decompose_backlog(_envelope(), provider)
    assert not any("Investigate whether SAP API access exists" in item.title for item in backlog.items)
    assert any(item.type is WorkItemType.TASK for item in backlog.items)
    sap = next(item for item in backlog.items if "SAP" in item.title)
    assert sap.child_ids
    _finding_ids_exist_on_tree(backlog, findings)


def test_warnings_match_the_accepted_grouped_tree() -> None:
    backlog, findings = decompose_backlog(_envelope(), MockProvider(_grouped_proposal()), repair=False)
    sap = next(item for item in backlog.items if "SAP" in item.title)
    assert not any(
        item.code is FindingCode.MISSING_IMPLEMENTATION_DETAIL and sap.canonical_id in item.canonical_ids
        for item in findings.items
    )
    assert not any(item.code is FindingCode.FEATURE_CHILD_MISALIGNMENT for item in findings.items)
    assert backlog.contract_id == "canonical.backlog.v1"
    assert all(item.status is WorkItemStatus.TODO for item in backlog.items)
    assert all(item.approval_state is ApprovalState.NOT_APPROVED for item in backlog.items)
