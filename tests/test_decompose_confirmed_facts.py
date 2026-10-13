from __future__ import annotations

import copy

import pytest

from backlog_agent.backlog.decompose import decompose_backlog
from backlog_agent.backlog.models import WorkItemType
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import FindingCode
from backlog_agent.llm.exceptions import DecompositionValidationError
from backlog_agent.llm.mock import MockProvider
from tests.copilot_envelopes import _complete, library_holds_envelope, warehouse_picking_envelope
from tests.test_decompose_generic import _item, _library_proposal, _src, _warehouse_proposal


def campus_self_service_envelope(*, contradict_access: bool = False) -> dict:
    access = "Access to the campus SIS API and technical documentation"
    if contradict_access:
        access = "Access to the campus SIS API and technical documentation is not yet granted."
    return _complete(
        user_request=(
            "Build a campus student self-service portal for enrollment changes. "
            "API access and technical documentation for the campus SIS and the payments "
            "processor are available. The Product Owner and key business stakeholders can "
            "participate in regular reviews."
        ),
        executive_summary="Students submit enrollment changes and track status after submission.",
        requirements={
            "functional_requirements": [
                "The system shall let a student submit an enrollment change request.",
                "The system shall show the request status after submission.",
            ],
            "acceptance_criteria": [
                "Given a complete change request, when the student submits, then a status is shown."
            ],
            "non_functional_requirements": [
                "Enrollment change submissions must be encrypted in transit.",
            ],
            "open_questions": [
                "Webhook signature verification and sandbox credentials for the payments "
                "processor are unspecified."
            ],
        },
        solution={
            "key_capabilities": [
                "Enrollment change submission",
                "Request status visibility",
            ]
        },
        delivery_plan={
            "delivery_phases": ["Build"],
            "workstreams": ["Student experience"],
            "milestones": ["Portal MVP"],
            "dependencies": [
                access,
                "Access to the payments processor API and technical documentation",
                "Product Owner and key business stakeholders for regular reviews",
            ],
        },
        sow={
            "in_scope": ["Enrollment change requests", "Status visibility"],
            "out_of_scope": ["Degree audit redesign"],
            "deliverables": ["Change request form", "Status page"],
        },
    )


def _campus_proposal(
    payload: dict,
    *,
    confirm_access_task: bool = False,
    sandbox_task: bool = True,
    confirmed_prereq: bool = False,
    conflict_prereq: bool = False,
    empty_story_description: bool = False,
) -> dict:
    fr = payload["requirements"]["functional_requirements"]
    ac = payload["requirements"]["acceptance_criteria"][0]
    nfr = payload["requirements"]["non_functional_requirements"][0]
    caps = payload["solution"]["key_capabilities"]
    question = payload["requirements"]["open_questions"][0]
    items = [
        _item(
            "epic-1",
            "epic",
            "Campus enrollment self-service",
            parent=None,
            provenance=[_src("$.user_request", "user_request", payload["user_request"])],
            title_origin="generated",
            description="",
        ),
        _item(
            "feat-submit",
            "feature",
            caps[0],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[0]", "key_capabilities", caps[0])],
        ),
        _item(
            "feat-status",
            "feature",
            caps[1],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[1]", "key_capabilities", caps[1])],
        ),
        _item(
            "story-submit",
            "user_story",
            fr[0],
            parent="feat-submit",
            provenance=[
                _src(
                    "$.requirements.functional_requirements[0]",
                    "functional_requirements",
                    fr[0],
                ),
                _src(
                    "$.requirements.non_functional_requirements[0]",
                    "non_functional_requirements",
                    nfr,
                ),
            ],
            description=""
            if empty_story_description
            else (
                "A student completes an enrollment change form and submits it so the campus "
                "record can be updated."
            ),
            tests=["Submit a complete change request and assert it is accepted."],
        ),
        _item(
            "story-status",
            "user_story",
            fr[1],
            parent="feat-status",
            provenance=[
                _src(
                    "$.requirements.functional_requirements[1]",
                    "functional_requirements",
                    fr[1],
                )
            ],
            description=(
                "After submission the student sees the current request status instead of an "
                "empty confirmation page."
            ),
            ac=[ac],
            ac_origins=["source"],
            tests=["Submit a request and assert the status value is visible."],
        ),
    ]
    if confirm_access_task:
        items.append(
            _item(
                "task-confirm-docs",
                "task",
                "Confirm whether campus SIS API documentation is available",
                parent="story-submit",
                provenance=[
                    _src(
                        "$.delivery_plan.dependencies[0]",
                        "dependencies",
                        payload["delivery_plan"]["dependencies"][0],
                    )
                ],
                title_origin="generated",
            )
        )
    if sandbox_task:
        items.append(
            _item(
                "task-sandbox",
                "task",
                "Determine whether payments processor sandbox credentials can be issued",
                parent="story-submit",
                provenance=[
                    _src(
                        "$.requirements.open_questions[0]",
                        "open_questions",
                        question,
                    )
                ],
                title_origin="generated",
            )
        )
    prerequisites = []
    if confirmed_prereq:
        prerequisites.append(
            {
                "text": "Campus SIS API access and technical documentation remain unresolved.",
                "status": "unresolved",
                "provenance": [
                    _src(
                        "$.delivery_plan.dependencies[0]",
                        "dependencies",
                        payload["delivery_plan"]["dependencies"][0],
                    )
                ],
            }
        )
    if conflict_prereq:
        prerequisites.append(
            {
                "text": payload["delivery_plan"]["dependencies"][0],
                "status": "unresolved",
                "provenance": [
                    _src(
                        "$.delivery_plan.dependencies[0]",
                        "dependencies",
                        payload["delivery_plan"]["dependencies"][0],
                    )
                ],
            }
        )
    stakeholder = payload["delivery_plan"]["dependencies"][2]
    if confirmed_prereq:
        prerequisites.append(
            {
                "text": "Stakeholder availability for regular reviews is unresolved.",
                "status": "unresolved",
                "provenance": [_src("$.delivery_plan.dependencies[2]", "dependencies", stakeholder)],
            }
        )
    return {
        "items": items,
        "prerequisites": prerequisites,
        "uncertainties": [question] if sandbox_task else [],
    }


def test_investigation_of_established_access_is_rejected_without_citing_the_dependency() -> None:
    payload = campus_self_service_envelope()
    proposal = _campus_proposal(payload, sandbox_task=False)
    proposal["items"].append(
        _item(
            "task-exist",
            "task",
            "Investigate whether campus SIS API access exists",
            parent="story-submit",
            provenance=[
                _src(
                    "$.requirements.functional_requirements[0]",
                    "functional_requirements",
                    payload["requirements"]["functional_requirements"][0],
                )
            ],
            title_origin="generated",
            description="Confirm that API access and technical documentation are available.",
        )
    )
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(
            CopilotWorkflowResponseV1.model_validate(payload),
            MockProvider(proposal),
            repair=False,
        )
    assert any("confirm an established source fact" in error for error in exc.value.errors)


def test_existence_check_citing_only_a_dependency_uses_request_facts() -> None:
    payload = campus_self_service_envelope()
    proposal = _campus_proposal(payload, sandbox_task=False)
    dependency = payload["delivery_plan"]["dependencies"][0]
    proposal["items"].append(
        _item(
            "task-exist-dep",
            "task",
            "Verify that API access and technical documentation are available",
            parent="story-submit",
            provenance=[_src("$.delivery_plan.dependencies[0]", "dependencies", dependency)],
            title_origin="generated",
            description="Confirm that API access and documentation are available.",
        )
    )
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(
            CopilotWorkflowResponseV1.model_validate(payload),
            MockProvider(proposal),
            repair=False,
        )
    assert any("confirm an established source fact" in error for error in exc.value.errors)


def test_library_confirmed_feed_access_rejects_availability_task() -> None:
    payload = copy.deepcopy(library_holds_envelope())
    payload["user_request"] = (
        payload["user_request"]
        + " Bibliographic feed documentation and API access are available."
    )
    payload["delivery_plan"]["dependencies"] = [
        "Access to the bibliographic vendor MARC feed and technical documentation"
    ]
    proposal = _library_proposal()
    fr = payload["requirements"]["functional_requirements"][0]
    proposal["items"].append(
        _item(
            "task-feed-docs",
            "task",
            "Check whether bibliographic documentation is available",
            parent="story-search",
            provenance=[_src("$.requirements.functional_requirements[0]", "functional_requirements", fr)],
            title_origin="generated",
            description="Verify whether API access and documentation are available.",
        )
    )
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(
            CopilotWorkflowResponseV1.model_validate(payload),
            MockProvider(proposal),
            repair=False,
        )
    assert any("confirm an established source fact" in error for error in exc.value.errors)


def test_task_that_restates_the_parent_story_is_rejected() -> None:
    payload = campus_self_service_envelope()
    proposal = _campus_proposal(payload, sandbox_task=False)
    proposal["items"].append(
        _item(
            "task-restate",
            "task",
            "Let a student submit an enrollment change request",
            parent="story-submit",
            provenance=[
                _src(
                    "$.requirements.functional_requirements[0]",
                    "functional_requirements",
                    payload["requirements"]["functional_requirements"][0],
                )
            ],
            title_origin="generated",
            description="The student submits an enrollment change request.",
        )
    )
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(
            CopilotWorkflowResponseV1.model_validate(payload),
            MockProvider(proposal),
            repair=False,
        )
    assert any("restates the parent" in error for error in exc.value.errors)


def test_warehouse_unconfirmed_detail_gets_a_task_on_the_matching_story() -> None:
    payload = warehouse_picking_envelope()
    proposal = _warehouse_proposal()
    proposal["items"] = [
        item
        for item in proposal["items"]
        if not str(item["local_id"]).startswith("task-")
    ]
    backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload),
        MockProvider(proposal),
        repair=False,
    )
    scan = next(item for item in backlog.items if "scanning the bin" in item.title.lower())
    assert scan.child_ids
    child_titles = [
        next(item for item in backlog.items if item.canonical_id == child_id).title.lower()
        for child_id in scan.child_ids
    ]
    assert any("barcode" in title or "symbology" in title or "bin" in title for title in child_titles)
    assert not any("exist" in title and "access" in title for title in child_titles)
    next_pick = next(item for item in backlog.items if "next pick location" in item.title.lower())
    assert next_pick.child_ids == []
    assert any(
        item.code is FindingCode.ATOMIC_STORY and next_pick.canonical_id in item.canonical_ids
        for item in findings.items
    )
    ids = {item.canonical_id for item in backlog.items}
    for finding in findings.items:
        assert all(canonical_id in ids for canonical_id in finding.canonical_ids)


def test_distinct_implementation_task_is_retained() -> None:
    payload = campus_self_service_envelope()
    proposal = _campus_proposal(payload, sandbox_task=False)
    proposal["items"].append(
        _item(
            "task-map",
            "task",
            "Map enrollment change fields onto the documented SIS records interface",
            parent="story-submit",
            provenance=[
                _src(
                    "$.requirements.functional_requirements[0]",
                    "functional_requirements",
                    payload["requirements"]["functional_requirements"][0],
                )
            ],
            title_origin="generated",
            description=(
                "Use the documented SIS interface to persist a submitted change "
                "without inventing extra fields."
            ),
        )
    )
    backlog, _findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload),
        MockProvider(proposal),
        repair=False,
    )
    assert any(
        item.type is WorkItemType.TASK and "documented SIS" in item.title
        for item in backlog.items
    )


def test_confirmed_dependency_rejects_existence_check_task() -> None:
    payload = campus_self_service_envelope()
    proposal = _campus_proposal(payload, confirm_access_task=True)
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(
            CopilotWorkflowResponseV1.model_validate(payload),
            MockProvider(proposal),
            repair=False,
        )
    assert any("confirm an established source fact" in error for error in exc.value.errors)


def test_confirmed_dependency_does_not_emit_unresolved_prerequisite() -> None:
    payload = campus_self_service_envelope()
    proposal = _campus_proposal(payload, confirmed_prereq=True)
    _backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload),
        MockProvider(proposal),
        repair=False,
    )
    unresolved = [item for item in findings.items if item.code is FindingCode.UNRESOLVED_PREREQUISITE]
    joined = " ".join(item.message.lower() for item in unresolved)
    assert "sis api" not in joined
    assert "stakeholder" not in joined
    assert not unresolved


def test_genuine_unknown_integration_detail_stays_visible() -> None:
    payload = campus_self_service_envelope()
    proposal = _campus_proposal(payload)
    backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload),
        MockProvider(proposal),
        repair=False,
    )
    assert any(
        item.type is WorkItemType.TASK and "sandbox credentials" in item.title.lower()
        for item in backlog.items
    )
    assert any(
        item.code is FindingCode.AMBIGUOUS_DECOMPOSITION and "sandbox" in item.message.lower()
        for item in findings.items
    )


def test_contradictory_access_statements_stay_unresolved() -> None:
    payload = campus_self_service_envelope(contradict_access=True)
    proposal = _campus_proposal(payload, conflict_prereq=True, sandbox_task=False)
    _backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload),
        MockProvider(proposal),
        repair=False,
    )
    unresolved = [item for item in findings.items if item.code is FindingCode.UNRESOLVED_PREREQUISITE]
    assert unresolved
    assert "not yet granted" in unresolved[0].message.lower()


def test_repair_may_drop_established_existence_check_without_omission_warning() -> None:
    payload = campus_self_service_envelope()
    initial = _campus_proposal(payload, confirm_access_task=True, empty_story_description=True)
    repaired = _campus_proposal(payload, confirm_access_task=False)
    _backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload),
        MockProvider(responses=[initial, repaired]),
    )
    assert not any("documentation is available" in item.title.lower() for item in _backlog.items)
    assert not any("omitted grounded investigation task" in item.message for item in findings.items)


def test_behavior_is_independent_of_insurance_wording() -> None:
    payload = campus_self_service_envelope()
    blob = str(payload).lower()
    assert "insurance" not in blob
    assert "policy" not in blob
    proposal = _campus_proposal(payload, confirmed_prereq=True)
    _backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload),
        MockProvider(copy.deepcopy(proposal)),
        repair=False,
    )
    assert not any(item.code is FindingCode.UNRESOLVED_PREREQUISITE for item in findings.items)
    assert any("sandbox" in item.title.lower() for item in _backlog.items)
