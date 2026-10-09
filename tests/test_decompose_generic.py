from __future__ import annotations

import pytest

from backlog_agent.backlog.decompose import decompose_backlog
from backlog_agent.backlog.models import WorkItemType
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import FindingCode
from backlog_agent.llm.exceptions import DecompositionValidationError
from backlog_agent.llm.mock import MockProvider
from backlog_agent.source import UnsupportedInputError
from tests.copilot_envelopes import library_holds_envelope, warehouse_picking_envelope


def _src(path: str, field: str, excerpt: str) -> dict[str, str]:
    return {"json_path": path, "field_name": field, "excerpt": excerpt}


def _item(
    local_id: str,
    type_name: str,
    title: str,
    *,
    parent: str | None,
    provenance: list[dict[str, str]],
    description: str = "Actor uses the capability and observes a concrete outcome.",
    depends: list[str] | None = None,
    ac: list[str] | None = None,
    tests: list[str] | None = None,
    title_origin: str = "source",
    ac_origins: list[str] | None = None,
) -> dict:
    payload = {
        "local_id": local_id,
        "type": type_name,
        "title": title,
        "description": description if type_name == "user_story" else "",
        "parent_local_id": parent,
        "depends_on_local_ids": depends or [],
        "acceptance_criteria": ac or [],
        "test_requirements": tests or [],
        "provenance": provenance,
        "title_origin": title_origin,
        "description_origin": "generated",
        "uncertainties": [],
    }
    if ac_origins is not None:
        payload["acceptance_criteria_origins"] = ac_origins
    if tests:
        payload["test_requirements_origins"] = ["generated"] * len(tests)
    return payload


def _library_proposal(*, cover_notify: bool = True, trace_nfr: bool = True) -> dict:
    payload = library_holds_envelope()
    fr = payload["requirements"]["functional_requirements"]
    ac = payload["requirements"]["acceptance_criteria"][0]
    caps = payload["solution"]["key_capabilities"]
    nfr = payload["requirements"]["non_functional_requirements"][0]
    items = [
        _item(
            "epic-1",
            "epic",
            "Member catalog and holds",
            parent=None,
            provenance=[_src("$.user_request", "user_request", payload["user_request"])],
            title_origin="generated",
            description="",
        ),
        _item(
            "feat-search",
            "feature",
            caps[0],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[0]", "key_capabilities", caps[0])],
        ),
        _item(
            "feat-hold",
            "feature",
            caps[1],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[1]", "key_capabilities", caps[1])],
        ),
        _item(
            "story-search",
            "user_story",
            fr[0],
            parent="feat-search",
            provenance=[_src("$.requirements.functional_requirements[0]", "functional_requirements", fr[0])],
            description=(
                "A signed-in member enters a title or author and receives matching holdings "
                "without paging through the full collection."
            ),
            ac=[ac],
            ac_origins=["source"],
            tests=["Run a catalog search fixture and assert matching holdings appear in ranked order."],
        ),
        _item(
            "story-hold",
            "user_story",
            fr[1],
            parent="feat-hold",
            provenance=[
                _src("$.requirements.functional_requirements[1]", "functional_requirements", fr[1]),
                *([_src("$.requirements.non_functional_requirements[0]", "non_functional_requirements", nfr)] if trace_nfr else []),
            ],
            description=(
                "A member selects an available copy and places a hold so the copy is reserved "
                "until pickup rather than remaining on the open shelf."
            ),
            tests=["Contract test: hold request against an available copy returns reserved status."],
        ),
    ]
    if cover_notify:
        items.append(
            _item(
                "story-notify",
                "user_story",
                fr[2],
                parent="feat-hold",
                provenance=[_src("$.requirements.functional_requirements[2]", "functional_requirements", fr[2])],
                description=(
                    "When a reserved copy reaches the pickup shelf, the member receives a notice "
                    "so they can collect it before the hold expires."
                ),
                tests=["Assert a pickup-ready event sends one notice to the holding member."],
                depends=["story-hold"],
            )
        )
    items.extend(
        [
            _item(
                "task-hold-api",
                "task",
                "Implement hold reservation against inventory availability",
                parent="story-hold",
                provenance=[_src("$.sow.in_scope[1]", "in_scope", payload["sow"]["in_scope"][1])],
                title_origin="generated",
            ),
            _item(
                "task-hold-ui",
                "task",
                "Add place-hold action on the holding detail screen",
                parent="story-hold",
                provenance=[_src("$.sow.in_scope[1]", "in_scope", payload["sow"]["in_scope"][1])],
                title_origin="generated",
            ),
            _item(
                "task-hold-notice",
                "task",
                "Queue pickup-ready notice after a successful hold",
                parent="story-hold",
                provenance=[_src("$.sow.deliverables[1]", "deliverables", payload["sow"]["deliverables"][1])],
                title_origin="generated",
            ),
        ]
    )
    return {
        "items": items,
        "prerequisites": [
            {
                "text": "Bibliographic vendor MARC feed access is not yet granted.",
                "status": "unresolved",
                "provenance": [
                    _src(
                        "$.delivery_plan.dependencies[0]",
                        "dependencies",
                        payload["delivery_plan"]["dependencies"][0],
                    )
                ],
            }
        ],
        "uncertainties": ["Identity provider for member login is not specified."],
    }


def _warehouse_proposal(*, omit_reprint: bool = False) -> dict:
    payload = warehouse_picking_envelope()
    fr = payload["requirements"]["functional_requirements"]
    ac = payload["requirements"]["acceptance_criteria"][0]
    caps = payload["solution"]["key_capabilities"]
    items = [
        _item(
            "epic-1",
            "epic",
            "Handheld warehouse picking",
            parent=None,
            provenance=[_src("$.user_request", "user_request", payload["user_request"])],
            title_origin="generated",
        ),
        _item(
            "feat-play",
            "feature",
            caps[0],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[0]", "key_capabilities", caps[0])],
        ),
        _item(
            "feat-scan",
            "feature",
            caps[1],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[1]", "key_capabilities", caps[1])],
        ),
        _item(
            "story-next",
            "user_story",
            fr[0],
            parent="feat-play",
            provenance=[_src("$.requirements.functional_requirements[0]", "functional_requirements", fr[0])],
            description=(
                "An operator opening an assigned list sees the next bin and SKU so they walk "
                "directly to the location instead of reading a paper ticket."
            ),
            tests=["Render the next incomplete pick and assert bin and SKU are visible."],
        ),
        _item(
            "story-scan",
            "user_story",
            fr[1],
            parent="feat-scan",
            provenance=[_src("$.requirements.functional_requirements[1]", "functional_requirements", fr[1])],
            description=(
                "An operator scans the presented bin and the pick is completed only when the "
                "scan matches the assigned location."
            ),
            ac=[ac],
            ac_origins=["source"],
            tests=["Scan the expected bin to complete; scan a different bin to keep the pick open."],
        ),
        _item(
            "task-scan-decode",
            "task",
            "Decode handheld barcode scans against the assigned bin",
            parent="story-scan",
            provenance=[_src("$.sow.in_scope[0]", "in_scope", payload["sow"]["in_scope"][0])],
            title_origin="generated",
        ),
        _item(
            "task-scan-complete",
            "task",
            "Mark the pick complete after a matching bin scan",
            parent="story-scan",
            provenance=[_src("$.sow.in_scope[0]", "in_scope", payload["sow"]["in_scope"][0])],
            title_origin="generated",
        ),
    ]
    if not omit_reprint:
        items.append(
            _item(
                "story-reprint",
                "user_story",
                fr[2],
                parent="feat-play",
                provenance=[_src("$.requirements.functional_requirements[2]", "functional_requirements", fr[2])],
                description=(
                    "A supervisor can reprint a completed pick list for audit without changing "
                    "the completed pick state."
                ),
                tests=["Reprint a completed list and assert pick statuses remain complete."],
            )
        )
    return {
        "items": items,
        "prerequisites": [
            {
                "text": "Barcode symbology for existing bins is unconfirmed.",
                "status": "unresolved",
                "provenance": [_src("$.sow.dependencies[0]", "dependencies", payload["sow"]["dependencies"][0])],
            }
        ],
        "uncertainties": ["Wi-Fi packet-loss target has no agreed measurement method."],
    }


def test_library_project_covers_frs_atomic_story_and_multi_task_story() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(library_holds_envelope())
    backlog, findings = decompose_backlog(envelope, MockProvider(_library_proposal()))

    stories = [item for item in backlog.items if item.type is WorkItemType.USER_STORY]
    search = next(item for item in stories if "search the catalog" in item.title.lower())
    hold = next(item for item in stories if "place a hold" in item.title.lower())
    assert search.child_ids == []
    assert len(hold.child_ids) == 3
    assert any(item.code is FindingCode.ATOMIC_STORY and search.canonical_id in item.canonical_ids for item in findings.items)
    assert not any(item.code is FindingCode.ATOMIC_STORY and hold.canonical_id in item.canonical_ids for item in findings.items)
    assert any(item.code is FindingCode.UNRESOLVED_PREREQUISITE for item in findings.items)
    assert any(item.code is FindingCode.AMBIGUOUS_DECOMPOSITION for item in findings.items)
    assert any(item.code is FindingCode.PLANNING_CONSTRAINT for item in findings.items)
    assert not any(item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT for item in findings.items)
    assert not any(item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT for item in findings.items)
    assert any(item.dependencies for item in backlog.items)
    generated = [item for item in findings.items if item.code is FindingCode.GENERATED_CONTENT]
    assert generated
    titles = {item.title for item in backlog.items}
    assert "Interlibrary loan routing" not in titles
    assert "Member experience" not in titles


def test_related_story_without_nfr_path_is_not_coverage() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(library_holds_envelope())
    proposal = _library_proposal(trace_nfr=False)
    _backlog, findings = decompose_backlog(envelope, MockProvider(proposal))

    uncovered = [
        item for item in findings.items if item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT
    ]
    assert len(uncovered) == 1
    assert "400 milliseconds" in uncovered[0].source_references[0].excerpt


def test_timeline_constraint_is_not_a_work_item() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(library_holds_envelope())
    proposal = _library_proposal()
    constraint = library_holds_envelope()["requirements"]["non_functional_requirements"][1]
    proposal["items"].append(
        _item(
            "feat-deadline",
            "feature",
            constraint,
            parent="epic-1",
            provenance=[
                _src("$.requirements.non_functional_requirements[1]", "non_functional_requirements", constraint)
            ],
        )
    )
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(envelope, MockProvider(proposal), repair=False)
    assert any("timeline or estimate" in error for error in exc.value.errors)


def test_warehouse_project_rejects_uncovered_requirement() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(warehouse_picking_envelope())
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(envelope, MockProvider(_warehouse_proposal(omit_reprint=True)), repair=False)
    assert any("Uncovered functional requirement" in error for error in exc.value.errors)
    assert any(
        item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT for item in exc.value.findings.items
    )


def test_warehouse_atomic_and_multi_task_stories_with_generated_provenance() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(warehouse_picking_envelope())
    backlog, findings = decompose_backlog(envelope, MockProvider(_warehouse_proposal()))

    stories = [item for item in backlog.items if item.type is WorkItemType.USER_STORY]
    next_pick = next(item for item in stories if "next pick" in item.title.lower())
    scan = next(item for item in stories if "confirm a pick" in item.title.lower())
    reprint = next(item for item in stories if "reprint" in item.title.lower())
    assert next_pick.child_ids == []
    assert reprint.child_ids == []
    assert len(scan.child_ids) == 2
    assert any("test_requirements" in item.message for item in findings.items if item.code is FindingCode.GENERATED_CONTENT)
    assert any(item.code is FindingCode.UNRESOLVED_PREREQUISITE for item in findings.items)
    nfr_uncovered = [
        item for item in findings.items if item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT
    ]
    assert nfr_uncovered
    assert "packet loss" in nfr_uncovered[0].source_references[0].excerpt.lower()
    titles = {item.title for item in backlog.items}
    assert "Packing optimization" not in titles


def test_out_of_scope_item_is_rejected_on_non_insurance_project() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(warehouse_picking_envelope())
    proposal = _warehouse_proposal()
    excluded = warehouse_picking_envelope()["sow"]["out_of_scope"][0]
    proposal["items"].append(
        _item(
            "feat-pack",
            "feature",
            excluded,
            parent="epic-1",
            provenance=[_src("$.sow.out_of_scope[0]", "out_of_scope", excluded)],
        )
    )
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(envelope, MockProvider(proposal), repair=False)
    assert any(item.code is FindingCode.OUT_OF_SCOPE_EXCLUDED for item in exc.value.findings.items)


def test_decompose_rejects_unsupported_input() -> None:
    with pytest.raises(UnsupportedInputError):
        decompose_backlog({"random": True}, MockProvider(_library_proposal()))
