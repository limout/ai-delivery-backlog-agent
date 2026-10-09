from __future__ import annotations

import pytest

from backlog_agent.backlog.decompose import decompose_backlog
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import FindingCode
from backlog_agent.llm.exceptions import DecompositionValidationError
from backlog_agent.llm.mock import MockProvider
from tests.copilot_envelopes import municipal_permits_envelope
from tests.test_decompose_generic import _item, _src


def _envelope():
    return CopilotWorkflowResponseV1.model_validate(municipal_permits_envelope())


def _base_items(*, split_intake: bool, atomic_calendar: bool = True) -> list[dict]:
    payload = municipal_permits_envelope()
    fr = payload["requirements"]["functional_requirements"]
    caps = payload["solution"]["key_capabilities"]
    nfr = payload["requirements"]["non_functional_requirements"][0]
    items = [
        _item(
            "epic-1",
            "epic",
            "Resident permitting desk",
            parent=None,
            provenance=[_src("$.user_request", "user_request", payload["user_request"])],
            title_origin="generated",
            description="",
        ),
        _item(
            "feat-desk",
            "feature",
            caps[0],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[0]", "key_capabilities", caps[0])],
        ),
        _item(
            "feat-clerk",
            "feature",
            caps[1],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[1]", "key_capabilities", caps[1])],
        ),
        _item(
            "story-calendar",
            "user_story",
            fr[1],
            parent="feat-desk",
            provenance=[_src("$.requirements.functional_requirements[1]", "functional_requirements", fr[1])],
            description="A resident opens the office calendar and sees dates when counters are closed.",
            tests=["Render closed dates from a fixture calendar."],
        ),
        _item(
            "story-letter",
            "user_story",
            fr[2],
            parent="feat-clerk",
            provenance=[_src("$.requirements.functional_requirements[2]", "functional_requirements", fr[2])],
            description=(
                "A clerk attaches a decision letter, the file is stored, and the upload retries "
                "on timeout so the resident can download the outcome without visiting the counter."
            ),
            tests=["Upload a PDF and assert it is stored and listed on the application."],
        ),
        _item(
            "task-letter-store",
            "task",
            "Persist the decision letter as an application document",
            parent="story-letter",
            provenance=[_src("$.sow.deliverables[1]", "deliverables", payload["sow"]["deliverables"][1])],
            title_origin="generated",
        ),
        _item(
            "task-letter-scan",
            "task",
            "Virus-scan uploaded decision letters before they are stored",
            parent="story-letter",
            provenance=[_src("$.sow.in_scope[1]", "in_scope", payload["sow"]["in_scope"][1])],
            title_origin="generated",
        ),
        _item(
            "task-letter-encrypt",
            "task",
            "Encrypt stored decision letters at rest",
            parent="story-letter",
            provenance=[
                _src("$.requirements.non_functional_requirements[0]", "non_functional_requirements", nfr)
            ],
            title_origin="generated",
        ),
    ]
    if split_intake:
        items.insert(
            3,
            _item(
                "story-apply",
                "user_story",
                "Submit a permit application",
                parent="feat-desk",
                provenance=[_src("$.requirements.functional_requirements[0]", "functional_requirements", fr[0])],
                title_origin="generated",
                description=(
                    "A resident starts an application, identifies the permit type, and stores a draft "
                    "so they can finish later."
                ),
                tests=["Create a draft application for a known permit type."],
            ),
        )
        items.append(
            _item(
                "task-apply-draft",
                "task",
                "Save a draft application for the selected permit type",
                parent="story-apply",
                provenance=[_src("$.sow.deliverables[0]", "deliverables", payload["sow"]["deliverables"][0])],
                title_origin="generated",
            )
        )
    else:
        items.insert(
            3,
            _item(
                "story-mega",
                "user_story",
                fr[0],
                parent="feat-desk",
                provenance=[_src("$.requirements.functional_requirements[0]", "functional_requirements", fr[0])],
                description=(
                    "A resident applies for a permit, uploads evidence, pays the fee, and tracks "
                    "status until a clerk publishes a decision."
                ),
                ac=[
                    "Application is stored.",
                    "Evidence is uploaded.",
                    "Fee is paid.",
                    "Status is visible.",
                ],
                ac_origins=["generated", "generated", "generated", "generated"],
                tests=["Walk the whole journey in one scenario."],
            ),
        )
    if not atomic_calendar:
        items = [item for item in items if item["local_id"] != "story-calendar"]
    return items


def _proposal(items: list[dict], *, uncertainties: list[str] | None = None) -> dict:
    payload = municipal_permits_envelope()
    return {
        "items": items,
        "prerequisites": [
            {
                "text": "Payment processor sandbox access is not granted.",
                "status": "unresolved",
                "provenance": [
                    _src("$.sow.dependencies[0]", "dependencies", payload["sow"]["dependencies"][0])
                ],
            }
        ],
        "uncertainties": uncertainties
        or ["The evidence file format and maximum size are unspecified."],
    }


def test_split_intake_keeps_calendar_atomic_and_traces_nfr() -> None:
    items = _base_items(split_intake=True)
    backlog, findings = decompose_backlog(_envelope(), MockProvider(_proposal(items)))

    calendar = next(item for item in backlog.items if "holiday calendar" in item.title.lower())
    letter = next(item for item in backlog.items if "decision letter" in item.title.lower())
    assert calendar.child_ids == []
    assert len(letter.child_ids) == 3
    codes = {item.code for item in findings.items if calendar.canonical_id in item.canonical_ids}
    assert FindingCode.ATOMIC_STORY in codes
    assert FindingCode.OVERSIZED_STORY not in {
        item.code for item in findings.items if letter.canonical_id in item.canonical_ids
    }
    assert any(item.code is FindingCode.UNRESOLVED_PREREQUISITE for item in findings.items)
    assert any(item.code is FindingCode.AMBIGUOUS_DECOMPOSITION for item in findings.items)
    assert not any(item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT for item in findings.items)
    assert not any(item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT for item in findings.items)


def test_bundled_intake_story_is_reported_as_oversized() -> None:
    items = _base_items(split_intake=False)
    _backlog, findings = decompose_backlog(_envelope(), MockProvider(_proposal(items)), repair=False)
    oversized = [item for item in findings.items if item.code is FindingCode.OVERSIZED_STORY]
    assert oversized
    assert any("independently testable" in item.message for item in oversized)


def test_broad_task_is_reported_without_inventing_effort() -> None:
    items = _base_items(split_intake=True)
    items.append(
        _item(
            "task-broad",
            "task",
            "Build the entire journey",
            parent="story-apply",
            provenance=[
                _src(
                    "$.sow.in_scope[0]",
                    "in_scope",
                    municipal_permits_envelope()["sow"]["in_scope"][0],
                )
            ],
            title_origin="generated",
        )
    )
    _backlog, findings = decompose_backlog(_envelope(), MockProvider(_proposal(items)))
    broad = [item for item in findings.items if item.code is FindingCode.BROAD_TASK]
    assert broad
    assert all("not an estimate" in item.message.lower() or "heuristic" in item.message.lower() for item in broad)


def test_missing_source_detail_is_a_finding_not_invented_work() -> None:
    items = _base_items(split_intake=True)
    items.append(
        _item(
            "task-idp",
            "task",
            "Choose and implement an unspecified identity provider",
            parent="story-apply",
            provenance=[
                _src(
                    "$.requirements.open_questions[0]",
                    "open_questions",
                    municipal_permits_envelope()["requirements"]["open_questions"][0],
                )
            ],
            title_origin="generated",
        )
    )
    # The task is concrete enough to pass title checks; uncertainty must still be visible.
    _backlog, findings = decompose_backlog(_envelope(), MockProvider(_proposal(items)))
    assert any(item.code is FindingCode.AMBIGUOUS_DECOMPOSITION for item in findings.items)
    assert not any("story points" in item.title.lower() for item in _backlog.items)


def test_story_without_needed_tasks_reports_missing_implementation() -> None:
    items = _base_items(split_intake=True)
    items = [item for item in items if not str(item["local_id"]).startswith("task-letter")]
    _backlog, findings = decompose_backlog(_envelope(), MockProvider(_proposal(items)), repair=False)
    missing = [item for item in findings.items if item.code is FindingCode.MISSING_IMPLEMENTATION_DETAIL]
    assert missing


def test_over_decomposition_and_duplicate_tasks() -> None:
    items = _base_items(split_intake=True)
    payload = municipal_permits_envelope()
    provenance = [_src("$.sow.in_scope[0]", "in_scope", payload["sow"]["in_scope"][0])]
    items.extend(
        _item(
            f"task-dup-{index}",
            "task",
            f"Save draft {index}",
            parent="story-apply",
            provenance=provenance,
            title_origin="generated",
        )
        for index in range(1, 5)
    )
    _backlog, findings = decompose_backlog(_envelope(), MockProvider(_proposal(items)))
    assert any(item.code is FindingCode.OVER_DECOMPOSITION for item in findings.items)

    items.append(
        _item(
            "task-dup-again",
            "task",
            "Save a draft application for the selected permit type",
            parent="story-apply",
            provenance=provenance,
            title_origin="generated",
        )
    )
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(_envelope(), MockProvider(_proposal(items)), repair=False)
    assert any("duplicate task title" in error for error in exc.value.errors)


def test_parent_restating_task_is_rejected() -> None:
    items = _base_items(split_intake=True)
    story = next(item for item in items if item["local_id"] == "story-calendar")
    items.append(
        _item(
            "task-copy",
            "task",
            story["title"],
            parent="story-calendar",
            provenance=story["provenance"],
            title_origin="source",
        )
    )
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(_envelope(), MockProvider(_proposal(items)), repair=False)
    assert any("restates the parent" in error for error in exc.value.errors)
