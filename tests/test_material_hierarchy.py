"""Material versus advisory hierarchy findings and their effect on acceptance.

The domain here (greenhouse management) is invented test data. Production code must not
mention it. Every scenario is built from the same neutral envelope so the only variable is the
shape of the proposed hierarchy.
"""

from __future__ import annotations

import copy

from backlog_agent.backlog.decompose import decompose_backlog
from backlog_agent.backlog.models import WorkItemType
from backlog_agent.backlog.quality_gate import (
    MATERIAL_HIERARCHY_PREFIX,
    blocking_quality_findings,
    is_blocking_quality_finding,
    is_material_hierarchy_finding,
    material_hierarchy_message,
)
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import Finding, FindingCode, FindingSeverity
from backlog_agent.llm.mock import MockProvider
from tests.copilot_envelopes import _complete
from tests.test_decompose_generic import _item, _src

# Each outcome has its own actor so a shared role word never links unrelated stories.
_FR = [
    "The system shall let a grower schedule watering cycles for a zone.",
    "The system shall let an agronomist view soil moisture readings for a zone.",
    "The system shall let an owner export a harvest summary.",
    "The system shall let an administrator invite a co-worker to a greenhouse.",
]
_AC = [
    "A grower can schedule a watering cycle for a zone.",
    "An agronomist can view the latest soil moisture reading from each sensor.",
    "An owner can export a harvest summary.",
    "An administrator can invite a co-worker to a greenhouse.",
]
_CAPS = ["Watering schedules", "Moisture monitoring", "Harvest reporting", "Greenhouse access"]
# Distinct vocabulary per story so shared filler words never make unrelated stories look related.
_DESCRIPTIONS = [
    "A grower picks days and times for irrigation to run in a zone.",
    "An agronomist reads the newest soil moisture measurement from each sensor in a zone.",
    "An owner downloads a spreadsheet summarising the crop yield.",
    "An administrator sends an invitation so a colleague can enter the greenhouse workspace.",
]
_QUESTION = "Which sensor calibration interval is required?"
_QUESTION_REPEATED = "What sensor calibration interval should apply?"


def _payload(*, questions: list[str] | None = None) -> dict:
    return _complete(
        user_request=(
            "Build a greenhouse management tool so growers can schedule watering, view soil "
            "moisture, export harvest summaries, and invite co-workers."
        ),
        executive_summary="Growers manage greenhouses from one tool.",
        requirements={
            "functional_requirements": list(_FR),
            "acceptance_criteria": list(_AC),
            "non_functional_requirements": [],
            "open_questions": list(questions if questions is not None else [_QUESTION]),
        },
        solution={"key_capabilities": list(_CAPS)},
        sow={
            "in_scope": ["Watering schedules", "Moisture monitoring", "Harvest reports"],
            "out_of_scope": ["Billing"],
            "deliverables": ["Greenhouse management tool"],
        },
    )


def _env(payload: dict) -> CopilotWorkflowResponseV1:
    return CopilotWorkflowResponseV1.model_validate(payload)


def _epic(payload: dict) -> dict:
    return _item(
        "epic-1",
        "epic",
        "Greenhouse management",
        parent=None,
        provenance=[_src("$.user_request", "user_request", payload["user_request"])],
        title_origin="generated",
    )


def _story(index: int, parent: str, *, description: str | None = None, extra_tests=()) -> dict:
    # Story 1 (moisture) owns the sensor-calibration subject in its own verification text.
    tests = [f"Verify: {_FR[index]}", *extra_tests]
    if index == 1:
        tests.append("Verify readings reflect the sensor calibration.")
    return _item(
        f"story-{index}",
        "user_story",
        _FR[index],
        parent=parent,
        provenance=[
            _src(f"$.requirements.functional_requirements[{index}]", "functional_requirements", _FR[index]),
            _src(f"$.requirements.acceptance_criteria[{index}]", "acceptance_criteria", _AC[index]),
        ],
        description=description or _DESCRIPTIONS[index],
        ac=[_AC[index]],
        ac_origins=["source"],
        tests=tests,
    )


def _clarify(
    local_id: str,
    parent: str,
    question_index: int,
    question: str,
    *,
    verb: str = "Clarify remaining implementation detail",
) -> dict:
    return _item(
        local_id,
        "task",
        f"{verb}: {question}",
        parent=parent,
        provenance=[
            _src(f"$.requirements.open_questions[{question_index}]", "open_questions", question)
        ],
        title_origin="generated",
    )


def _feature(local_id: str, title: str, cap_index: int) -> dict:
    return _item(
        local_id,
        "feature",
        title,
        parent="epic-1",
        provenance=[_src(f"$.solution.key_capabilities[{cap_index}]", "key_capabilities", _CAPS[cap_index])],
        title_origin="generated",
    )


def _grouped(payload: dict, *, tasks: list[dict] | None = None, moisture_task_parent="story-1") -> dict:
    """Each capability under a feature that describes it, plus one correctly placed task."""

    items = [
        _epic(payload),
        *[_feature(f"feat-{index}", _CAPS[index], index) for index in range(4)],
        *[_story(index, f"feat-{index}") for index in range(4)],
        *(tasks if tasks is not None else [_clarify("task-calibration", moisture_task_parent, 0, _QUESTION)]),
    ]
    return {"items": items, "prerequisites": [], "uncertainties": []}


def _catch_all(payload: dict, *, tasks: list[dict] | None = None) -> dict:
    """Four unrelated outcomes under one feature whose own title covers none of them."""

    items = [
        _epic(payload),
        _feature("feat-all", _CAPS[0], 0),
        *[_story(index, "feat-all") for index in range(4)],
        *(tasks if tasks is not None else [_clarify("task-calibration", "story-1", 0, _QUESTION)]),
    ]
    # Re-title the feature so it names no child capability. Provenance stays on a real source.
    items[1]["title"] = "Operations toolkit"
    return {"items": items, "prerequisites": [], "uncertainties": []}


def _weak_grouping(payload: dict) -> dict:
    """Three stories, two clusters: a weak signal, not a clearly unrelated collection."""

    items = [
        _epic(payload),
        _feature("feat-a", "Greenhouse tool", 0),
        _feature("feat-b", _CAPS[3], 3),
        *[_story(index, "feat-a") for index in range(3)],
        _story(3, "feat-b"),
        _clarify("task-calibration", "story-1", 0, _QUESTION),
    ]
    return {"items": items, "prerequisites": [], "uncertainties": []}


def _run(payload: dict, *responses: dict, repair: bool = True):
    provider = MockProvider(responses=list(responses)) if len(responses) > 1 else MockProvider(responses[0])
    backlog, findings = decompose_backlog(_env(payload), provider, repair=repair)
    return backlog, findings, provider


def _material(findings, code: FindingCode | None = None) -> list[Finding]:
    return [
        item
        for item in findings.items
        if is_material_hierarchy_finding(item) and (code is None or item.code is code)
    ]


def _advisory(findings, code: FindingCode) -> list[Finding]:
    return [
        item
        for item in findings.items
        if item.code is code and not is_material_hierarchy_finding(item)
    ]


def _accepted(findings) -> list[bool]:
    return [call.accepted for call in findings.llm_calls]


# -- the marker and the predicate ------------------------------------------------------------


def test_marker_alone_decides_material_hierarchy_status_and_only_for_hierarchy_codes() -> None:
    marked = Finding(
        severity=FindingSeverity.WARNING,
        code=FindingCode.FEATURE_CHILD_MISALIGNMENT,
        message=material_hierarchy_message("grouping"),
    )
    unmarked = Finding(
        severity=FindingSeverity.WARNING,
        code=FindingCode.FEATURE_CHILD_MISALIGNMENT,
        message="grouping",
    )
    wrong_code = Finding(
        severity=FindingSeverity.WARNING,
        code=FindingCode.OVERSIZED_STORY,
        message=MATERIAL_HIERARCHY_PREFIX + "x",
    )
    assert is_blocking_quality_finding(marked) and is_material_hierarchy_finding(marked)
    assert not is_blocking_quality_finding(unmarked)
    assert not is_blocking_quality_finding(wrong_code)
    assert blocking_quality_findings([unmarked, marked, wrong_code]) == [marked]


# -- material feature grouping ---------------------------------------------------------------


def test_material_feature_defect_remaining_after_repair_prevents_acceptance() -> None:
    payload = _payload()
    _backlog, findings, provider = _run(payload, _catch_all(payload), _catch_all(payload))

    assert [call.kind for call in findings.llm_calls] == ["initial", "quality_repair"]
    assert len(provider.calls) == 2  # one repair, never a loop
    assert _accepted(findings) == [False, False]
    assert all(call.validation_passed for call in findings.llm_calls)
    assert _material(findings, FindingCode.FEATURE_CHILD_MISALIGNMENT)
    assert "Material hierarchy defect" in provider.calls[1][0]
    rejected = [item for item in findings.items if item.code is FindingCode.QUALITY_REPAIR_REJECTED]
    assert len(rejected) == 1 and "FEATURE_CHILD_MISALIGNMENT" in rejected[0].message


def test_material_feature_defect_fixed_by_the_repair_is_accepted() -> None:
    payload = _payload()
    backlog, findings, _provider = _run(payload, _catch_all(payload), _grouped(payload))

    assert _accepted(findings) == [False, True]
    assert blocking_quality_findings(findings.items) == []
    assert sum(1 for item in backlog.items if item.type is WorkItemType.FEATURE) == 4


def test_weak_grouping_signal_stays_advisory_and_does_not_block() -> None:
    payload = _payload()
    _backlog, findings, _provider = _run(payload, _weak_grouping(payload), repair=False)

    assert _advisory(findings, FindingCode.FEATURE_CHILD_MISALIGNMENT)
    assert _material(findings) == []
    assert blocking_quality_findings(findings.items) == []
    assert _accepted(findings) == [True]


def test_weak_grouping_signal_unchanged_by_repair_is_still_accepted() -> None:
    payload = _payload()
    _backlog, findings, provider = _run(payload, _weak_grouping(payload), _weak_grouping(payload))

    assert len(provider.calls) == 2
    assert _material(findings) == []
    assert _accepted(findings)[-1] is True
    assert blocking_quality_findings(findings.items) == []


# -- clarification task parenting ------------------------------------------------------------


def test_clearly_misparented_clarification_task_after_repair_prevents_acceptance() -> None:
    payload = _payload()
    bad = _grouped(payload, moisture_task_parent="story-2")  # export story shares nothing
    _backlog, findings, provider = _run(payload, bad, copy.deepcopy(bad))

    assert len(provider.calls) == 2
    misplaced = _material(findings, FindingCode.AMBIGUOUS_DECOMPOSITION)
    assert misplaced and "shares no part of its unresolved subject" in misplaced[0].message
    assert _accepted(findings) == [False, False]


def test_misparented_clarification_task_moved_by_the_repair_is_accepted() -> None:
    payload = _payload()
    bad = _grouped(payload, moisture_task_parent="story-2")
    _backlog, findings, _provider = _run(payload, bad, _grouped(payload))
    assert _accepted(findings) == [False, True]
    assert _material(findings) == []


def test_uncertain_parent_assignment_is_advisory() -> None:
    payload = _payload()
    proposal = _grouped(payload, moisture_task_parent="story-0")
    # The schedule story mentions the interval, so the parent is related, only a weaker fit.
    schedule = next(item for item in proposal["items"] if item["local_id"] == "story-0")
    schedule["test_requirements"] = [*schedule["test_requirements"], "Verify the fixed interval."]
    schedule["test_requirements_origins"] = ["generated"] * len(schedule["test_requirements"])
    _backlog, findings, _provider = _run(payload, proposal, repair=False)

    uncertain = _advisory(findings, FindingCode.AMBIGUOUS_DECOMPOSITION)
    assert uncertain and uncertain[0].message.startswith("Investigation task ")
    assert _material(findings) == []
    assert _accepted(findings) == [True]


# -- duplicate clarification tasks -----------------------------------------------------------


def _duplicated(payload: dict) -> dict:
    return _grouped(
        payload,
        tasks=[
            _clarify("task-a", "story-1", 0, _QUESTION),
            _clarify("task-b", "story-1", 1, _QUESTION_REPEATED),
        ],
    )


def test_duplicate_clarification_remaining_after_repair_prevents_acceptance() -> None:
    payload = _payload(questions=[_QUESTION, _QUESTION_REPEATED])
    _backlog, findings, provider = _run(payload, _duplicated(payload), _duplicated(payload))

    duplicates = _material(findings, FindingCode.OVER_DECOMPOSITION)
    assert duplicates and "Overlapping investigation tasks" in duplicates[0].message
    assert len(provider.calls) == 2
    assert _accepted(findings) == [False, False]


def test_duplicate_clarification_consolidated_by_the_repair_is_accepted() -> None:
    payload = _payload(questions=[_QUESTION, _QUESTION_REPEATED])
    consolidated = _grouped(
        payload, tasks=[_clarify("task-a", "story-1", 0, _QUESTION)]
    )
    backlog, findings, _provider = _run(payload, _duplicated(payload), consolidated)

    assert _accepted(findings) == [False, True]
    assert sum(1 for item in backlog.items if item.type is WorkItemType.TASK) == 1
    assert _material(findings) == []


def test_two_tasks_citing_one_source_question_are_a_material_duplicate() -> None:
    payload = _payload()
    same_source = _grouped(
        payload,
        tasks=[
            _clarify("task-a", "story-1", 0, _QUESTION),
            _clarify("task-b", "story-1", 0, _QUESTION, verb="Investigate unresolved detail"),
        ],
    )
    _backlog, findings, _provider = _run(payload, same_source, repair=False)
    assert any("cite the same source unknown" in item.message for item in _material(findings))
    assert _accepted(findings) == [False]


# -- legitimate unresolved questions ---------------------------------------------------------


def test_one_relevant_clarification_task_neither_blocks_nor_triggers_repair() -> None:
    payload = _payload()
    # A single scripted response: any repair call would raise "no scripted response".
    backlog, findings, provider = _run(payload, _grouped(payload))

    assert len(provider.calls) == 1
    assert [call.kind for call in findings.llm_calls] == ["initial"]
    assert _accepted(findings) == [True]
    assert blocking_quality_findings(findings.items) == []
    assert [item.title for item in backlog.items if item.type is WorkItemType.TASK] == [
        f"Clarify remaining implementation detail: {_QUESTION}"
    ]
    # The unresolved behavior is not pretended to be implemented: no story claims it.
    assert not any(
        "calibration interval" in " ".join([item.title, item.description])
        for item in backlog.items
        if item.type is WorkItemType.USER_STORY
    )


def test_unresolved_material_defect_never_creates_a_repair_loop() -> None:
    payload = _payload()
    still_bad = _catch_all(payload)
    provider = MockProvider(responses=[still_bad, copy.deepcopy(still_bad), _grouped(payload)])
    _backlog, findings = decompose_backlog(_env(payload), provider)
    assert len(provider.calls) == 2
    assert [call.kind for call in findings.llm_calls] == ["initial", "quality_repair"]
    assert _accepted(findings) == [False, False]


def test_hard_repaired_candidate_with_a_material_defect_is_not_accepted() -> None:
    payload = _payload()
    invalid = _grouped(payload)
    invalid["items"][5]["parent_local_id"] = "feat-missing"  # structural error
    _backlog, findings, provider = _run(payload, invalid, _catch_all(payload))

    assert [call.kind for call in findings.llm_calls] == ["initial", "hard_repair"]
    assert len(provider.calls) == 2
    assert findings.llm_calls[1].validation_passed is True
    assert _accepted(findings) == [False, False]
    assert _material(findings, FindingCode.FEATURE_CHILD_MISALIGNMENT)
