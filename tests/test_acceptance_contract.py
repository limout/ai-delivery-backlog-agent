"""Acceptance contract: a candidate is accepted only without blocking quality defects.

Domain terms live in the shared envelopes only. Production code is tested through two
unrelated domains so no wording can be special-cased.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backlog_agent.backlog.decompose import _build_repair_prompt, decompose_backlog
from backlog_agent.backlog.granularity import (
    _investigation_subjects_overlap,
    _investigation_subject_tokens,
    item_represents_requirement,
)
from backlog_agent.backlog.models import WorkItemType
from backlog_agent.backlog.quality_gate import (
    blocking_quality_findings,
    is_blocking_quality_finding,
)
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import Finding, FindingCode, FindingSeverity
from backlog_agent.llm.mock import MockProvider
from tests.copilot_envelopes import clinic_rounds_envelope, field_inspection_envelope
from tests.test_decompose_generic import _item, _src

_DOMAINS = [
    pytest.param(field_inspection_envelope, id="field-inspection"),
    pytest.param(clinic_rounds_envelope, id="clinic-rounds"),
]


def _finding(code: FindingCode, message: str = "msg", severity=FindingSeverity.WARNING) -> Finding:
    return Finding(severity=severity, code=code, message=message)


def _proposal(
    payload: dict,
    *,
    verify_nfr: bool,
    nfr_only_story: bool = False,
    partial_first_story: bool = False,
    duplicate_tasks: bool = False,
) -> dict:
    """One feature and one story per functional requirement, built from the envelope only."""

    fr = payload["requirements"]["functional_requirements"]
    ac = payload["requirements"]["acceptance_criteria"]
    caps = payload["solution"]["key_capabilities"]
    nfr = payload["requirements"]["non_functional_requirements"][0]
    nfr_src = _src("$.requirements.non_functional_requirements[0]", "non_functional_requirements", nfr)
    items = [
        _item(
            "epic-1",
            "epic",
            "Delivery",
            parent=None,
            provenance=[_src("$.user_request", "user_request", payload["user_request"])],
            title_origin="generated",
        )
    ]
    last = len(fr) - 1
    for index, text in enumerate(fr):
        items.append(
            _item(
                f"feat-{index}",
                "feature",
                caps[index],
                parent="epic-1",
                provenance=[_src(f"$.solution.key_capabilities[{index}]", "key_capabilities", caps[index])],
                title_origin="generated",
            )
        )
        tests = [f"Verify: {text}"]
        description = f"An actor completes requirement {index + 1} and observes the result."
        provenance = [
            _src(f"$.requirements.functional_requirements[{index}]", "functional_requirements", text),
            _src(f"$.requirements.acceptance_criteria[{index}]", "acceptance_criteria", ac[index]),
        ]
        if partial_first_story and index == 0:
            tests = ["Open the screen and assert it renders."]
        if index == last:
            provenance.append(nfr_src)
            if verify_nfr:
                tests.append(f"Verify: {nfr}")
            else:
                # Mentioned in the description and cited as provenance, but never verified.
                description += f" {nfr}"
        items.append(
            _item(
                f"story-{index}",
                "user_story",
                text,
                parent=f"feat-{index}",
                provenance=provenance,
                description=description,
                ac=[ac[index]],
                ac_origins=["source"],
                tests=tests,
            )
        )
    if nfr_only_story:
        items.append(
            _item(
                "story-nfr-only",
                "user_story",
                nfr,
                parent=f"feat-{last}",
                provenance=[nfr_src],
                title_origin="source",
            )
        )
    if duplicate_tasks:
        questions = payload["requirements"]["open_questions"]
        for index, question in enumerate(questions):
            items.append(
                _item(
                    f"task-clarify-{index}",
                    "task",
                    f"Clarify remaining implementation detail: {question}",
                    parent="story-0",
                    provenance=[_src(f"$.requirements.open_questions[{index}]", "open_questions", question)],
                    title_origin="generated",
                )
            )
    return {"items": items, "prerequisites": [], "uncertainties": []}


def _payload(factory) -> dict:
    return factory()


def _env(payload: dict) -> CopilotWorkflowResponseV1:
    return CopilotWorkflowResponseV1.model_validate(payload)


def _kinds(findings) -> list[str]:
    return [call.kind for call in findings.llm_calls]


def _blocking(findings) -> list[Finding]:
    return blocking_quality_findings(findings.items)


# -- the contract definition ---------------------------------------------------------------


def test_blocking_and_advisory_findings_are_distinguished() -> None:
    blocking = [
        _finding(FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT),
        _finding(FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT),
        _finding(FindingCode.DUPLICATE_ACCEPTANCE_CRITERION),
        _finding(FindingCode.OVERLAPPING_STORY_RESPONSIBILITY),
        _finding(
            FindingCode.MISSING_VERIFICATION_COVERAGE,
            "User story 'x' does not verify every outcome in its cited source requirement",
        ),
    ]
    advisory = [
        _finding(FindingCode.FEATURE_CHILD_MISALIGNMENT),
        _finding(FindingCode.AMBIGUOUS_DECOMPOSITION),
        _finding(FindingCode.OVER_DECOMPOSITION),
        _finding(FindingCode.OVERSIZED_STORY),
        _finding(FindingCode.MISSING_IMPLEMENTATION_DETAIL),
        _finding(FindingCode.UNRESOLVED_PREREQUISITE),
        _finding(FindingCode.UNCOVERED_REQUEST_OUTCOME),
        _finding(FindingCode.PLANNING_CONSTRAINT, severity=FindingSeverity.INFO),
        _finding(
            FindingCode.MISSING_VERIFICATION_COVERAGE,
            "User story has neither acceptance criteria nor test requirements.",
        ),
    ]
    assert all(is_blocking_quality_finding(item) for item in blocking)
    assert not any(is_blocking_quality_finding(item) for item in advisory)
    assert blocking_quality_findings([*advisory, *blocking]) == blocking


# -- hard repair ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", _DOMAINS)
def test_hard_repair_that_leaves_an_uncovered_constraint_is_not_accepted(factory) -> None:
    payload = _payload(factory)
    invalid = _proposal(payload, verify_nfr=False, nfr_only_story=True)
    repaired = _proposal(payload, verify_nfr=False)
    provider = MockProvider(responses=[invalid, repaired])
    backlog, findings = decompose_backlog(_env(payload), provider)

    assert _kinds(findings) == ["initial", "hard_repair"]
    initial, hard = findings.llm_calls
    assert initial.validation_passed is False and initial.accepted is False
    # Structurally valid, so it is returned. Not accepted, because the constraint is unresolved.
    assert hard.validation_passed is True
    assert hard.accepted is False
    assert hard.blocking_quality_findings >= 1
    assert backlog.items
    assert any(
        item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT for item in findings.items
    )
    rejected = [item for item in findings.items if item.code is FindingCode.QUALITY_REPAIR_REJECTED]
    assert len(rejected) == 1
    assert "UNCOVERED_NON_FUNCTIONAL_REQUIREMENT" in rejected[0].message
    assert "not accepted" in rejected[0].message


@pytest.mark.parametrize("factory", _DOMAINS)
def test_hard_repair_that_verifies_the_constraint_is_accepted(factory) -> None:
    payload = _payload(factory)
    invalid = _proposal(payload, verify_nfr=False, nfr_only_story=True)
    repaired = _proposal(payload, verify_nfr=True)
    provider = MockProvider(responses=[invalid, repaired])
    _backlog, findings = decompose_backlog(_env(payload), provider)

    assert _kinds(findings) == ["initial", "hard_repair"]
    assert findings.llm_calls[1].accepted is True
    assert findings.llm_calls[1].blocking_quality_findings == 0
    assert _blocking(findings) == []
    assert not any(item.code is FindingCode.QUALITY_REPAIR_REJECTED for item in findings.items)


def test_hard_repair_prompt_forbids_dropping_the_constraint_without_verification() -> None:
    prompt = _build_repair_prompt("BASE", {"items": []}, ["story-x: nfr-only story"]).lower()
    assert "do not fix that error by deleting the behavior" in prompt
    assert "acceptance criterion or test requirement" in prompt


# -- quality repair ------------------------------------------------------------------------


@pytest.mark.parametrize("factory", _DOMAINS)
def test_quality_repair_that_resolves_the_defect_is_accepted(factory) -> None:
    payload = _payload(factory)
    initial = _proposal(payload, verify_nfr=False)
    repaired = _proposal(payload, verify_nfr=True)
    provider = MockProvider(responses=[initial, repaired])
    _backlog, findings = decompose_backlog(_env(payload), provider)

    assert _kinds(findings) == ["initial", "quality_repair"]
    assert [call.accepted for call in findings.llm_calls] == [False, True]
    assert findings.llm_calls[0].blocking_quality_findings >= 1
    assert findings.llm_calls[1].blocking_quality_findings == 0
    assert _blocking(findings) == []


@pytest.mark.parametrize("factory", _DOMAINS)
def test_unchanged_repair_leaves_both_candidates_unaccepted(factory) -> None:
    payload = _payload(factory)
    provider = MockProvider(
        responses=[_proposal(payload, verify_nfr=False), _proposal(payload, verify_nfr=False)]
    )
    _backlog, findings = decompose_backlog(_env(payload), provider)

    assert [call.accepted for call in findings.llm_calls] == [False, False]
    assert all(call.validation_passed for call in findings.llm_calls)
    assert _blocking(findings)
    assert any(item.code is FindingCode.QUALITY_REPAIR_REJECTED for item in findings.items)


def test_partially_improved_repair_is_returned_but_not_accepted() -> None:
    payload = _payload(field_inspection_envelope)
    initial = _proposal(payload, verify_nfr=False, partial_first_story=True)
    improved = _proposal(payload, verify_nfr=True, partial_first_story=True)
    provider = MockProvider(responses=[initial, improved])
    backlog, findings = decompose_backlog(_env(payload), provider)

    before, after = (call.blocking_quality_findings for call in findings.llm_calls)
    assert after < before
    assert [call.accepted for call in findings.llm_calls] == [False, False]
    # The improved tree is the one returned: it carries the verified constraint.
    story_text = " ".join(
        " ".join(item.test_requirements)
        for item in backlog.items
        if item.type is WorkItemType.USER_STORY
    ).lower()
    assert "unreliable" in story_text
    assert any(
        item.code is FindingCode.MISSING_VERIFICATION_COVERAGE for item in findings.items
    )
    rejected = [item for item in findings.items if item.code is FindingCode.QUALITY_REPAIR_REJECTED]
    assert len(rejected) == 1
    assert "resolved some defects" in rejected[0].message


# -- initial candidate and advisory findings -----------------------------------------------


@pytest.mark.parametrize("factory", _DOMAINS)
def test_initial_candidate_with_blocking_defect_is_not_accepted_without_repair(factory) -> None:
    payload = _payload(factory)
    _backlog, findings = decompose_backlog(
        _env(payload), MockProvider(_proposal(payload, verify_nfr=False)), repair=False
    )
    assert _kinds(findings) == ["initial"]
    assert findings.llm_calls[0].validation_passed is True
    assert findings.llm_calls[0].accepted is False


def test_repair_budget_is_never_exceeded_by_the_gate() -> None:
    payload = _payload(field_inspection_envelope)
    provider = MockProvider(
        responses=[
            _proposal(payload, verify_nfr=False, nfr_only_story=True),
            _proposal(payload, verify_nfr=False),
            _proposal(payload, verify_nfr=True),
        ]
    )
    _backlog, findings = decompose_backlog(_env(payload), provider)
    assert len(provider.calls) == 2
    assert _kinds(findings) == ["initial", "hard_repair"]
    assert findings.llm_calls[1].accepted is False


# -- the verification requirement ------------------------------------------------------------


def test_description_mention_does_not_verify_a_constraint() -> None:
    nfr = field_inspection_envelope()["requirements"]["non_functional_requirements"][0]
    story = SimpleNamespace(
        title="Complete an inspection",
        description=f"A technician completes an inspection. {nfr}",
        acceptance_criteria=["A technician can complete an inspection."],
        test_requirements=[],
    )
    assert item_represents_requirement(story, nfr) is True
    assert item_represents_requirement(story, nfr, verification_only=True) is False

    verified = SimpleNamespace(
        title="Complete an inspection",
        description="A technician completes an inspection.",
        acceptance_criteria=["A technician can complete an inspection."],
        test_requirements=[f"Verify: {nfr}"],
    )
    assert item_represents_requirement(verified, nfr, verification_only=True) is True


@pytest.mark.parametrize("factory", _DOMAINS)
def test_description_only_constraint_is_reported_uncovered(factory) -> None:
    payload = _payload(factory)
    _backlog, findings = decompose_backlog(
        _env(payload), MockProvider(_proposal(payload, verify_nfr=False)), repair=False
    )
    assert any(
        item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT for item in findings.items
    )


# -- duplicate and misplaced clarification tasks -------------------------------------------


def _task(canonical_id: str, title: str, path: str, excerpt: str):
    return SimpleNamespace(
        canonical_id=canonical_id,
        type=WorkItemType.TASK,
        title=title,
        description="Resolve this remaining source detail for the parent story.",
        acceptance_criteria=[],
        test_requirements=[],
        provenance=[SimpleNamespace(json_path=path, excerpt=excerpt, field_name="x")],
    )


def test_same_question_from_different_entries_is_an_overlap() -> None:
    left = _task(
        "a",
        "Clarify remaining implementation detail: Exact offline behavior and synchronization rules",
        "$.discovery.unknowns[0]",
        "Exact offline behavior and synchronization rules for unreliable mobile connectivity",
    )
    right = _task(
        "b",
        "Clarify remaining implementation detail: What are the offline behavior requirements?",
        "$.requirements.open_questions[1]",
        "What are the specific requirements for offline behavior and data synchronization "
        "when technicians work at remote sites with unreliable connectivity?",
    )
    assert _investigation_subjects_overlap(
        _investigation_subject_tokens(left), _investigation_subject_tokens(right)
    )


def test_different_questions_sharing_a_word_are_not_overlapping() -> None:
    left = _task("a", "Clarify", "$.requirements.open_questions[0]", "Which hosting provider is required?")
    right = _task("b", "Clarify", "$.requirements.open_questions[1]", "Which payment provider is required?")
    assert not _investigation_subjects_overlap(
        _investigation_subject_tokens(left), _investigation_subject_tokens(right)
    )


def test_boilerplate_wording_does_not_drive_parent_placement() -> None:
    task = _task(
        "a",
        "Clarify remaining implementation detail: Detailed inspection checklist requirements",
        "$.requirements.open_questions[4]",
        "Detailed inspection checklist requirements",
    )
    tokens = _investigation_subject_tokens(task)
    assert "clarify" not in tokens and "remaining" not in tokens
    assert {"checklist", "inspection"} <= tokens


@pytest.mark.parametrize("factory", _DOMAINS)
def test_distinct_questions_are_not_flagged_as_overlapping(factory) -> None:
    payload = _payload(factory)
    payload["requirements"]["open_questions"] = [
        payload["requirements"]["open_questions"][0],
        "Which identity provider issues the sign-in tokens?",
    ]
    proposal = _proposal(payload, verify_nfr=True, duplicate_tasks=True)
    _backlog, findings = decompose_backlog(_env(payload), MockProvider(proposal), repair=False)
    assert not any("Overlapping investigation tasks " in item.message for item in findings.items)
