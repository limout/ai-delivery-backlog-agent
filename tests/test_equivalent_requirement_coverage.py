"""Equivalent source rows share a coverage result without combining different outcomes.

Clinic wording is fixture data. Production grouping uses statement keys only.
"""

from __future__ import annotations

import pytest

from backlog_agent.backlog.decompose import (
    _bind_exact_functional_citations,
    _try_materialize,
    decompose_backlog,
)
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import FindingCode
from backlog_agent.llm.exceptions import DecompositionValidationError
from backlog_agent.llm.mock import MockProvider
from backlog_agent.source import load_project_source
from tests.copilot_envelopes import _complete
from tests.test_decompose_generic import _item, _src

NOTE = "A nurse can record the visit note"
NOTE_PERIOD = "A nurse can record the visit note."
LAB = "A nurse can order a lab test."
COMPOUND = "A nurse can record the visit note and order a lab test."
TEMP = "Samples must remain within the documented temperature range"
TEMP_PERIOD = "Samples must remain within the documented temperature range."


def _envelope(
    *,
    functional: list[str],
    non_functional: list[str] | None = None,
) -> dict:
    return _complete(
        user_request="Nurses record visit notes and order lab tests for clinic visits.",
        executive_summary="Clinic staff record visit notes.",
        requirements={
            "functional_requirements": functional,
            "acceptance_criteria": ["The chart page opens for the signed-in clinician."],
            "non_functional_requirements": non_functional or [],
        },
        solution={"key_capabilities": ["Visit documentation."]},
        sow={
            "in_scope": ["Visit documentation"],
            "out_of_scope": ["Billing"],
            "deliverables": ["Visit note"],
        },
    )


def _tree(payload: dict, stories: list[dict], tasks: list[dict] | None = None) -> dict:
    items = [
        _item(
            "epic-1",
            "epic",
            "Clinic visits",
            parent=None,
            provenance=[_src("$.user_request", "user_request", payload["user_request"])],
            title_origin="generated",
        ),
        _item(
            "feat-1",
            "feature",
            "Visit documentation.",
            parent="epic-1",
            provenance=[
                _src("$.solution.key_capabilities[0]", "key_capabilities", "Visit documentation.")
            ],
            title_origin="generated",
        ),
        *stories,
        *(tasks or []),
    ]
    return {"items": items, "prerequisites": [], "uncertainties": []}


def _story(
    local_id: str,
    title: str,
    description: str,
    provenance: list[dict[str, str]],
    *,
    acceptance: str = "The chart page opens for the signed-in clinician.",
) -> dict:
    return _item(
        local_id,
        "user_story",
        title,
        parent="feat-1",
        provenance=provenance,
        description=description,
        title_origin="generated",
        ac=[acceptance],
        ac_origins=["generated"],
        tests=["Open the chart and confirm the page is shown."],
    )


def _fr(index: int, text: str) -> dict[str, str]:
    return _src(f"$.requirements.functional_requirements[{index}]", "functional_requirements", text)


def _nfr(index: int, text: str) -> dict[str, str]:
    return _src(
        f"$.requirements.non_functional_requirements[{index}]",
        "non_functional_requirements",
        text,
    )


def _functional_pairs(item) -> list[tuple[str, str]]:
    return [
        (reference.json_path, reference.excerpt or "")
        for reference in item.provenance
        if reference.json_path.startswith("$.requirements.functional_requirements")
    ]


def _uncovered_functional(findings) -> list[str]:
    return [
        reference.excerpt or ""
        for item in findings.items
        if item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT
        for reference in item.source_references
    ]


def test_missing_sibling_is_added_to_the_citing_story_only() -> None:
    payload = _envelope(functional=[NOTE, NOTE_PERIOD])
    story = _story(
        "story-record",
        "Record Visit Note",
        "A nurse records the visit note in the patient chart.",
        [_fr(0, NOTE)],
    )
    source = load_project_source(payload)
    result = _try_materialize(source, _tree(payload, [story]))

    assert result.ok
    stored = next(item for item in result.proposal.items if item.local_id == "story-record")
    assert _functional_pairs(stored) == [
        ("$.requirements.functional_requirements[0]", NOTE),
        ("$.requirements.functional_requirements[1]", NOTE_PERIOD),
    ]
    assert _uncovered_functional(result.findings) == []
    _bind_exact_functional_citations(source, result.proposal)
    assert _functional_pairs(stored) == [
        ("$.requirements.functional_requirements[0]", NOTE),
        ("$.requirements.functional_requirements[1]", NOTE_PERIOD),
    ]


def test_two_stories_keep_their_own_equivalent_citations() -> None:
    payload = _envelope(functional=[NOTE, NOTE_PERIOD])
    record = _story(
        "story-record",
        "Record Visit Note",
        "A nurse records the visit note in the patient chart.",
        [_fr(0, NOTE)],
    )
    document = _story(
        "story-document",
        "Document Visit Note",
        "A nurse documents the visit note for the encounter.",
        [_fr(1, NOTE_PERIOD)],
        acceptance="The documented note remains available on the chart.",
    )
    source = load_project_source(payload)
    proposal = _tree(payload, [record, document])
    result = _try_materialize(source, proposal)

    assert result.ok
    by_id = {item.local_id: item for item in result.proposal.items}
    assert _functional_pairs(by_id["story-record"]) == [
        ("$.requirements.functional_requirements[0]", NOTE)
    ]
    assert _functional_pairs(by_id["story-document"]) == [
        ("$.requirements.functional_requirements[1]", NOTE_PERIOD)
    ]
    assert {item.local_id for item in result.proposal.items} >= {"story-record", "story-document"}
    assert _uncovered_functional(result.findings) == []
    before = [_functional_pairs(item) for item in result.proposal.items]
    _bind_exact_functional_citations(source, result.proposal)
    assert [_functional_pairs(item) for item in result.proposal.items] == before


def test_clarification_task_does_not_cover_either_equivalent_row() -> None:
    payload = _envelope(functional=[NOTE, NOTE_PERIOD, LAB])
    lab = _story(
        "story-lab",
        "Order Lab Test",
        "A nurse orders the requested lab test for the visit.",
        [_fr(2, LAB)],
        acceptance="The ordered lab test is listed on the visit.",
    )
    task = _item(
        "task-note",
        "task",
        "Clarify the visit-note requirement",
        parent="story-lab",
        provenance=[_fr(0, NOTE)],
        description="This task asks for the missing note wording and does not implement it.",
        title_origin="generated",
    )
    provider = MockProvider(responses=[_tree(payload, [lab], [task])])
    backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload), provider, repair=False
    )

    story_paths = {
        reference.json_path
        for item in backlog.items
        if item.type.value == "user_story"
        for reference in item.provenance
    }
    assert "$.requirements.functional_requirements[0]" not in story_paths
    assert "$.requirements.functional_requirements[1]" not in story_paths
    uncovered = _uncovered_functional(findings)
    assert NOTE in uncovered
    assert NOTE_PERIOD in uncovered
    assert findings.llm_calls[0].accepted is False


def test_compound_sentence_does_not_cover_a_shorter_row() -> None:
    payload = _envelope(functional=[NOTE, NOTE_PERIOD])
    story = _story(
        "story-both",
        "Visit Documentation",
        COMPOUND,
        [_src("$.solution.key_capabilities[0]", "key_capabilities", "Visit documentation.")],
    )
    result = _try_materialize(load_project_source(payload), _tree(payload, [story]))

    assert result.ok
    uncovered = _uncovered_functional(result.findings)
    assert NOTE in uncovered
    assert NOTE_PERIOD in uncovered
    stored = next(item for item in result.proposal.items if item.local_id == "story-both")
    assert _functional_pairs(stored) == []


def test_different_outcomes_still_cannot_share_a_story() -> None:
    payload = _envelope(functional=[NOTE, LAB])
    story = _story(
        "story-both",
        "Visit Documentation",
        "A nurse records the visit note and orders a lab test.",
        [_fr(0, NOTE), _fr(1, LAB)],
    )
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(
            CopilotWorkflowResponseV1.model_validate(payload),
            MockProvider(responses=[_tree(payload, [story])]),
            repair=False,
        )
    assert any("combine independently" in error for error in exc.value.errors)


def test_equivalent_uncovered_constraints_are_one_finding() -> None:
    payload = _envelope(functional=[NOTE], non_functional=[TEMP, TEMP_PERIOD])
    story = _story(
        "story-record",
        "Record Visit Note",
        "A nurse records the visit note in the patient chart.",
        [_fr(0, NOTE), _nfr(0, TEMP)],
    )
    result = _try_materialize(load_project_source(payload), _tree(payload, [story]))

    assert result.ok
    findings = [
        item
        for item in result.findings.items
        if item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT
    ]
    assert len(findings) == 1
    assert findings[0].severity.value == "warning"
    assert [(reference.json_path, reference.excerpt) for reference in findings[0].source_references] == [
        ("$.requirements.non_functional_requirements[0]", TEMP),
        ("$.requirements.non_functional_requirements[1]", TEMP_PERIOD),
    ]
    assert _uncovered_functional(result.findings) == []


def test_functional_citation_does_not_cover_the_same_non_functional_sentence() -> None:
    payload = _envelope(functional=[NOTE], non_functional=[NOTE])
    story = _story(
        "story-record",
        "Record Visit Note",
        "A nurse records the visit note in the patient chart.",
        [_fr(0, NOTE)],
    )
    result = _try_materialize(load_project_source(payload), _tree(payload, [story]))

    assert result.ok
    assert _uncovered_functional(result.findings) == []
    non_functional = [
        item
        for item in result.findings.items
        if item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT
    ]
    assert len(non_functional) == 1
    assert non_functional[0].source_references[0].json_path == (
        "$.requirements.non_functional_requirements[0]"
    )
    assert non_functional[0].source_references[0].excerpt == NOTE
