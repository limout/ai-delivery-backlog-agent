from __future__ import annotations

import json
from pathlib import Path

import pytest

from backlog_agent.backlog.decompose import decompose_backlog
from backlog_agent.backlog.models import WorkItemType
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import FindingCode
from backlog_agent.llm.exceptions import DecompositionValidationError
from backlog_agent.llm.mock import MockProvider
from backlog_agent.source import load_project_source
from tests.copilot_envelopes import library_holds_envelope
from tests.test_decompose_generic import _item, _library_proposal, _src

_INPUT = Path(__file__).resolve().parents[1] / "input" / "copilot-complete.json"


def _library_with_sections() -> tuple[dict, dict]:
    payload = library_holds_envelope()
    payload["user_request"] += (
        "\n\n### FR-04: Renewal Eligibility\n"
        "An ineligible policy cannot proceed through the renewal journey.\n"
        "\n### NFR-04: Performance and Availability\n"
        "Performance targets have not been defined.\n"
    )
    proposal = _library_proposal()
    epic = next(item for item in proposal["items"] if item["local_id"] == "epic-1")
    epic["provenance"][0]["excerpt"] = payload["user_request"]
    return payload, proposal


def test_exact_narrative_excerpt_is_kept_and_paraphrase_is_generated() -> None:
    payload, proposal = _library_with_sections()
    source = load_project_source(CopilotWorkflowResponseV1.model_validate(payload))
    section = next(item for item in source.narrative_sections if item.label == "FR-04")
    search = next(item for item in proposal["items"] if item["local_id"] == "story-search")
    search["title"] = "Renewal Eligibility"
    search["title_origin"] = "source"
    search["provenance"].append(
        _src("$.user_request", "user_request", section.excerpt)
    )
    backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload),
        MockProvider(proposal),
        repair=False,
    )
    stored = next(
        item
        for item in backlog.items
        if item.type is WorkItemType.USER_STORY and "eligibility" in item.title.casefold()
    )
    assert any(reference.excerpt == section.excerpt for reference in stored.provenance)
    assert all(reference.excerpt != payload["user_request"] or reference.json_path != "$.user_request" for reference in stored.provenance)
    assert any(
        item.code is FindingCode.GENERATED_CONTENT
        and stored.canonical_id in item.canonical_ids
        and "title" in item.message
        for item in findings.items
    )
    assert not any(
        item.code is FindingCode.UNCOVERED_REQUEST_OUTCOME and " FR-04:" in item.message
        for item in findings.items
    )


def test_unsupported_title_still_fails_against_a_narrative_excerpt() -> None:
    payload, proposal = _library_with_sections()
    source = load_project_source(CopilotWorkflowResponseV1.model_validate(payload))
    section = next(item for item in source.narrative_sections if item.label == "FR-04")
    search = next(item for item in proposal["items"] if item["local_id"] == "story-search")
    search["title"] = "Quarterly billing export dashboard"
    search["title_origin"] = "source"
    search["provenance"].append(_src("$.user_request", "user_request", section.excerpt))
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(
            CopilotWorkflowResponseV1.model_validate(payload),
            MockProvider(proposal),
            repair=False,
        )
    assert any("title_origin=source" in error for error in exc.value.errors)


def test_narrative_excerpt_does_not_cover_a_structured_requirement() -> None:
    payload, proposal = _library_with_sections()
    source = load_project_source(CopilotWorkflowResponseV1.model_validate(payload))
    section = next(item for item in source.narrative_sections if item.label == "FR-04")
    proposal["items"] = [item for item in proposal["items"] if item["local_id"] != "story-notify"]
    proposal["items"].append(
        _item(
            "story-eligibility",
            "user_story",
            "Renewal Eligibility",
            parent="feat-hold",
            provenance=[_src("$.user_request", "user_request", section.excerpt)],
            description="A member is stopped when the policy is not eligible to continue.",
            title_origin="generated",
        )
    )
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(
            CopilotWorkflowResponseV1.model_validate(payload),
            MockProvider(proposal),
            repair=False,
        )
    assert any("notify the member" in error or "functional" in error.lower() for error in exc.value.errors)
    assert any(
        item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT for item in exc.value.findings.items
    )


def test_unconfirmed_section_stays_unresolved_until_cited() -> None:
    payload, proposal = _library_with_sections()
    envelope = CopilotWorkflowResponseV1.model_validate(payload)
    _backlog, findings = decompose_backlog(envelope, MockProvider(proposal), repair=False)
    assert any(
        item.code is FindingCode.UNCOVERED_REQUEST_OUTCOME
        and "NFR-04" in item.message
        and "unconfirmed" in item.message
        for item in findings.items
    )

    source = load_project_source(envelope)
    section = next(item for item in source.narrative_sections if item.label == "NFR-04")
    cited = _library_with_sections()[1]
    hold = next(item for item in cited["items"] if item["local_id"] == "story-hold")
    hold["uncertainties"] = ["Performance and availability targets are unconfirmed."]
    _backlog, cited_findings = decompose_backlog(envelope, MockProvider(cited), repair=False)
    assert not any(
        item.code is FindingCode.UNCOVERED_REQUEST_OUTCOME and "NFR-04" in item.message
        for item in cited_findings.items
    )
    assert section.excerpt not in {fact.text for fact in source.functional_requirements}


def test_supplied_fixture_separates_narrative_evidence_from_structured_requirements() -> None:
    payload = json.loads(_INPUT.read_text(encoding="utf-8"))
    envelope = CopilotWorkflowResponseV1.model_validate(payload)
    source = load_project_source(envelope)
    by_label = {section.label: section for section in source.narrative_sections if section.label}
    assert set(by_label) >= {f"FR-{index:02d}" for index in range(1, 11)}
    assert set(by_label) >= {f"NFR-{index:02d}" for index in range(1, 8)}
    for section in by_label.values():
        assert section.json_path == "$.user_request"
        assert section.excerpt in payload["user_request"]
        assert section.excerpt != payload["user_request"]
        assert section.excerpt not in {fact.text for fact in source.functional_requirements}
    assert {section.evidence_kind for section in by_label.values()} >= {"outcome", "constraint", "unconfirmed"}
    assert source.prompt_facts["request"]["full_text_omitted_from_prompt"] is True
    assert "narrative_sections" in source.prompt_facts
    assert len(source.functional_requirements) == 6

    outcomes = [section for section in by_label.values() if section.evidence_kind == "outcome"]
    unconfirmed = [section for section in by_label.values() if section.evidence_kind == "unconfirmed"]
    assert {section.label for section in outcomes} >= {
        "FR-01",
        "FR-02",
        "FR-03",
        "FR-04",
        "FR-05",
        "FR-06",
        "FR-07",
        "FR-08",
        "FR-09",
        "FR-10",
    }
    assert unconfirmed
    assert "NFR-04" in {section.label for section in unconfirmed}

    proposal = _fixture_proposal(payload, source)
    backlog, findings = decompose_backlog(envelope, MockProvider(proposal), repair=False)
    covered = {
        reference.json_path
        for item in backlog.items
        if item.type is WorkItemType.USER_STORY
        for reference in item.provenance
        if reference.json_path.startswith("$.requirements.functional_requirements")
    }
    assert covered == {fact.json_path for fact in source.functional_requirements}
    messages = [
        item.message
        for item in findings.items
        if item.code is FindingCode.UNCOVERED_REQUEST_OUTCOME
    ]
    for section in outcomes:
        assert not any(f" {section.label}:" in message for message in messages)
    assert any("NFR-04" in message and "unconfirmed" in message for message in messages)
    cited = {
        reference.excerpt
        for item in backlog.items
        for reference in item.provenance
        if reference.json_path == "$.user_request"
    }
    assert all(section.excerpt in cited for section in outcomes)
    assert not any(section.excerpt in cited for section in unconfirmed)


def _fixture_proposal(payload: dict, source) -> dict:
    caps = payload["solution"]["key_capabilities"]
    items = [
        _item(
            "epic-1",
            "epic",
            "Insurance policy renewal portal",
            parent=None,
            provenance=[_src("$.user_request", "user_request", source.narrative_sections[0].excerpt)],
            title_origin="generated",
        ),
        _item(
            "feat-review",
            "feature",
            caps[1],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[1]", "key_capabilities", caps[1])],
        ),
        _item(
            "feat-auth",
            "feature",
            caps[0],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[0]", "key_capabilities", caps[0])],
        ),
    ]
    for index, fact in enumerate(source.functional_requirements):
        parent = "feat-auth" if index in {3, 4} else "feat-review"
        items.append(
            _item(
                f"story-fr-{index}",
                "user_story",
                fact.text,
                parent=parent,
                provenance=[_src(fact.json_path, fact.field_name or "functional_requirements", fact.text)],
                description=(
                    "An actor completes this behavior and an observable result is recorded "
                    "for later review."
                ),
            )
        )
    for section in source.narrative_sections:
        if section.evidence_kind != "outcome":
            continue
        items.append(
            _item(
                f"story-{section.label.lower()}",
                "user_story",
                section.title,
                parent="feat-review",
                provenance=[_src("$.user_request", "user_request", section.excerpt)],
                description=(
                    "An actor completes the labelled behavior and the portal records the "
                    "observable result without inventing a missing rule."
                ),
                title_origin="generated",
            )
        )
    unknown = source.unknowns[0]
    items.append(
        _item(
            "task-api-docs",
            "task",
            "Confirm whether policy administration API documentation is available",
            parent="story-fr-3",
            provenance=[_src(unknown.json_path, unknown.field_name or "unknowns", unknown.text)],
            title_origin="generated",
        )
    )
    return {"items": items, "prerequisites": [], "uncertainties": []}
