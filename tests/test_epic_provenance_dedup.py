"""Equivalent epic provenance collapses to one fact without dropping sources."""

from __future__ import annotations

from backlog_agent.backlog.decompose import _to_canonical
from backlog_agent.backlog.generate import _FindingSink
from backlog_agent.backlog.models import SourceReference
from backlog_agent.backlog.proposal import BacklogProposal, ProposedProvenance
from backlog_agent.backlog.provenance import collapse_equivalent_provenance
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.source import load_project_source
from tests.llm_proposal_fixtures import representative_complete_proposal
from tests.test_generate_canonical_backlog import load_fixture

_CONTRACT = "copilot.workflow_response.v1"

_CANCELLATION = [
    ("$.discovery.unknowns[2]", "unknowns", "Have cancellation policy and proration rules been approved?"),
    ("$.requirements.open_questions[2]", "open_questions", "Have cancellation policy and proration rules been approved?"),
    (
        "$.delivery_plan.dependencies[1]",
        "dependencies",
        "Final business sign-off and approval on cancellation policies and proration rules",
    ),
    (
        "$.solution.dependencies[1]",
        "dependencies",
        "Final business sign-off and approval on cancellation policies and proration rules",
    ),
    (
        "$.sow.dependencies[1]",
        "dependencies",
        "Final business sign-off and approval on cancellation policies and proration rules.",
    ),
]
_VOLUME = [
    (
        "$.discovery.unknowns[3]",
        "unknowns",
        "Can the business confirm and sign off on the expected customer volumes, peak transaction loads, and concurrent user metrics for production?",
    ),
    (
        "$.requirements.open_questions[3]",
        "open_questions",
        "Can the business confirm and sign off on the expected customer volumes, peak transaction loads, and concurrent user metrics for production?",
    ),
    (
        "$.delivery_plan.dependencies[2]",
        "dependencies",
        "Confirmation of production customer volumes, peak transaction loads, and concurrent user metrics",
    ),
    (
        "$.solution.dependencies[2]",
        "dependencies",
        "Confirmation of production customer volumes, peak transaction loads, and concurrent user metrics",
    ),
    (
        "$.sow.dependencies[2]",
        "dependencies",
        "Confirmation of production customer volumes, peak transaction loads, and concurrent user metrics.",
    ),
]
_RETENTION = [
    ("$.discovery.unknowns[0]", "unknowns", "Has the photo retention period been approved?"),
    ("$.sow.dependencies[0]", "dependencies", "Final sign-off on the photo retention period."),
]
_HOSTING = [
    ("$.discovery.unknowns[1]", "unknowns", "Has the hosting region been approved?"),
]


def _ref(path: str, field: str, excerpt: str) -> SourceReference:
    return SourceReference(
        contract_id=_CONTRACT,
        json_path=path,
        field_name=field,
        excerpt=excerpt,
    )


def _refs(rows: list[tuple[str, str, str]]) -> list[SourceReference]:
    return [_ref(path, field, excerpt) for path, field, excerpt in rows]


def _all_paths(reference: SourceReference) -> list[str]:
    return [reference.json_path, *[item.json_path for item in reference.equivalent_references]]


def _all_excerpts(reference: SourceReference) -> list[str]:
    return [reference.excerpt or "", *[item.excerpt or "" for item in reference.equivalent_references]]


def test_exact_and_formatted_duplicates_collapse_and_keep_every_source() -> None:
    rows = _refs(_CANCELLATION)
    rows.append(_ref("$.discovery.unknowns[2]", "unknowns", "  HAVE   cancellation policy and proration rules been approved? "))
    rows.append(_ref("$.sow.dependencies[4]", "dependencies", "Have cancellation POLICY and proration rules been approved."))

    collapsed = collapse_equivalent_provenance(rows)

    assert len(collapsed) == 1
    paths = _all_paths(collapsed[0])
    assert paths == [
        "$.discovery.unknowns[2]",
        "$.requirements.open_questions[2]",
        "$.delivery_plan.dependencies[1]",
        "$.solution.dependencies[1]",
        "$.sow.dependencies[1]",
        "$.sow.dependencies[4]",
    ]
    assert "Final business sign-off and approval on cancellation policies and proration rules." in _all_excerpts(
        collapsed[0]
    )
    dumped = collapsed[0].model_dump(mode="json")
    for item in [dumped, *dumped["equivalent_references"]]:
        assert set(item) >= {"contract_id", "json_path", "field_name", "excerpt"}
        assert item["contract_id"] == _CONTRACT


def test_billing_concerns_stay_separate_and_volume_wording_collapses() -> None:
    collapsed = collapse_equivalent_provenance(_refs([*_CANCELLATION, *_VOLUME]))

    assert len(collapsed) == 2
    assert _all_paths(collapsed[0]) == [path for path, _field, _excerpt in _CANCELLATION]
    assert _all_paths(collapsed[1]) == [path for path, _field, _excerpt in _VOLUME]
    assert collapsed[0].excerpt == _CANCELLATION[0][2]
    assert collapsed[1].excerpt == _VOLUME[0][2]


def test_similar_subjects_with_different_qualifiers_remain_separate() -> None:
    production = _ref(
        "$.delivery_plan.dependencies[0]",
        "dependencies",
        "Confirmation of production photo upload volumes",
    )
    pilot = _ref(
        "$.delivery_plan.dependencies[1]",
        "dependencies",
        "Confirmation of pilot photo upload volumes",
    )
    policy_only = _ref(
        "$.discovery.unknowns[0]",
        "unknowns",
        "Has the cancellation policy been approved?",
    )
    policy_and_proration = _ref(
        "$.discovery.unknowns[1]",
        "unknowns",
        "Have cancellation policy and proration rules been approved?",
    )
    denied = _ref(
        "$.discovery.unknowns[2]",
        "unknowns",
        "Cancellation policy and proration rules have not been approved.",
    )

    collapsed = collapse_equivalent_provenance(
        [production, pilot, policy_only, policy_and_proration, denied]
    )

    assert [item.excerpt for item in collapsed] == [
        production.excerpt,
        pilot.excerpt,
        policy_only.excerpt,
        policy_and_proration.excerpt,
        denied.excerpt,
    ]
    assert all(item.equivalent_references == [] for item in collapsed)


def test_library_retention_collapses_and_hosting_stays_separate() -> None:
    collapsed = collapse_equivalent_provenance(_refs([*_RETENTION, *_HOSTING]))

    assert len(collapsed) == 2
    assert _all_paths(collapsed[0]) == ["$.discovery.unknowns[0]", "$.sow.dependencies[0]"]
    assert collapsed[0].equivalent_references[0].excerpt == "Final sign-off on the photo retention period."
    assert collapsed[1].json_path == "$.discovery.unknowns[1]"
    assert collapsed[1].equivalent_references == []


def test_collapse_is_deterministic_and_idempotent() -> None:
    first = collapse_equivalent_provenance(_refs([*_VOLUME, *_CANCELLATION]))
    second = collapse_equivalent_provenance(first)
    third = collapse_equivalent_provenance(_refs([*_VOLUME, *_CANCELLATION]))

    assert [item.model_dump(mode="json") for item in second] == [
        item.model_dump(mode="json") for item in first
    ]
    assert [item.model_dump(mode="json") for item in third] == [
        item.model_dump(mode="json") for item in first
    ]
    assert [item.json_path for item in first] == [
        "$.discovery.unknowns[3]",
        "$.discovery.unknowns[2]",
    ]


def test_batches_from_separate_aggregation_paths_collapse_together() -> None:
    questions = _refs(_CANCELLATION[:2])
    dependencies = _refs(_CANCELLATION[2:])
    collapsed = collapse_equivalent_provenance([*questions, *dependencies])

    assert len(collapsed) == 1
    assert _all_paths(collapsed[0]) == [path for path, _field, _excerpt in _CANCELLATION]


def test_existing_provenance_payload_remains_compatible() -> None:
    original = {
        "contract_id": _CONTRACT,
        "json_path": "$.user_request",
        "field_name": "excerpt",
        "excerpt": "Build a reading-room request desk.",
    }
    loaded = SourceReference.model_validate(original)
    dumped = loaded.model_dump(mode="json")

    assert dumped == original
    assert loaded.equivalent_references == []


def test_canonical_epic_collapses_equivalent_provenance_and_stories_do_not() -> None:
    source = load_project_source(CopilotWorkflowResponseV1.model_validate(load_fixture()))
    proposal = BacklogProposal.model_validate(representative_complete_proposal())
    epic = next(item for item in proposal.items if item.type.value == "epic")
    story = next(item for item in proposal.items if item.type.value == "user_story")
    story_paths = [reference.json_path for reference in story.provenance]
    epic.provenance = [
        ProposedProvenance(json_path=path, field_name=field, excerpt=excerpt)
        for path, field, excerpt in (*_CANCELLATION, *_VOLUME, *_HOSTING)
    ]

    backlog = _to_canonical(source, proposal, _FindingSink())
    canonical_epic = next(item for item in backlog.items if item.type.value == "epic")
    canonical_story = next(item for item in backlog.items if item.title == story.title)

    epic_dump = [item.model_dump(mode="json") for item in canonical_epic.provenance]
    assert len(epic_dump) == 3
    assert [len(item.get("equivalent_references") or []) for item in epic_dump] == [4, 4, 0]
    assert epic_dump[0]["json_path"] == "$.discovery.unknowns[2]"
    assert epic_dump[1]["json_path"] == "$.discovery.unknowns[3]"
    assert {item["json_path"] for item in epic_dump[0]["equivalent_references"]} == {
        "$.requirements.open_questions[2]",
        "$.delivery_plan.dependencies[1]",
        "$.solution.dependencies[1]",
        "$.sow.dependencies[1]",
    }
    assert [item.json_path for item in canonical_story.provenance] == story_paths
    assert all(item.equivalent_references == [] for item in canonical_story.provenance)
    assert "equivalent_references" not in canonical_story.provenance[0].model_dump(mode="json")
