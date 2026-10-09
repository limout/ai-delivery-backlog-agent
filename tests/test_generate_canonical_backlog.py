from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backlog_agent.backlog.generate import generate_canonical_backlog, resolve_json_path
from backlog_agent.backlog.ids import canonical_id_from_item_identity
from backlog_agent.backlog.models import ApprovalState, WorkItemStatus, WorkItemType
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import FindingCode

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "copilot.workflow_response.v1.complete.json"


def load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_fixture_validates_against_copilot_contract() -> None:
    parsed = CopilotWorkflowResponseV1.model_validate(load_fixture())

    assert parsed.envelope.status == "COMPLETE"
    assert parsed.envelope.workflow_status == "COMPLETE"


def test_generation_is_deterministic() -> None:
    payload = load_fixture()

    first_backlog, first_findings = generate_canonical_backlog(payload)
    second_backlog, second_findings = generate_canonical_backlog(payload)

    assert first_backlog.model_dump() == second_backlog.model_dump()
    assert first_findings.model_dump() == second_findings.model_dump()


def test_ids_are_content_addressed_not_index_paths() -> None:
    payload = load_fixture()
    backlog, _findings = generate_canonical_backlog(payload)
    story = next(item for item in backlog.items if item.type is WorkItemType.USER_STORY)

    assert story.canonical_id == canonical_id_from_item_identity(
        "$.requirements.functional_requirements", story.title, 0
    )
    assert story.canonical_id != canonical_id_from_item_identity(
        story.provenance[0].json_path, story.title, 0
    )
    assert story.provenance[0].json_path.startswith("$.requirements.functional_requirements[")


def test_ids_stable_when_unrelated_array_changes() -> None:
    payload = load_fixture()
    original_backlog, _ = generate_canonical_backlog(payload)
    original_ids = {item.title: item.canonical_id for item in original_backlog.items}

    payload["delivery_plan"]["workstreams"].insert(0, "Reporting")
    updated_backlog, _ = generate_canonical_backlog(payload)

    for title, canonical_id in original_ids.items():
        if title == "Reporting":
            continue
        match = next((item for item in updated_backlog.items if item.title == title), None)
        if match is None:
            continue
        if match.type is WorkItemType.FEATURE and title in {"Customer portal", "Notifications"}:
            continue
        assert match.canonical_id == canonical_id, title


def test_ids_stable_when_same_array_inserts_unrelated_text() -> None:
    payload = load_fixture()
    original_backlog, _ = generate_canonical_backlog(payload)
    first_fr = payload["requirements"]["functional_requirements"][0]
    original = next(item for item in original_backlog.items if item.title == first_fr)

    payload["requirements"]["functional_requirements"].insert(0, "The system shall log renewal attempts.")
    updated_backlog, _ = generate_canonical_backlog(payload)
    moved = next(item for item in updated_backlog.items if item.title == first_fr)

    assert moved.canonical_id == original.canonical_id
    assert moved.provenance[0].json_path == "$.requirements.functional_requirements[1]"
    assert original.provenance[0].json_path == "$.requirements.functional_requirements[0]"


def test_ids_change_when_item_text_changes() -> None:
    payload = load_fixture()
    original_backlog, _ = generate_canonical_backlog(payload)
    original = next(
        item
        for item in original_backlog.items
        if item.provenance[0].json_path == "$.requirements.functional_requirements[0]"
    )

    payload["requirements"]["functional_requirements"][0] = "The system shall allow a broker to renew a policy."
    updated_backlog, _ = generate_canonical_backlog(payload)
    changed = next(
        item
        for item in updated_backlog.items
        if item.provenance[0].json_path == "$.requirements.functional_requirements[0]"
    )

    assert changed.canonical_id != original.canonical_id


def test_reorder_updates_provenance_index_not_id() -> None:
    payload = load_fixture()
    first, second = payload["requirements"]["functional_requirements"][:2]
    original_backlog, _ = generate_canonical_backlog(payload)
    original_first = next(item for item in original_backlog.items if item.title == first)

    payload["requirements"]["functional_requirements"][0], payload["requirements"]["functional_requirements"][1] = (
        second,
        first,
    )
    updated_backlog, _ = generate_canonical_backlog(payload)
    moved = next(item for item in updated_backlog.items if item.title == first)

    assert moved.canonical_id == original_first.canonical_id
    assert moved.provenance[0].json_path == "$.requirements.functional_requirements[1]"


def test_every_item_has_real_provenance() -> None:
    payload = load_fixture()
    backlog, _findings = generate_canonical_backlog(payload)

    assert backlog.items
    for item in backlog.items:
        assert item.provenance
        for reference in item.provenance:
            resolved = resolve_json_path(payload, reference.json_path)
            assert resolved is not None, reference.json_path
            if isinstance(resolved, str):
                assert reference.excerpt == resolved


def test_out_of_scope_content_is_excluded() -> None:
    payload = load_fixture()
    backlog, findings = generate_canonical_backlog(payload)
    titles = {item.title for item in backlog.items}

    assert "Claims adjudication workspace" not in titles
    assert "Broker commission management" not in titles
    assert any(item.code is FindingCode.OUT_OF_SCOPE_EXCLUDED for item in findings.items)
    assert not any(
        item.code is FindingCode.COPILOT_CONTRADICTION
        and "contradicts sow.out_of_scope" in item.message
        for item in findings.items
    )


def test_unparented_stories_are_retained_and_still_cover_requirements() -> None:
    payload = load_fixture()
    payload["solution"]["key_capabilities"] = []
    payload["delivery_plan"]["workstreams"] = []

    backlog, findings = generate_canonical_backlog(payload)
    stories = [item for item in backlog.items if item.type is WorkItemType.USER_STORY]

    assert len(stories) == 3
    assert all(item.parent_id is None for item in stories)
    assert all(item.approval_state is ApprovalState.NOT_APPROVED for item in stories)
    assert not any(item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT for item in findings.items)
    assert any(item.code is FindingCode.AMBIGUOUS_DECOMPOSITION for item in findings.items)


def test_fr_coverage_requires_emitted_provenance_path() -> None:
    payload = load_fixture()
    backlog, findings = generate_canonical_backlog(payload)
    fr_paths = [
        f"$.requirements.functional_requirements[{index}]"
        for index, _value in enumerate(payload["requirements"]["functional_requirements"])
    ]
    covered = {
        reference.json_path
        for item in backlog.items
        if item.type is WorkItemType.USER_STORY
        for reference in item.provenance
    }
    assert set(fr_paths) <= covered
    assert not any(item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT for item in findings.items)


def test_requirement_negation_is_a_contradiction() -> None:
    payload = load_fixture()
    payload["requirements"]["functional_requirements"].append(
        "The system shall not allow a customer to renew an eligible policy online."
    )

    _backlog, findings = generate_canonical_backlog(payload)
    contradictions = [
        item
        for item in findings.items
        if item.code is FindingCode.COPILOT_CONTRADICTION and "shall / shall-not" in item.message
    ]
    assert len(contradictions) == 1


def test_one_finding_per_sow_overlap() -> None:
    payload = load_fixture()
    payload["sow"]["in_scope"].append("Policy renewal workflow")
    payload["sow"]["out_of_scope"].append("Policy renewal workflow")

    backlog, findings = generate_canonical_backlog(payload)
    scope_conflicts = [item for item in findings.items if item.code is FindingCode.SCOPE_CONFLICT]

    assert "Policy renewal workflow" not in {item.title for item in backlog.items}
    assert len(scope_conflicts) == 1
    assert not any(item.code is FindingCode.COPILOT_CONTRADICTION for item in findings.items)


def test_scope_short_fragment_does_not_exclude() -> None:
    payload = load_fixture()
    payload["sow"]["out_of_scope"].append("email notifications")

    backlog, _findings = generate_canonical_backlog(payload)
    titles = {item.title for item in backlog.items}

    assert "Notifications" in titles


def test_scope_full_sow_line_contained_in_source_excludes() -> None:
    payload = load_fixture()
    payload["sow"]["out_of_scope"].append("renew an eligible policy online")

    backlog, findings = generate_canonical_backlog(payload)
    titles = {item.title for item in backlog.items}

    assert "The system shall allow a customer to renew an eligible policy online." not in titles
    assert any(item.code is FindingCode.OUT_OF_SCOPE_EXCLUDED for item in findings.items)


def test_no_parent_link_when_multiple_features() -> None:
    payload = load_fixture()
    backlog, findings = generate_canonical_backlog(payload)
    stories = [item for item in backlog.items if item.type is WorkItemType.USER_STORY]
    features = [item for item in backlog.items if item.type is WorkItemType.FEATURE]

    assert len(features) > 1
    assert stories
    assert all(item.parent_id is None for item in stories)
    assert all(not item.child_ids or item.type is WorkItemType.EPIC or item.type is WorkItemType.FEATURE for item in backlog.items)
    assert all(not feature.child_ids for feature in features)
    assert any(item.code is FindingCode.AMBIGUOUS_DECOMPOSITION for item in findings.items)


def test_unique_parent_still_links() -> None:
    payload = load_fixture()
    payload["solution"]["key_capabilities"] = ["Policy renewal workflow"]
    payload["delivery_plan"]["workstreams"] = []

    backlog, _findings = generate_canonical_backlog(payload)
    features = [item for item in backlog.items if item.type is WorkItemType.FEATURE]
    stories = [item for item in backlog.items if item.type is WorkItemType.USER_STORY]
    epic = next(item for item in backlog.items if item.type is WorkItemType.EPIC)

    assert len(features) == 1
    assert features[0].parent_id == epic.canonical_id
    assert {item.canonical_id for item in stories} <= set(features[0].child_ids)
    assert all(item.parent_id == features[0].canonical_id for item in stories)


def test_generated_items_are_not_approved_and_todo() -> None:
    payload = load_fixture()
    backlog, _findings = generate_canonical_backlog(payload)

    assert backlog.items
    for item in backlog.items:
        assert item.status is WorkItemStatus.TODO
        assert item.approval_state is ApprovalState.NOT_APPROVED
        assert item.effort is None
        assert item.priority is None


def test_happy_path_fixture_retains_all_in_scope_types() -> None:
    payload = load_fixture()
    backlog, findings = generate_canonical_backlog(payload)
    types = {item.type for item in backlog.items}

    assert WorkItemType.EPIC in types
    assert WorkItemType.FEATURE in types
    assert WorkItemType.USER_STORY in types
    assert WorkItemType.TASK in types
    assert WorkItemType.SUBTASK in types
    assert any(item.parent_id is None for item in backlog.items if item.type is WorkItemType.USER_STORY)
    assert any(item.code is FindingCode.AMBIGUOUS_DECOMPOSITION for item in findings.items)
