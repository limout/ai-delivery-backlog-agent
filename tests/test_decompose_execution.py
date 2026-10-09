from __future__ import annotations

import pytest

from backlog_agent.backlog.decompose import decompose_backlog
from backlog_agent.backlog.models import WorkItemType
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import FindingCode
from backlog_agent.llm.exceptions import DecompositionValidationError
from backlog_agent.llm.mock import MockProvider
from tests.llm_proposal_fixtures import execution_ready_proposal, representative_complete_proposal
from tests.test_generate_canonical_backlog import load_fixture


def _envelope(extra: dict | None = None) -> CopilotWorkflowResponseV1:
    payload = load_fixture()
    if extra:
        payload.update(extra)
    return CopilotWorkflowResponseV1.model_validate(payload)


def test_story_has_useful_description_ac_and_tests() -> None:
    provider = MockProvider(execution_ready_proposal())
    backlog, findings = decompose_backlog(_envelope(), provider)

    story = next(
        item
        for item in backlog.items
        if item.type is WorkItemType.USER_STORY and "renew" in item.title.lower()
    )
    assert "authenticated customer" in story.description.lower()
    assert story.title not in story.description
    assert any("eligible policy" in criterion.lower() or "Eligible" in criterion for criterion in story.acceptance_criteria)
    assert len(story.acceptance_criteria) == 2
    assert story.test_requirements
    assert all("Given an eligible policy" not in req for req in story.test_requirements)
    generated = [
        item
        for item in findings.items
        if item.code is FindingCode.GENERATED_CONTENT and story.canonical_id in item.canonical_ids
    ]
    assert generated
    assert "acceptance_criteria" in generated[0].message
    assert "test_requirements" in generated[0].message


def test_capability_can_have_multiple_implementation_tasks() -> None:
    provider = MockProvider(execution_ready_proposal())
    backlog, _findings = decompose_backlog(_envelope(), provider)

    stories = [item for item in backlog.items if item.type is WorkItemType.USER_STORY]
    story = next(item for item in stories if len(item.child_ids) >= 2)
    tasks = [item for item in backlog.items if item.canonical_id in story.child_ids]
    assert len(tasks) >= 2
    assert all(item.type is WorkItemType.TASK for item in tasks)
    assert len({item.title for item in tasks}) == len(tasks)


def test_justified_dependency_and_unresolved_prerequisite() -> None:
    provider = MockProvider(execution_ready_proposal())
    backlog, findings = decompose_backlog(_envelope(), provider)

    assert any(item.dependencies for item in backlog.items)
    prereqs = [item for item in findings.items if item.code is FindingCode.UNRESOLVED_PREREQUISITE]
    assert prereqs
    assert "not confirmed" in prereqs[0].message.lower() or "not yet" in prereqs[0].message.lower()
    assert not any(item.title.lower().startswith("access to") for item in backlog.items)


def test_deliverables_are_not_copied_one_for_one() -> None:
    payload = load_fixture()
    proposal = representative_complete_proposal()
    for item in list(proposal["items"]):
        if item["type"] == "task" and item["title"] in payload["sow"]["deliverables"]:
            proposal["items"].remove(item)
    # Keep execution tasks that are not a 1:1 deliverable dump.
    proposal["items"].append(
        {
            "local_id": "task-review-screen",
            "type": "task",
            "title": "Implement renewal review screen using policy administration data",
            "description": "Render premium and coverage returned by the policy administration API.",
            "parent_local_id": "story-renew",
            "depends_on_local_ids": [],
            "acceptance_criteria": [],
            "test_requirements": [],
            "provenance": [
                {
                    "json_path": "$.sow.in_scope[0]",
                    "field_name": "in_scope",
                    "excerpt": payload["sow"]["in_scope"][0],
                }
            ],
            "title_origin": "generated",
            "description_origin": "generated",
            "uncertainties": [],
        }
    )
    # Drop orphaned subtask that pointed at the removed MVP task.
    proposal["items"] = [item for item in proposal["items"] if item["type"] != "subtask"]
    provider = MockProvider(proposal)
    backlog, _findings = decompose_backlog(_envelope(), provider)

    task_titles = [item.title for item in backlog.items if item.type is WorkItemType.TASK]
    copied = [title for title in payload["sow"]["deliverables"] if title in task_titles]
    assert copied == []
    assert any("review screen" in title.lower() for title in task_titles)


def test_nfr_traced_when_applicable() -> None:
    payload = load_fixture()
    payload["requirements"]["non_functional_requirements"] = [
        "The existing policy administration system must remain the source of truth.",
        "The web portal must be delivered as a secure MVP within approximately 12 weeks pending feasibility assessment.",
    ]
    proposal = execution_ready_proposal()
    nfr = payload["requirements"]["non_functional_requirements"][0]
    story = next(item for item in proposal["items"] if item["local_id"] == "story-renew")
    story["provenance"].append(
        {
            "json_path": "$.requirements.non_functional_requirements[0]",
            "field_name": "non_functional_requirements",
            "excerpt": nfr,
        }
    )
    provider = MockProvider(proposal)
    _backlog, findings = decompose_backlog(CopilotWorkflowResponseV1.model_validate(payload), provider)

    uncovered = [
        item for item in findings.items if item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT
    ]
    assert uncovered == []


def test_untraced_nfr_is_warning_not_hard_failure() -> None:
    payload = load_fixture()
    payload["requirements"]["non_functional_requirements"] = [
        "Encrypt sensitive information in transit."
    ]
    provider = MockProvider(execution_ready_proposal())
    _backlog, findings = decompose_backlog(CopilotWorkflowResponseV1.model_validate(payload), provider)

    uncovered = [
        item for item in findings.items if item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT
    ]
    assert len(uncovered) == 1
    assert uncovered[0].severity.value == "warning"


def test_rejects_generic_and_duplicate_tasks() -> None:
    proposal = representative_complete_proposal()
    story = "story-renew"
    proposal["items"].extend(
        [
            {
                "local_id": "task-generic",
                "type": "task",
                "title": "Testing",
                "description": "Test the thing.",
                "parent_local_id": story,
                "depends_on_local_ids": [],
                "acceptance_criteria": [],
                "test_requirements": [],
                "provenance": proposal["items"][0]["provenance"],
                "title_origin": "generated",
                "description_origin": "generated",
                "uncertainties": [],
            },
            {
                "local_id": "task-dup-a",
                "type": "task",
                "title": "Build renewal confirmation mailer",
                "description": "Send mail after submit.",
                "parent_local_id": story,
                "depends_on_local_ids": [],
                "acceptance_criteria": [],
                "test_requirements": [],
                "provenance": proposal["items"][0]["provenance"],
                "title_origin": "generated",
                "description_origin": "generated",
                "uncertainties": [],
            },
            {
                "local_id": "task-dup-b",
                "type": "task",
                "title": "Build renewal confirmation mailer",
                "description": "Same work again.",
                "parent_local_id": story,
                "depends_on_local_ids": [],
                "acceptance_criteria": [],
                "test_requirements": [],
                "provenance": proposal["items"][0]["provenance"],
                "title_origin": "generated",
                "description_origin": "generated",
                "uncertainties": [],
            },
            {
                "local_id": "task-workstream",
                "type": "task",
                "title": "Customer portal",
                "description": "Copied workstream.",
                "parent_local_id": story,
                "depends_on_local_ids": [],
                "acceptance_criteria": [],
                "test_requirements": [],
                "provenance": proposal["items"][0]["provenance"],
                "title_origin": "generated",
                "description_origin": "generated",
                "uncertainties": [],
            },
        ]
    )
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(_envelope(), MockProvider(proposal), repair=False)

    joined = " ".join(exc.value.errors)
    assert "generic" in joined.lower()
    assert "duplicate" in joined.lower()
    assert "workstreams" in joined.lower()


def test_blocking_source_dependency_must_be_prerequisite_not_task() -> None:
    payload = load_fixture()
    dependency = "Access to policy administration system API documentation and sandbox environment"
    payload["delivery_plan"]["dependencies"] = [dependency]
    proposal = representative_complete_proposal()
    proposal["items"].append(
        {
            "local_id": "task-access",
            "type": "task",
            "title": dependency,
            "description": "Pretend access is already granted.",
            "parent_local_id": "story-renew",
            "depends_on_local_ids": [],
            "acceptance_criteria": [],
            "test_requirements": [],
            "provenance": [
                {
                    "json_path": "$.delivery_plan.dependencies[0]",
                    "field_name": "dependencies",
                    "excerpt": dependency,
                }
            ],
            "title_origin": "source",
            "description_origin": "generated",
            "uncertainties": [],
        }
    )
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(
            CopilotWorkflowResponseV1.model_validate(payload),
            MockProvider(proposal),
            repair=False,
        )

    assert any("prerequisite" in error for error in exc.value.errors)


def test_rejects_empty_story_description_and_title_restatement() -> None:
    proposal = representative_complete_proposal()
    story = next(item for item in proposal["items"] if item["local_id"] == "story-premium")
    story["description"] = story["title"]
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(_envelope(), MockProvider(proposal), repair=False)
    assert any("restates the title" in error for error in exc.value.errors)

    story["description"] = ""
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(_envelope(), MockProvider(proposal), repair=False)
    assert any("useful description" in error for error in exc.value.errors)


def test_mismatched_origin_list_length_is_aligned_not_rejected() -> None:
    proposal = execution_ready_proposal()
    story = next(item for item in proposal["items"] if item["local_id"] == "story-renew")
    story["test_requirements"] = ["API contract test for eligibility", "UI test for confirm hidden"]
    story["test_requirements_origins"] = ["generated"]
    backlog, findings = decompose_backlog(_envelope(), MockProvider(proposal), repair=False)
    stored = next(item for item in backlog.items if item.canonical_id and "renew" in item.title.lower() and item.type is WorkItemType.USER_STORY)
    assert len(stored.test_requirements) == 2
    assert any(
        item.code is FindingCode.GENERATED_CONTENT and "test_requirements" in item.message
        for item in findings.items
        if stored.canonical_id in item.canonical_ids
    )


def test_generated_ac_cannot_claim_source_origin() -> None:
    proposal = execution_ready_proposal()
    story = next(item for item in proposal["items"] if item["local_id"] == "story-renew")
    story["acceptance_criteria_origins"] = ["source", "source"]
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(_envelope(), MockProvider(proposal), repair=False)
    assert any("origin=source" in error for error in exc.value.errors)


def test_long_scalar_provenance_uses_source_when_excerpt_drifts() -> None:
    proposal = execution_ready_proposal()
    epic = next(item for item in proposal["items"] if item["local_id"] == "epic-1")
    drifted = load_fixture()["user_request"] + " (paraphrased)"
    epic["provenance"][0]["excerpt"] = drifted
    backlog, _findings = decompose_backlog(_envelope(), MockProvider(proposal), repair=False)
    stored = next(item for item in backlog.items if item.type is WorkItemType.EPIC)
    assert stored.provenance[0].excerpt == load_fixture()["user_request"]
    assert stored.provenance[0].excerpt.endswith("paraphrased") is False


def test_one_repair_still_applies_to_execution_errors() -> None:
    bad = representative_complete_proposal()
    next(item for item in bad["items"] if item["local_id"] == "story-premium")["description"] = ""
    good = execution_ready_proposal()
    provider = MockProvider(responses=[bad, good])
    backlog, findings = decompose_backlog(_envelope(), provider)
    assert len(provider.calls) == 2
    assert any(item.type is WorkItemType.TASK for item in backlog.items)
    assert any(item.code is FindingCode.UNRESOLVED_PREREQUISITE for item in findings.items)
