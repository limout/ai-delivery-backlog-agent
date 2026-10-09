from __future__ import annotations

import copy

import pytest

from backlog_agent.backlog.decompose import decompose_backlog
from backlog_agent.backlog.models import WorkItemType
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import FindingCode
from backlog_agent.llm.exceptions import AIProviderError, DecompositionValidationError
from backlog_agent.llm.mock import MockProvider
from tests.llm_proposal_fixtures import proposal_with_out_of_scope_feature, representative_complete_proposal
from tests.test_generate_canonical_backlog import load_fixture


def _envelope() -> CopilotWorkflowResponseV1:
    return CopilotWorkflowResponseV1.model_validate(load_fixture())


def test_decompose_complete_fixture_with_mock_provider() -> None:
    provider = MockProvider(representative_complete_proposal())
    backlog, findings = decompose_backlog(_envelope(), provider)

    types = {item.type for item in backlog.items}
    assert WorkItemType.EPIC in types
    epic = next(item for item in backlog.items if item.type is WorkItemType.EPIC)
    assert epic.title == "Customer self-service policy renewal portal"
    assert epic.title != load_fixture()["user_request"]

    feature_titles = [item.title for item in backlog.items if item.type is WorkItemType.FEATURE]
    assert "Authenticated customer portal" in feature_titles
    assert "Policy renewal workflow" in feature_titles
    assert "Customer portal" not in feature_titles
    assert "Notifications" not in feature_titles
    assert "Claims adjudication workspace" not in feature_titles

    stories = [item for item in backlog.items if item.type is WorkItemType.USER_STORY]
    assert len(stories) == 3
    assert all(item.parent_id is not None for item in stories)
    covered = {
        ref.json_path
        for story in stories
        for ref in story.provenance
        if ref.json_path.startswith("$.requirements.functional_requirements")
    }
    assert covered == {
        "$.requirements.functional_requirements[0]",
        "$.requirements.functional_requirements[1]",
        "$.requirements.functional_requirements[2]",
    }

    assert any(item.acceptance_criteria for item in stories)
    assert any(item.test_requirements for item in stories)
    assert any(item.dependencies for item in backlog.items)
    assert any(item.code is FindingCode.GENERATED_CONTENT for item in findings.items)
    assert any(item.code is FindingCode.AMBIGUOUS_DECOMPOSITION for item in findings.items)
    assert provider.calls
    assert "Do not copy delivery workstreams" in provider.calls[0][0]


def test_decompose_rejects_out_of_scope_item() -> None:
    provider = MockProvider(proposal_with_out_of_scope_feature())
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(_envelope(), provider, repair=False)

    assert any("out_of_scope" in error for error in exc.value.errors)
    assert any(item.code is FindingCode.OUT_OF_SCOPE_EXCLUDED for item in exc.value.findings.items)


def test_decompose_rejects_ungrounded_provenance() -> None:
    proposal = representative_complete_proposal()
    proposal["items"][1]["provenance"][0]["excerpt"] = "not in the envelope"
    provider = MockProvider(proposal)

    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(_envelope(), provider, repair=False)

    assert any("excerpt" in error for error in exc.value.errors)
    assert any(item.code is FindingCode.UNGROUNDED_PROVENANCE for item in exc.value.findings.items)


def test_decompose_rejects_wrong_parent_type() -> None:
    proposal = representative_complete_proposal()
    story = next(item for item in proposal["items"] if item["local_id"] == "story-renew")
    story["parent_local_id"] = "epic-1"
    provider = MockProvider(proposal)

    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(_envelope(), provider, repair=False)

    assert any("parent must be a feature" in error for error in exc.value.errors)


def test_decompose_rejects_hierarchy_cycle() -> None:
    proposal = representative_complete_proposal()
    epic = next(item for item in proposal["items"] if item["local_id"] == "epic-1")
    feature = next(item for item in proposal["items"] if item["local_id"] == "feat-portal")
    epic["type"] = "feature"
    epic["parent_local_id"] = "feat-portal"
    feature["parent_local_id"] = "epic-1"
    # schema still requires an epic
    proposal["items"].append(
        {
            **copy.deepcopy(epic),
            "local_id": "epic-real",
            "type": "epic",
            "parent_local_id": None,
            "title_origin": "generated",
        }
    )
    provider = MockProvider(proposal)

    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(_envelope(), provider, repair=False)

    assert any("cycle" in error for error in exc.value.errors)


def test_decompose_rejects_dependency_cycle() -> None:
    proposal = representative_complete_proposal()
    mvp = next(item for item in proposal["items"] if item["local_id"] == "task-mvp")
    email = next(item for item in proposal["items"] if item["local_id"] == "task-email")
    mvp["depends_on_local_ids"] = ["task-email"]
    email["depends_on_local_ids"] = ["task-mvp"]
    provider = MockProvider(proposal)

    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(_envelope(), provider, repair=False)

    assert any("dependency cycle" in error for error in exc.value.errors)


def test_decompose_rejects_uncovered_requirement() -> None:
    proposal = representative_complete_proposal()
    proposal["items"] = [item for item in proposal["items"] if item["local_id"] != "story-premium"]
    # remove child task parent pointing at missing story is fine; premium had no children
    provider = MockProvider(proposal)

    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(_envelope(), provider, repair=False)

    assert any("Uncovered functional requirement" in error for error in exc.value.errors)
    assert any(
        item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT for item in exc.value.findings.items
    )


def test_decompose_repairs_once_then_succeeds() -> None:
    bad = representative_complete_proposal()
    bad["items"] = [item for item in bad["items"] if item["local_id"] != "story-premium"]
    good = representative_complete_proposal()
    provider = MockProvider(responses=[bad, good])

    backlog, _findings = decompose_backlog(_envelope(), provider)

    assert len(provider.calls) == 2
    assert "failed deterministic validation" in provider.calls[1][0]
    assert any(item.type is WorkItemType.USER_STORY and "premium" in item.title.lower() for item in backlog.items)


def test_decompose_repairs_once_then_raises_actionable_errors() -> None:
    bad = representative_complete_proposal()
    bad["items"] = [item for item in bad["items"] if item["local_id"] != "story-premium"]
    provider = MockProvider(responses=[bad, copy.deepcopy(bad)])

    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(_envelope(), provider)

    assert len(provider.calls) == 2
    assert any(item.code is FindingCode.LLM_VALIDATION_FAILED for item in exc.value.findings.items)
    assert exc.value.errors


def test_decompose_does_not_call_network_on_provider_error() -> None:
    provider = MockProvider(error=AIProviderError("upstream down"))
    with pytest.raises(AIProviderError, match="upstream down"):
        decompose_backlog(_envelope(), provider)
    assert len(provider.calls) == 1


def test_source_origin_title_must_match_excerpt() -> None:
    proposal = representative_complete_proposal()
    proposal["items"][1]["title"] = "Portal capability (paraphrased)"
    proposal["items"][1]["title_origin"] = "source"
    provider = MockProvider(proposal)

    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(_envelope(), provider, repair=False)

    assert any("title_origin=source" in error for error in exc.value.errors)
