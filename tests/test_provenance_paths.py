from __future__ import annotations

import copy

import pytest

from backlog_agent.backlog.decompose import decompose_backlog
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.llm.exceptions import DecompositionValidationError
from backlog_agent.llm.mock import MockProvider
from backlog_agent.source import load_project_source
from tests.llm_proposal_fixtures import representative_complete_proposal
from tests.test_generate_canonical_backlog import load_fixture


def _envelope(extra: dict | None = None) -> CopilotWorkflowResponseV1:
    payload = load_fixture()
    if extra:
        payload.update(extra)
    return CopilotWorkflowResponseV1.model_validate(payload)


def test_prompt_facts_use_envelope_json_paths_not_adapter_keys() -> None:
    source = load_project_source(_envelope())
    facts = source.prompt_facts

    assert facts["request"]["json_path"] == "$.user_request"
    assert facts["functional_requirements"][0]["json_path"] == (
        "$.requirements.functional_requirements[0]"
    )
    assert facts["capabilities"][0]["json_path"] == "$.solution.key_capabilities[0]"
    assert facts["acceptance_criteria"][0]["json_path"] == (
        "$.requirements.acceptance_criteria[0]"
    )
    assert "contract_id" not in facts
    assert facts["metadata"]["source_contract_id"]


def test_prompt_includes_envelope_paths_for_normalized_fact_groups() -> None:
    provider = MockProvider(representative_complete_proposal())
    decompose_backlog(_envelope(), provider, repair=False)
    prompt = provider.calls[0][0]

    assert "$.requirements.functional_requirements[0]" in prompt
    assert "$.solution.key_capabilities[0]" in prompt
    assert "$.user_request" in prompt
    assert "Do not cite prompt-object keys" in prompt


@pytest.mark.parametrize(
    "json_path",
    [
        "$.user_request",
        "$.solution.key_capabilities[0]",
        "$.requirements.functional_requirements[0]",
        "$.requirements.acceptance_criteria[0]",
    ],
)
def test_envelope_provenance_paths_are_accepted(json_path: str) -> None:
    payload = load_fixture()
    excerpts = {
        "$.user_request": payload["user_request"],
        "$.solution.key_capabilities[0]": payload["solution"]["key_capabilities"][0],
        "$.requirements.functional_requirements[0]": payload["requirements"][
            "functional_requirements"
        ][0],
        "$.requirements.acceptance_criteria[0]": payload["requirements"]["acceptance_criteria"][0],
    }
    proposal = representative_complete_proposal()
    epic = next(item for item in proposal["items"] if item["local_id"] == "epic-1")
    epic["provenance"] = [
        {
            "json_path": json_path,
            "field_name": "check",
            "excerpt": excerpts[json_path],
        }
    ]
    epic["title_origin"] = "generated"
    decompose_backlog(_envelope(), MockProvider(proposal), repair=False)


@pytest.mark.parametrize(
    "json_path",
    [
        "$.contract_id",
        "$.capabilities",
        "$.functional_requirements",
        "$.acceptance_criteria",
        "$.non_functional_requirements",
        "$.blocking_dependencies",
        "$.functional_requirements[0]",
        "$.capabilities[0]",
    ],
)
def test_normalized_source_keys_are_rejected_as_provenance(json_path: str) -> None:
    proposal = representative_complete_proposal()
    payload = load_fixture()
    epic = next(item for item in proposal["items"] if item["local_id"] == "epic-1")
    epic["provenance"] = [
        {
            "json_path": json_path,
            "field_name": "check",
            "excerpt": payload["user_request"],
        }
    ]
    epic["title_origin"] = "generated"
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(_envelope(), MockProvider(proposal), repair=False)
    assert any("is not an allowed source field" in error for error in exc.value.errors)
    assert any(json_path in error for error in exc.value.errors)


def test_merged_blocking_dependency_uses_envelope_path_not_prompt_key() -> None:
    payload = load_fixture()
    payload["sow"] = {**payload["sow"], "dependencies": ["Identity provider access is unconfirmed."]}
    envelope = CopilotWorkflowResponseV1.model_validate(payload)
    source = load_project_source(envelope)
    assert source.blocking_dependencies[0].json_path == "$.sow.dependencies[0]"
    assert source.prompt_facts["blocking_dependencies"][0]["json_path"] == "$.sow.dependencies[0]"

    valid = representative_complete_proposal()
    valid["prerequisites"] = [
        {
            "text": "Identity provider access is unconfirmed.",
            "status": "unresolved",
            "provenance": [
                {
                    "json_path": "$.sow.dependencies[0]",
                    "field_name": "dependencies",
                    "excerpt": "Identity provider access is unconfirmed.",
                }
            ],
        }
    ]
    decompose_backlog(envelope, MockProvider(valid), repair=False)

    invalid = copy.deepcopy(valid)
    invalid["prerequisites"][0]["provenance"][0]["json_path"] = "$.blocking_dependencies[0]"
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(envelope, MockProvider(invalid), repair=False)
    assert any("$.blocking_dependencies[0]" in error for error in exc.value.errors)
