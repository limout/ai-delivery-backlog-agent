from __future__ import annotations

from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.source import load_project_source
from backlog_agent.source.model import classify_claim
from tests.copilot_envelopes import library_holds_envelope
from tests.test_decompose_confirmed_facts import campus_self_service_envelope


def test_listed_dependency_is_established_when_request_confirms_it() -> None:
    source = load_project_source(
        CopilotWorkflowResponseV1.model_validate(campus_self_service_envelope())
    )
    access = source.blocking_dependencies[0]
    stakeholders = source.blocking_dependencies[2]
    assert classify_claim(access.text, source, skip_path=access.json_path) == "established"
    assert classify_claim(stakeholders.text, source, skip_path=stakeholders.json_path) == "established"
    assert source.prompt_facts["blocking_dependencies"][0]["evidence_status"] == "established"
    assert source.prompt_facts["blocking_dependencies"][2]["evidence_status"] == "established"


def test_remaining_unknown_is_not_swallowed_by_confirmed_access() -> None:
    source = load_project_source(
        CopilotWorkflowResponseV1.model_validate(campus_self_service_envelope())
    )
    unknown = source.unknowns[0]
    assert "sandbox" in unknown.text.lower()
    assert classify_claim(unknown.text, source, skip_path=unknown.json_path) == "unresolved"
    assert source.prompt_facts["unknowns"][0]["evidence_status"] == "unresolved"


def test_unresolved_dependency_wording_stays_unresolved() -> None:
    source = load_project_source(CopilotWorkflowResponseV1.model_validate(library_holds_envelope()))
    dependency = source.blocking_dependencies[0]
    assert classify_claim(dependency.text, source, skip_path=dependency.json_path) == "unresolved"
    assert source.prompt_facts["blocking_dependencies"][0]["evidence_status"] == "unresolved"


def test_contradictory_access_statements_are_conflict() -> None:
    source = load_project_source(
        CopilotWorkflowResponseV1.model_validate(campus_self_service_envelope(contradict_access=True))
    )
    dependency = source.blocking_dependencies[0]
    assert classify_claim(dependency.text, source, skip_path=dependency.json_path) == "conflict"
    assert source.prompt_facts["blocking_dependencies"][0]["evidence_status"] == "conflict"
