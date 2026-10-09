from __future__ import annotations

import pytest

from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.source import UnsupportedInputError, load_project_source
from tests.test_generate_canonical_backlog import load_fixture


def test_copilot_adapter_extracts_generic_facts() -> None:
    source = load_project_source(CopilotWorkflowResponseV1.model_validate(load_fixture()))

    assert source.request.json_path == "$.user_request"
    assert len(source.functional_requirements) == 3
    assert source.functional_requirements[0].json_path == "$.requirements.functional_requirements[0]"
    assert source.capabilities
    assert source.out_of_scope
    assert "Claims adjudication workspace" in {fact.text for fact in source.out_of_scope}
    assert source.prompt_facts["functional_requirements"]
    assert source.prompt_facts["functional_requirements"][0]["json_path"] == (
        "$.requirements.functional_requirements[0]"
    )
    assert source.prompt_facts["request"]["json_path"] == "$.user_request"


def test_unsupported_arbitrary_json_is_rejected() -> None:
    with pytest.raises(UnsupportedInputError, match="arbitrary JSON"):
        load_project_source({"issues": [{"key": "ABC-1"}]})


def test_insufficient_copilot_envelope_is_rejected() -> None:
    payload = load_fixture()
    payload["status"] = "NEEDS_INFO"
    payload["workflow_status"] = "NEEDS_INFO"
    with pytest.raises(UnsupportedInputError, match="insufficient"):
        load_project_source(payload)
