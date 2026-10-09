from __future__ import annotations

import json
from pathlib import Path

import pytest

from backlog_agent.cli import main, serialize_preview
from backlog_agent.backlog.generate import generate_canonical_backlog
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1

FIXTURE = Path(__file__).parent / "fixtures" / "copilot.workflow_response.v1.complete.json"


def test_cli_success_writes_backlog_and_findings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "preview.json"

    code = main(["--input", str(FIXTURE), "--output", str(output)])
    captured = capsys.readouterr()

    assert code == 0
    assert output.is_file()
    document = json.loads(output.read_text(encoding="utf-8"))
    assert "backlog" in document
    assert "findings" in document
    assert document["findings"]["items"]
    assert document["backlog"]["items"]
    assert "Items by type:" in captured.out
    assert "Findings by code:" in captured.out
    assert "Functional requirements without coverage:" in captured.out


def test_cli_output_matches_generator_serialization(tmp_path: Path) -> None:
    output = tmp_path / "preview.json"
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    envelope = CopilotWorkflowResponseV1.model_validate(payload)
    backlog, findings = generate_canonical_backlog(envelope)

    assert main(["--input", str(FIXTURE), "--output", str(output)]) == 0

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == serialize_preview(backlog, findings)
    json.dumps(written)


def test_cli_rejects_missing_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "missing.json"
    output = tmp_path / "out.json"

    code = main(["--input", str(missing), "--output", str(output)])

    assert code == 1
    assert "error:" in capsys.readouterr().err
    assert not output.exists()


def test_cli_rejects_invalid_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    invalid = tmp_path / "not.json"
    invalid.write_text("{not json", encoding="utf-8")
    output = tmp_path / "out.json"

    code = main(["--input", str(invalid), "--output", str(output)])

    assert code == 1
    assert "error:" in capsys.readouterr().err
    assert not output.exists()


def test_cli_llm_flag_uses_provider_not_deterministic_titles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backlog_agent.llm.mock import MockProvider
    from tests.llm_proposal_fixtures import representative_complete_proposal

    monkeypatch.setattr(
        "backlog_agent.cli.get_provider",
        lambda: MockProvider(representative_complete_proposal()),
    )
    output = tmp_path / "preview.json"

    code = main(["--input", str(FIXTURE), "--output", str(output), "--llm"])

    assert code == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    epics = [item for item in document["backlog"]["items"] if item["type"] == "epic"]
    assert epics[0]["title"] == "Customer self-service policy renewal portal"


def test_cli_rejects_incomplete_copilot_envelope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["status"] = "NEEDS_INFO"
    payload["workflow_status"] = "NEEDS_INFO"
    invalid = tmp_path / "needs_info.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "out.json"

    code = main(["--input", str(invalid), "--output", str(output)])

    assert code == 1
    assert "error:" in capsys.readouterr().err
    assert not output.exists()
