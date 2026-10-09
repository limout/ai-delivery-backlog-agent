"""Minimal CLI to preview a canonical backlog from a saved Copilot JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from pydantic import ValidationError

from backlog_agent.backlog.decompose import decompose_backlog
from backlog_agent.backlog.generate import generate_canonical_backlog
from backlog_agent.backlog.models import CanonicalBacklogV1, WorkItemType
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import FindingCode, GenerationFindings
from backlog_agent.llm.exceptions import AIProviderError, DecompositionValidationError
from backlog_agent.llm.factory import get_provider


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="backlog-preview",
        description=(
            "Validate a saved Copilot workflow_response JSON file, generate a "
            "canonical backlog, and write backlog + findings to JSON. "
            "Default generation is deterministic. Pass --llm to use the "
            "configured AI_PROVIDER (Gemini unless overridden)."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to a saved Copilot workflow_response JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to write the serialized canonical backlog and findings",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Propose the backlog with the configured LLM provider, then validate in Python",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        payload = _load_json(Path(args.input))
        envelope = CopilotWorkflowResponseV1.model_validate(payload)
        if args.llm:
            backlog, findings = decompose_backlog(envelope, get_provider())
        else:
            backlog, findings = generate_canonical_backlog(envelope)
        document = serialize_preview(backlog, findings)
        _write_json(Path(args.output), document)
    except (
        OSError,
        json.JSONDecodeError,
        ValidationError,
        ValueError,
        AIProviderError,
        DecompositionValidationError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print_summary(backlog, findings, file=sys.stdout)
    return 0


def serialize_preview(backlog: CanonicalBacklogV1, findings: GenerationFindings) -> dict[str, Any]:
    return {
        "backlog": backlog.model_dump(mode="json"),
        "findings": findings.model_dump(mode="json"),
    }


def print_summary(
    backlog: CanonicalBacklogV1,
    findings: GenerationFindings,
    *,
    file: TextIO,
) -> None:
    type_counts = Counter(item.type.value for item in backlog.items)
    unparented = [
        item
        for item in backlog.items
        if item.parent_id is None and item.type is not WorkItemType.EPIC
    ]
    severity_counts = Counter(item.severity.value for item in findings.items)
    code_counts = Counter(item.code.value for item in findings.items)
    uncovered = [
        item for item in findings.items if item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT
    ]

    print("Canonical backlog preview", file=file)
    print(
        "Items by type: "
        + ", ".join(f"{name}={type_counts.get(name, 0)}" for name in _type_order()),
        file=file,
    )
    print(f"Items without parent (non-epic): {len(unparented)}", file=file)
    print(
        "Findings by severity: " + _format_counts(severity_counts, ("error", "warning", "info")),
        file=file,
    )
    print("Findings by code: " + _format_counts(code_counts, sorted(code_counts)), file=file)
    print(f"Functional requirements without coverage: {len(uncovered)}", file=file)
    for finding in uncovered:
        excerpts = [ref.excerpt for ref in finding.source_references if ref.excerpt]
        if excerpts:
            print(f"  - {excerpts[0]}", file=file)


def _type_order() -> tuple[str, ...]:
    return tuple(member.value for member in WorkItemType)


def _format_counts(counts: Mapping[str, int], keys: Sequence[str]) -> str:
    if not keys:
        return "none"
    return ", ".join(f"{key}={counts.get(key, 0)}" for key in keys)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
