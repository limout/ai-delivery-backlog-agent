"""Adapter: copilot.workflow_response.v1 → NormalizedProjectSource."""

from __future__ import annotations

from typing import Any, Mapping

from backlog_agent.backlog.generate import _list_at, resolve_json_path
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.contracts.versions import COPILOT_WORKFLOW_RESPONSE_V1
from backlog_agent.source.model import (
    NormalizedProjectSource,
    SourceFact,
    looks_like_timeline_or_estimate,
)

_ALLOWED_PREFIXES = (
    "$.user_request",
    "$.executive_summary",
    "$.requirements",
    "$.solution",
    "$.delivery_plan",
    "$.sow",
    "$.discovery",
    "$.proposal",
)


def from_copilot(envelope: CopilotWorkflowResponseV1) -> NormalizedProjectSource:
    original = envelope.original_envelope
    request_text = str(original.get("user_request") or "")
    return NormalizedProjectSource(
        contract_id=COPILOT_WORKFLOW_RESPONSE_V1,
        original=original,
        allowed_path_prefixes=_ALLOWED_PREFIXES,
        request=SourceFact("$.user_request", "user_request", request_text),
        capabilities=_facts(original, "$.solution.key_capabilities", "key_capabilities"),
        functional_requirements=_facts(
            original, "$.requirements.functional_requirements", "functional_requirements"
        ),
        non_functional_requirements=_facts(
            original, "$.requirements.non_functional_requirements", "non_functional_requirements"
        ),
        acceptance_criteria=_facts(
            original, "$.requirements.acceptance_criteria", "acceptance_criteria"
        )
        + _facts(original, "$.sow.acceptance", "acceptance"),
        deliverables=_facts(original, "$.sow.deliverables", "deliverables"),
        in_scope=_facts(original, "$.sow.in_scope", "in_scope"),
        out_of_scope=_facts(original, "$.sow.out_of_scope", "out_of_scope"),
        blocking_dependencies=(
            _facts(original, "$.delivery_plan.dependencies", "dependencies")
            + _facts(original, "$.solution.dependencies", "dependencies")
            + _facts(original, "$.sow.dependencies", "dependencies")
        ),
        workstreams=_facts(original, "$.delivery_plan.workstreams", "workstreams"),
        assumptions=(
            _facts(original, "$.discovery.assumptions", "assumptions")
            + _facts(original, "$.solution.assumptions", "assumptions")
            + _facts(original, "$.sow.assumptions", "assumptions")
        ),
        unknowns=(
            _facts(original, "$.discovery.unknowns", "unknowns")
            + _facts(original, "$.requirements.open_questions", "open_questions")
        ),
        prompt_facts=_prompt_facts(original),
    )


def _facts(original: Mapping[str, Any], path: str, field_name: str) -> tuple[SourceFact, ...]:
    values = _list_at(original, path)
    facts: list[SourceFact] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            continue
        facts.append(
            SourceFact(
                json_path=f"{path}[{index}]",
                field_name=field_name,
                text=value,
                is_timeline_or_estimate=looks_like_timeline_or_estimate(value),
            )
        )
    return tuple(facts)


def _prompt_facts(original: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract_id": COPILOT_WORKFLOW_RESPONSE_V1,
        "request": original.get("user_request"),
        "executive_summary": original.get("executive_summary"),
        "functional_requirements": _listed(original, "$.requirements.functional_requirements"),
        "non_functional_requirements": _listed(
            original, "$.requirements.non_functional_requirements"
        ),
        "acceptance_criteria": _listed(original, "$.requirements.acceptance_criteria"),
        "capabilities": _listed(original, "$.solution.key_capabilities"),
        "deliverables": _listed(original, "$.sow.deliverables"),
        "in_scope": _listed(original, "$.sow.in_scope"),
        "out_of_scope": _listed(original, "$.sow.out_of_scope"),
        "blocking_dependencies": (
            _listed(original, "$.delivery_plan.dependencies")
            + _listed(original, "$.solution.dependencies")
            + _listed(original, "$.sow.dependencies")
        ),
        "workstreams": _listed(original, "$.delivery_plan.workstreams"),
        "milestones": _listed(original, "$.delivery_plan.milestones"),
        "delivery_phases": _listed(original, "$.delivery_plan.delivery_phases"),
        "assumptions": _listed(original, "$.discovery.assumptions")
        + _listed(original, "$.solution.assumptions"),
        "unknowns": _listed(original, "$.discovery.unknowns")
        + _listed(original, "$.requirements.open_questions"),
    }


def _listed(original: Mapping[str, Any], path: str) -> list[str]:
    value = resolve_json_path(original, path)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]
