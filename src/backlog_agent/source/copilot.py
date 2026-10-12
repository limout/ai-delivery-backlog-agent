"""Adapter: copilot.workflow_response.v1 → NormalizedProjectSource."""

from __future__ import annotations

from typing import Any, Mapping

from backlog_agent.backlog.generate import _list_at
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.contracts.versions import COPILOT_WORKFLOW_RESPONSE_V1
from backlog_agent.source.model import (
    NormalizedProjectSource,
    SourceFact,
    looks_like_timeline_or_estimate,
)
from backlog_agent.source.request_headings import (
    NarrativeSection,
    parse_labeled_request_outcomes,
    parse_narrative_sections,
)

_REQUEST_PROMPT_TEXT_LIMIT = 2500

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
    request = SourceFact("$.user_request", "user_request", request_text)
    capabilities = _facts(original, "$.solution.key_capabilities", "key_capabilities")
    functional_requirements = _facts(
        original, "$.requirements.functional_requirements", "functional_requirements"
    )
    non_functional_requirements = _facts(
        original, "$.requirements.non_functional_requirements", "non_functional_requirements"
    )
    acceptance_criteria = _facts(
        original, "$.requirements.acceptance_criteria", "acceptance_criteria"
    ) + _facts(original, "$.sow.acceptance", "acceptance")
    deliverables = _facts(original, "$.sow.deliverables", "deliverables")
    in_scope = _facts(original, "$.sow.in_scope", "in_scope")
    out_of_scope = _facts(original, "$.sow.out_of_scope", "out_of_scope")
    blocking_dependencies = (
        _facts(original, "$.delivery_plan.dependencies", "dependencies")
        + _facts(original, "$.solution.dependencies", "dependencies")
        + _facts(original, "$.sow.dependencies", "dependencies")
    )
    workstreams = _facts(original, "$.delivery_plan.workstreams", "workstreams")
    assumptions = (
        _facts(original, "$.discovery.assumptions", "assumptions")
        + _facts(original, "$.solution.assumptions", "assumptions")
        + _facts(original, "$.sow.assumptions", "assumptions")
    )
    unknowns = (
        _facts(original, "$.discovery.unknowns", "unknowns")
        + _facts(original, "$.requirements.open_questions", "open_questions")
    )
    narrative_sections = parse_narrative_sections(request_text)
    return NormalizedProjectSource(
        contract_id=COPILOT_WORKFLOW_RESPONSE_V1,
        original=original,
        allowed_path_prefixes=_ALLOWED_PREFIXES,
        request=request,
        capabilities=capabilities,
        functional_requirements=functional_requirements,
        non_functional_requirements=non_functional_requirements,
        acceptance_criteria=acceptance_criteria,
        deliverables=deliverables,
        in_scope=in_scope,
        out_of_scope=out_of_scope,
        blocking_dependencies=blocking_dependencies,
        workstreams=workstreams,
        assumptions=assumptions,
        unknowns=unknowns,
        narrative_sections=narrative_sections,
        prompt_facts=_prompt_facts(
            original=original,
            request=request,
            capabilities=capabilities,
            functional_requirements=functional_requirements,
            non_functional_requirements=non_functional_requirements,
            acceptance_criteria=acceptance_criteria,
            deliverables=deliverables,
            in_scope=in_scope,
            out_of_scope=out_of_scope,
            blocking_dependencies=blocking_dependencies,
            workstreams=workstreams,
            assumptions=assumptions,
            unknowns=unknowns,
            narrative_sections=narrative_sections,
        ),
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


def _as_prompt_fact(fact: SourceFact) -> dict[str, str]:
    return {"json_path": fact.json_path, "text": fact.text}


def _narrative_prompt_fact(section: NarrativeSection) -> dict[str, Any]:
    return {
        "label": section.label,
        "title": section.title,
        "json_path": section.json_path,
        "excerpt": section.excerpt,
        "excerpt_complete": section.excerpt_complete,
        "evidence_kind": section.evidence_kind,
        "role": "narrative evidence inside $.user_request; not a structured requirement",
    }


def _prompt_facts(
    *,
    original: Mapping[str, Any],
    request: SourceFact,
    capabilities: tuple[SourceFact, ...],
    functional_requirements: tuple[SourceFact, ...],
    non_functional_requirements: tuple[SourceFact, ...],
    acceptance_criteria: tuple[SourceFact, ...],
    deliverables: tuple[SourceFact, ...],
    in_scope: tuple[SourceFact, ...],
    out_of_scope: tuple[SourceFact, ...],
    blocking_dependencies: tuple[SourceFact, ...],
    workstreams: tuple[SourceFact, ...],
    assumptions: tuple[SourceFact, ...],
    unknowns: tuple[SourceFact, ...],
    narrative_sections: tuple[NarrativeSection, ...] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metadata": {"source_contract_id": COPILOT_WORKFLOW_RESPONSE_V1},
        "request": _as_prompt_fact(request),
        "required_functional_coverage": [
            {
                "json_path": fact.json_path,
                "text": fact.text,
                "requires": "one user_story whose provenance cites this exact path and text",
            }
            for fact in functional_requirements
        ],
        "functional_requirements": [_as_prompt_fact(fact) for fact in functional_requirements],
        "non_functional_requirements": [
            _as_prompt_fact(fact) for fact in non_functional_requirements
        ],
        "acceptance_criteria": [_as_prompt_fact(fact) for fact in acceptance_criteria],
        "capabilities": [_as_prompt_fact(fact) for fact in capabilities],
        "deliverables": [_as_prompt_fact(fact) for fact in deliverables],
        "in_scope": [_as_prompt_fact(fact) for fact in in_scope],
        "out_of_scope": [_as_prompt_fact(fact) for fact in out_of_scope],
        "blocking_dependencies": [_as_prompt_fact(fact) for fact in blocking_dependencies],
        "workstreams": [_as_prompt_fact(fact) for fact in workstreams],
        "milestones": [
            _as_prompt_fact(fact)
            for fact in _facts(original, "$.delivery_plan.milestones", "milestones")
        ],
        "delivery_phases": [
            _as_prompt_fact(fact)
            for fact in _facts(original, "$.delivery_plan.delivery_phases", "delivery_phases")
        ],
        "assumptions": [_as_prompt_fact(fact) for fact in assumptions],
        "unknowns": [_as_prompt_fact(fact) for fact in unknowns],
    }
    outcomes = [section for section in narrative_sections if section.evidence_kind != "context"]
    context = [section for section in narrative_sections if section.evidence_kind == "context"]
    if outcomes:
        payload["narrative_sections"] = [_narrative_prompt_fact(section) for section in outcomes]
    if context:
        payload["narrative_context"] = [_narrative_prompt_fact(section) for section in context]
    if narrative_sections and len(request.text) > _REQUEST_PROMPT_TEXT_LIMIT:
        payload["request"] = {
            "json_path": "$.user_request",
            "character_count": len(request.text),
            "full_text_omitted_from_prompt": True,
            "note": (
                "The full request is retained for validation. Labelled sections appear once "
                "as exact excerpts in narrative_sections and narrative_context. Do not invent "
                "json paths or treat those excerpts as structured requirements."
            ),
        }
    headings = parse_labeled_request_outcomes(request.text)
    if headings:
        payload["labeled_request_outcomes"] = [
            {
                "label": item.label,
                "title": item.title,
                "source": "heading inside $.user_request; not an independently addressable json_path",
            }
            for item in headings
        ]
    executive = original.get("executive_summary")
    if isinstance(executive, str) and executive.strip():
        payload["executive_summary"] = {
            "json_path": "$.executive_summary",
            "text": executive,
        }
    return payload
