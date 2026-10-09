"""Tracker-independent source facts extracted by an input adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from backlog_agent.backlog.ids import normalize_source_text

_TIMELINE_OR_ESTIMATE = (
    "week",
    "weeks",
    "deadline",
    "launch within",
    "person-day",
    "person-days",
    "estimate",
)


@dataclass(frozen=True)
class SourceFact:
    json_path: str
    field_name: str | None
    text: str
    is_timeline_or_estimate: bool = False


@dataclass(frozen=True)
class NormalizedProjectSource:
    """Generic project facts. Decomposition must not assume a Copilot field layout."""

    contract_id: str
    original: Mapping[str, Any]
    allowed_path_prefixes: tuple[str, ...]
    request: SourceFact
    capabilities: tuple[SourceFact, ...] = ()
    functional_requirements: tuple[SourceFact, ...] = ()
    non_functional_requirements: tuple[SourceFact, ...] = ()
    acceptance_criteria: tuple[SourceFact, ...] = ()
    deliverables: tuple[SourceFact, ...] = ()
    in_scope: tuple[SourceFact, ...] = ()
    out_of_scope: tuple[SourceFact, ...] = ()
    blocking_dependencies: tuple[SourceFact, ...] = ()
    workstreams: tuple[SourceFact, ...] = ()
    assumptions: tuple[SourceFact, ...] = ()
    unknowns: tuple[SourceFact, ...] = ()
    prompt_facts: dict[str, Any] = field(default_factory=dict)


def looks_like_timeline_or_estimate(text: str) -> bool:
    normalized = normalize_source_text(text)
    return any(token in normalized for token in _TIMELINE_OR_ESTIMATE)


def scope_line_applies(source_text: str, scope_line: str) -> bool:
    normalized_source = normalize_source_text(source_text)
    normalized_scope = normalize_source_text(scope_line)
    if not normalized_source or not normalized_scope:
        return False
    if normalized_source == normalized_scope:
        return True
    return len(normalized_scope) >= 12 and normalized_scope in normalized_source


def classify_scope(source: NormalizedProjectSource, text: str) -> str:
    in_hits = [fact for fact in source.in_scope if scope_line_applies(text, fact.text)]
    out_hits = [fact for fact in source.out_of_scope if scope_line_applies(text, fact.text)]
    if in_hits and out_hits:
        return "conflict"
    if out_hits:
        return "out"
    return "in"


def fact_texts(facts: tuple[SourceFact, ...]) -> set[str]:
    return {normalize_source_text(fact.text) for fact in facts if fact.text.strip()}
