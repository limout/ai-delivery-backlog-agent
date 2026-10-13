"""Tracker-independent source facts extracted by an input adapter."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from backlog_agent.backlog.ids import normalize_source_text
from backlog_agent.source.request_headings import NarrativeSection

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
    narrative_sections: tuple[NarrativeSection, ...] = ()
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


ClaimStatus = str  # "established" | "unresolved" | "conflict"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_UNRESOLVED_MARK = re.compile(
    r"\b("
    r"unknown|unconfirmed|unspecified|unclear|pending|"
    r"not yet|not confirmed|not granted|not specified|not available|"
    r"to be confirmed|to be determined|\btbd\b|"
    r"need to confirm|need to verify|investigate whether|"
    r"if available|open question|cannot participate|no access"
    r")\b",
    re.IGNORECASE,
)
_ESTABLISHED_MARK = re.compile(
    r"\b("
    r"(?:are|is|will be)\s+available|"
    r"(?:have|has)\s+access|"
    r"can participate|will participate|"
    r"(?:are|is|will be)\s+provided|"
    r"(?:is|are)\s+granted|"
    r"(?:is|are)\s+confirmed|"
    r"in place"
    r")\b",
    re.IGNORECASE,
)
_CLAIM_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "can",
        "for",
        "from",
        "have",
        "has",
        "if",
        "in",
        "is",
        "key",
        "not",
        "of",
        "on",
        "or",
        "regular",
        "the",
        "to",
        "whether",
        "will",
        "with",
        "yet",
    }
)


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text or "") if part.strip()]
    return parts or ([text.strip()] if text and text.strip() else [])


def statement_polarity(text: str) -> str | None:
    """Return established, unresolved, or None when the sentence has no status cue."""

    if _UNRESOLVED_MARK.search(text or ""):
        return "unresolved"
    if _ESTABLISHED_MARK.search(text or ""):
        return "established"
    return None


def claim_content_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in normalize_source_text(text).split():
        token = raw.strip(".,;:!?\"'`()[]{}")
        if len(token) >= 3 and token not in _CLAIM_STOPWORDS:
            tokens.add(token)
    return tokens


def claims_are_related(claim: str, statement: str) -> bool:
    claim_tokens = claim_content_tokens(claim)
    if not claim_tokens:
        return False
    matched = claim_tokens & claim_content_tokens(statement)
    if len(matched) < 2:
        return False
    return len(matched) / len(claim_tokens) >= 0.5


def iter_source_statements(source: NormalizedProjectSource) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for sentence in _sentences(source.request.text):
        rows.append((source.request.json_path, sentence))
    executive = source.original.get("executive_summary")
    if isinstance(executive, str) and executive.strip():
        for sentence in _sentences(executive):
            rows.append(("$.executive_summary", sentence))
    groups: tuple[tuple[SourceFact, ...], ...] = (
        source.capabilities,
        source.functional_requirements,
        source.non_functional_requirements,
        source.acceptance_criteria,
        source.deliverables,
        source.in_scope,
        source.out_of_scope,
        source.blocking_dependencies,
        source.workstreams,
        source.assumptions,
        source.unknowns,
    )
    for group in groups:
        for fact in group:
            if fact.text.strip():
                rows.append((fact.json_path, fact.text))
    return tuple(rows)


def classify_claim(
    text: str,
    source: NormalizedProjectSource,
    *,
    skip_path: str | None = None,
) -> ClaimStatus:
    """Reconcile a candidate fact with the rest of the source envelope.

    Established when the claim or a related statement confirms it. Unresolved when
    the claim or a related statement leaves it open. Conflict when both polarities
    appear for the same subject. A listed dependency with no confirming evidence
    stays unresolved; confirmation of one fact does not resolve a more specific
    remaining unknown.
    """

    return classify_claim_against_statements(
        text, iter_source_statements(source), skip_path=skip_path
    )


def classify_claim_against_statements(
    text: str,
    statements: Iterable[tuple[str, str]],
    *,
    skip_path: str | None = None,
) -> ClaimStatus:
    own = statement_polarity(text)
    related_established = own == "established"
    related_unresolved = own == "unresolved"
    for path, statement in statements:
        if skip_path and path == skip_path:
            continue
        if statement.strip() == (text or "").strip():
            continue
        polarity = statement_polarity(statement)
        if polarity is None or not claims_are_related(text, statement):
            continue
        if polarity == "established":
            related_established = True
        elif polarity == "unresolved":
            related_unresolved = True
    if related_established and related_unresolved:
        return "conflict"
    if related_established:
        return "established"
    return "unresolved"


def annotate_prompt_evidence_status(source: NormalizedProjectSource) -> None:
    """Attach evidence_status to prompt fact groups without changing json_path values."""

    for group in ("blocking_dependencies", "unknowns"):
        entries = source.prompt_facts.get(group)
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "")
            path = str(item.get("json_path") or "") or None
            item["evidence_status"] = classify_claim(text, source, skip_path=path)
