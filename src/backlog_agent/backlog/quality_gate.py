"""Acceptance contract for LLM-proposed backlog candidates.

Three separate questions are answered for every candidate tree, and they must
not be conflated:

``validation_passed``
    The proposal parsed, satisfied the structural rules (parents, cycles,
    provenance, hard coverage of functional requirements) and materialized into
    a ``CanonicalBacklogV1``. This is a *hard* gate. A failing candidate is
    never returned.

``blocking quality defects``
    Soft findings that nevertheless mean required work is demonstrably missing
    or unverified. They are defined here, once, by ``BLOCKING_QUALITY_CODES``
    and the narrow message predicates below. Today they are:

    * an in-scope non-functional requirement that no Story verifies and no
      implementation Task covers (``UNCOVERED_NON_FUNCTIONAL_REQUIREMENT``);
    * a Story whose acceptance criteria verify only part of the outcomes of the
      source requirement it cites (``MISSING_VERIFICATION_COVERAGE`` with the
      "does not verify every outcome" message);
    * an uncovered functional requirement (``UNCOVERED_FUNCTIONAL_REQUIREMENT``).
      Hard validation already rejects this, it is listed so the contract holds
      if that rule is ever relaxed;
    * a **material hierarchy defect**: a hierarchy or parenting finding that
      the detector itself rated high confidence and marked with
      ``MATERIAL_HIERARCHY_PREFIX``. The detectors only set the marker when the
      evidence is strong:

      - a Feature whose Stories form at least three separate capability
        clusters, no cluster holds more than half of them, and the Feature's own
        title and source excerpt cover fewer than half of them;
      - a clarification Task whose own unresolved subject shares nothing with its
        parent Story while exactly one other Story clearly shares it;
      - clarification Tasks that repeat one unresolved question, either by citing
        the same source entry or by asking it from different entries. One
        appropriately placed Task per unresolved question is the valid shape.

    A legitimately unresolved requirement is **not** a defect: one relevant
    clarification Task represents the uncertainty and never blocks acceptance.

``advisory quality findings``
    Everything else: weaker feature grouping signals (compound titles, partial
    provenance or description overlap, only two clusters), uncertain parent
    assignments (the Task shares some subject with its parent), oversized stories,
    overlapping responsibilities, missing implementation detail, empty acceptance
    criteria, uncertainty and prerequisite notes. They are reported, and some of
    them may spend the single repair call, but they never decide acceptance. They
    rest on token heuristics and a candidate must not be rejected for them.

``accepted``
    A candidate is *accepted* only when it passed hard validation **and** carries
    no blocking quality defect. A repaired candidate is held to the same rule.
    A candidate that is returned despite a blocking defect (because it is the
    only structurally valid tree available) is reported with ``accepted=false``
    and a ``QUALITY_REPAIR_REJECTED`` warning that names the unresolved defects.
"""

from __future__ import annotations

from collections.abc import Iterable

from backlog_agent.findings.models import Finding, FindingCode

BLOCKING_QUALITY_CODES: frozenset[FindingCode] = frozenset(
    {
        FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT,
        FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT,
    }
)

_PARTIAL_VERIFICATION_MARKER = "does not verify every outcome"

#: Detectors start the message of a high-confidence hierarchy or parenting finding with this
#: marker. Weaker findings of the same code never carry it and stay advisory.
MATERIAL_HIERARCHY_PREFIX = "Material hierarchy defect: "

MATERIAL_HIERARCHY_CODES: frozenset[FindingCode] = frozenset(
    {
        FindingCode.FEATURE_CHILD_MISALIGNMENT,
        FindingCode.AMBIGUOUS_DECOMPOSITION,
        FindingCode.OVER_DECOMPOSITION,
    }
)


def material_hierarchy_message(text: str) -> str:
    """Mark ``text`` as a high-confidence material hierarchy defect."""

    return MATERIAL_HIERARCHY_PREFIX + text


def is_material_hierarchy_finding(finding: Finding) -> bool:
    return (
        finding.code in MATERIAL_HIERARCHY_CODES
        and finding.message.startswith(MATERIAL_HIERARCHY_PREFIX)
    )


def is_blocking_quality_finding(finding: Finding) -> bool:
    """True when ``finding`` means required work is missing or unverified, or the hierarchy
    has a high-confidence material defect."""

    if finding.code in BLOCKING_QUALITY_CODES:
        return True
    if is_material_hierarchy_finding(finding):
        return True
    return (
        finding.code is FindingCode.MISSING_VERIFICATION_COVERAGE
        and _PARTIAL_VERIFICATION_MARKER in finding.message
    )


def blocking_quality_findings(findings: Iterable[Finding]) -> list[Finding]:
    return [finding for finding in findings if is_blocking_quality_finding(finding)]


def describe_blocking_findings(findings: Iterable[Finding], *, limit: int = 4) -> str:
    """Short, deterministic summary naming each defect by its source excerpt."""

    parts: list[str] = []
    for finding in blocking_quality_findings(findings):
        if is_material_hierarchy_finding(finding):
            detail = finding.message[len(MATERIAL_HIERARCHY_PREFIX) :]
            label = f"{finding.code.value} ({' '.join(detail.split())[:140]})"
        else:
            excerpt = next((ref.excerpt for ref in finding.source_references if ref.excerpt), None)
            label = finding.code.value if not excerpt else f"{finding.code.value} ({excerpt[:120]})"
        if label not in parts:
            parts.append(label)
    shown = parts[:limit]
    extra = len(parts) - len(shown)
    text = "; ".join(shown)
    return f"{text}; and {extra} more" if extra > 0 else text
