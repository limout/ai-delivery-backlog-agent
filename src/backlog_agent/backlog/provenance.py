"""Collapse semantically equivalent canonical provenance without dropping sources.

One logical entry is kept for excerpts that state the same fact. Every distinct
source path stays on that entry. Records that add or change a subject token
stay separate. The comparison does not know about a domain or a section name.
"""

from __future__ import annotations

import re

from backlog_agent.backlog.models import SourceReference

_TOKEN = re.compile(r"[a-z0-9]+")
_STATUS_TOKENS = frozenset(
    {
        "approval",
        "approve",
        "approved",
        "approving",
        "been",
        "business",
        "can",
        "confirm",
        "confirmation",
        "confirmed",
        "confirming",
        "could",
        "expected",
        "final",
        "have",
        "may",
        "must",
        "need",
        "needed",
        "needs",
        "off",
        "pending",
        "please",
        "should",
        "sign",
        "signoff",
        "whether",
        "will",
        "would",
        "about",
        "all",
        "also",
        "and",
        "any",
        "are",
        "but",
        "each",
        "for",
        "from",
        "had",
        "has",
        "into",
        "its",
        "onto",
        "our",
        "out",
        "over",
        "per",
        "than",
        "that",
        "the",
        "their",
        "then",
        "they",
        "this",
        "under",
        "via",
        "was",
        "were",
        "with",
        "you",
        "your",
    }
)


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _subject_tokens(text: str) -> frozenset[str]:
    stems = {
        _stem(token)
        for token in _TOKEN.findall((text or "").casefold())
        if len(token) >= 3 and token not in _STATUS_TOKENS and _stem(token) not in _STATUS_TOKENS
    }
    return frozenset(stems)


def _normalized_excerpt(text: str | None) -> str:
    return " ".join(_TOKEN.findall((text or "").casefold()))


def _reference_key(reference: SourceReference) -> tuple[str, str, str, str]:
    return (
        reference.contract_id,
        reference.json_path,
        reference.field_name or "",
        _normalized_excerpt(reference.excerpt),
    )


def _same_fact(left: str | None, right: str | None) -> bool:
    """True when two excerpts state the same subject, not merely a shared word."""

    if _normalized_excerpt(left) == _normalized_excerpt(right):
        return True
    left_tokens = _subject_tokens(left or "")
    right_tokens = _subject_tokens(right or "")
    return len(left_tokens) >= 2 and left_tokens == right_tokens


def _flatten(references: list[SourceReference]) -> list[SourceReference]:
    flat: list[SourceReference] = []
    for reference in references:
        flat.append(reference.model_copy(update={"equivalent_references": []}))
        flat.extend(_flatten(list(reference.equivalent_references)))
    return flat


def collapse_equivalent_provenance(references: list[SourceReference]) -> list[SourceReference]:
    """Return one entry per equivalent fact, preserving first-seen order.

    The first excerpt in a group stays on the entry. Later distinct sources are
    stored on ``equivalent_references`` in the order they appeared. Repeating
    the call does not change the grouping or the order.
    """

    groups: list[list[SourceReference]] = []
    for reference in _flatten(references):
        matched = next(
            (
                group
                for group in groups
                if _same_fact(group[0].excerpt, reference.excerpt)
            ),
            None,
        )
        if matched is None:
            groups.append([reference])
            continue
        if any(_reference_key(reference) == _reference_key(existing) for existing in matched):
            continue
        matched.append(reference)

    collapsed: list[SourceReference] = []
    for group in groups:
        primary, *rest = group
        collapsed.append(
            primary.model_copy(
                update={
                    "equivalent_references": [
                        item.model_copy(update={"equivalent_references": []}) for item in rest
                    ]
                }
            )
        )
    return collapsed
