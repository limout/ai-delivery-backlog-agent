"""Heuristic labeled FR/NFR headings inside a free-text request.

These are not independently addressable JSON paths and must never be used as
structured provenance. Token matching does not prove semantic coverage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from backlog_agent.backlog.ids import normalize_source_text

_HEADING = re.compile(
    r"^(?:#{2,6}\s*|\*\*)?((?:FR|NFR)[-–]?\s*\d+)\s*[:.—–-]\s*(.+?)\s*\**\s*$",
    re.MULTILINE | re.IGNORECASE,
)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "by",
        "can",
        "customer",
        "customers",
        "existing",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "online",
        "or",
        "shall",
        "system",
        "the",
        "their",
        "to",
        "user",
        "users",
        "via",
        "with",
    }
)


@dataclass(frozen=True)
class LabeledRequestOutcome:
    label: str
    title: str


@dataclass(frozen=True)
class NarrativeSection:
    """Exact labelled span inside ``$.user_request``.

    ``excerpt`` is a verbatim slice of the request, not a summary. ``json_path``
    stays ``$.user_request``. This is not a structured requirement.
    """

    label: str
    title: str
    excerpt: str
    evidence_kind: str
    json_path: str = "$.user_request"
    excerpt_complete: bool = True


_ATX_HEADING = re.compile(r"^(#{2,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_LABEL_IN_HEADING = re.compile(
    r"^((?:FR|NFR)[-–]?\s*\d+)\s*[:.—–-]\s*(.+)$",
    re.IGNORECASE,
)
_UNCONFIRMED_SECTION = re.compile(
    r"(?i)(?:"
    r"not been defined|have not been|has not been|unspecified|unconfirmed|"
    r"not yet (?:defined|confirmed|specified|known)|to be confirmed|"
    r"are clarified|rather than assigning arbitrary"
    r")"
)
_MAX_NARRATIVE_SECTIONS = 40
_OUTCOME_EXCERPT_CAP = 1500
_CONTEXT_EXCERPT_CAP = 900


def _bounded_verbatim_excerpt(span: str, cap: int) -> tuple[str, bool]:
    if len(span) <= cap:
        return span, True
    cut = span.rfind("\n", 0, cap)
    if cut < 80:
        cut = cap
    return span[:cut].rstrip(), False


def parse_narrative_sections(text: str) -> tuple[NarrativeSection, ...]:
    """Return exact ATX-heading spans from a free-text request.

    FR/NFR headings become outcome, constraint, or unconfirmed evidence.
    Other headings are context. Nothing here is a structured requirement path.
    """

    if not text:
        return ()
    matches = list(_ATX_HEADING.finditer(text))
    sections: list[NarrativeSection] = []
    seen_labels: set[str] = set()
    for index, match in enumerate(matches):
        if len(sections) >= _MAX_NARRATIVE_SECTIONS:
            break
        heading = " ".join(match.group(2).split())
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if not body:
            continue
        span = text[match.start() : end].strip()
        label_match = _LABEL_IN_HEADING.match(heading)
        if label_match:
            label = re.sub(r"[\s–]+", "", label_match.group(1).upper()).replace("—", "-")
            if label in seen_labels:
                continue
            seen_labels.add(label)
            title = " ".join(label_match.group(2).split())[:160]
            has_obligation = bool(re.search(r"(?m)^\s*-\s+\S", body))
            if _UNCONFIRMED_SECTION.search(body) and not has_obligation:
                kind = "unconfirmed"
            elif label.startswith("NFR"):
                kind = "constraint"
            else:
                kind = "outcome"
            cap = _OUTCOME_EXCERPT_CAP
        else:
            label = ""
            title = heading[:160]
            kind = "context"
            cap = _CONTEXT_EXCERPT_CAP
        excerpt, complete = _bounded_verbatim_excerpt(span, cap)
        if not excerpt or excerpt not in text:
            continue
        sections.append(
            NarrativeSection(
                label=label,
                title=title,
                excerpt=excerpt,
                evidence_kind=kind,
                excerpt_complete=complete,
            )
        )
    return tuple(sections)


def parse_labeled_request_outcomes(text: str) -> tuple[LabeledRequestOutcome, ...]:
    """Return genuine labeled FR/NFR markdown headings. Ignores ordinary bullets."""

    seen: set[str] = set()
    items: list[LabeledRequestOutcome] = []
    for match in _HEADING.finditer(text or ""):
        label = re.sub(r"[\s–]+", "", match.group(1).upper()).replace("—", "-")
        title = " ".join(match.group(2).split())
        if not title or label in seen:
            continue
        seen.add(label)
        items.append(LabeledRequestOutcome(label=label, title=title[:160]))
    return tuple(items)


_TOKEN_EDGE = re.compile(r"^[^a-z0-9]+|[^a-z0-9]+$")


def _bare_token(token: str) -> str:
    return _TOKEN_EDGE.sub("", token)


def distinctive_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in normalize_source_text(text).split():
        token = _bare_token(raw)
        if len(token) >= 4 and token not in _STOPWORDS:
            tokens.add(token)
    return tokens


_LISTING_TOKENS = frozenset({"list", "lists", "listing", "listings", "listed"})
_DISPLAY_TOKENS = _LISTING_TOKENS | frozenset(
    {
        "browse",
        "browses",
        "browsing",
        "display",
        "displayed",
        "displays",
        "review",
        "reviewed",
        "reviewing",
        "reviews",
        "show",
        "showing",
        "shows",
        "view",
        "viewed",
        "viewing",
        "views",
    }
)


def _related_token(left: str, right: str) -> bool:
    """True for the same word or a shared stem such as authenticate/authentication.

    Short shared prefixes are not enough, so unrelated words like integration and
    integrity do not count. This does not prove the outcome is implemented.
    """

    if left == right:
        return True
    if min(len(left), len(right)) < 8:
        return False
    if left.startswith(right) or right.startswith(left):
        return True
    shared = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        shared += 1
    return shared >= 8


def _plural_stem(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 5:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
        return token[:-1]
    return token


def _same_word(left: str, right: str) -> bool:
    if _related_token(left, right):
        return True
    left_stem = _plural_stem(left)
    right_stem = _plural_stem(right)
    return len(left_stem) >= 4 and left_stem == right_stem


def heading_appears_represented(title: str, blob: str) -> bool:
    """True when enough distinctive heading tokens appear in ``blob``.

    A heading such as "Policy Listing" can match a story that reviews or views
    those policies. Shared prefixes shorter than eight characters, such as
    integration and integrity, do not match. This is a recall heuristic, not
    proof that the outcome is implemented.
    """

    tokens = distinctive_tokens(title)
    if not tokens:
        return True
    haystack = distinctive_tokens(blob)
    matched = {token for token in tokens if any(_same_word(token, other) for other in haystack)}
    listing = tokens & _LISTING_TOKENS
    if listing and matched and any(token in _DISPLAY_TOKENS for token in haystack):
        matched |= listing
    needed = 2 if len(tokens) >= 2 else 1
    return len(matched) >= needed
