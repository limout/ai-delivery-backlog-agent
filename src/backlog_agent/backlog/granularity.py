"""Planning heuristics for backlog granularity. Not effort estimates."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from backlog_agent.backlog.ids import normalize_source_text
from backlog_agent.backlog.models import WorkItemType
from backlog_agent.findings.models import Finding, FindingCode, FindingSeverity
from backlog_agent.source.model import NormalizedProjectSource

_ACTION_VERBS = (
    "authenticate",
    "authorize",
    "search",
    "create",
    "update",
    "delete",
    "submit",
    "notify",
    "export",
    "import",
    "review",
    "approve",
    "pay",
    "schedule",
    "assign",
    "escalate",
    "integrate",
    "encrypt",
    "audit",
    "upload",
    "download",
    "track",
    "refund",
    "publish",
    "archive",
    "apply",
)

_IMPLEMENTATION_SIGNALS = (
    "integrat",
    "external",
    "webhook",
    "encrypt",
    "authoriz",
    "timeout",
    "retry",
    "audit",
    "observab",
    "metric",
    "logging",
    "fallback",
    "error handling",
    "exception",
    "sandbox",
    "third-party",
    "third party",
    "queue",
    "schema",
)

_BROAD_PHRASES = (
    "end-to-end",
    "end to end",
    "the whole",
    "everything",
    "all remaining",
    "entire journey",
    "complete journey",
    "the full ",
    "the entire ",
    "implement the feature",
    "build the solution",
    "build the system",
    "implement the system",
    "handle everything",
)

_BROAD_PREFIX = re.compile(
    r"^(implement|build|develop|handle|support|deliver)\b",
    re.I,
)
_STEP_TITLE = re.compile(r"^(step|part|phase)\s*\d+\b", re.I)
_TRAILING_INDEX = re.compile(r"\s+\d+$")
_FUNCTIONAL_REQUIREMENT_PATH = "$.requirements.functional_requirements"
_ACCEPTANCE_CRITERION_PREFIXES = (
    "$.requirements.acceptance_criteria",
    "$.sow.acceptance",
)
_GENERIC_CONTEXT_WORDS = frozenset(
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
        "details",
        "eligible",
        "existing",
        "for",
        "from",
        "i",
        "in",
        "into",
        "is",
        "let",
        "member",
        "my",
        "of",
        "on",
        "online",
        "or",
        "portal",
        "request",
        "shall",
        "so",
        "system",
        "that",
        "the",
        "their",
        "to",
        "user",
        "users",
        "using",
        "via",
        "want",
        "web",
        "when",
        "with",
    }
)


def _is_functional_requirement_path(json_path: str) -> bool:
    return json_path == _FUNCTIONAL_REQUIREMENT_PATH or json_path.startswith(
        _FUNCTIONAL_REQUIREMENT_PATH + "["
    )


def functional_requirement_paths(item: Any) -> list[str]:
    paths: list[str] = []
    for reference in getattr(item, "provenance", ()):
        path = getattr(reference, "json_path", "")
        if _is_functional_requirement_path(path):
            paths.append(path)
    return paths


def _is_acceptance_criterion_path(json_path: str) -> bool:
    return any(
        json_path == prefix or json_path.startswith(prefix + "[")
        for prefix in _ACCEPTANCE_CRITERION_PREFIXES
    )


def acceptance_criterion_paths(item: Any) -> list[str]:
    paths: list[str] = []
    for reference in getattr(item, "provenance", ()):
        path = getattr(reference, "json_path", "")
        if _is_acceptance_criterion_path(path):
            paths.append(path)
    return paths


def _significant_tokens(text: str) -> list[str]:
    return [
        token
        for token in normalize_source_text(text).split()
        if token and token not in _GENERIC_CONTEXT_WORDS
    ]


def owned_title_phrases(item: Any) -> set[str]:
    """Distinctive adjacent token pairs from the story title. Not generic context."""

    tokens = _significant_tokens(getattr(item, "title", "") or "")
    phrases: set[str] = set()
    for index in range(len(tokens) - 1):
        left, right = tokens[index], tokens[index + 1]
        if min(len(left), len(right)) < 4:
            continue
        phrases.add(f"{left} {right}")
    return phrases


def _story_body_tokens(item: Any) -> list[str]:
    parts = [getattr(item, "description", "") or ""]
    parts.extend(list(getattr(item, "acceptance_criteria", ()) or ()))
    return normalize_source_text(" ".join(parts)).split()


def _contains_phrase_tokens(haystack: list[str], phrase: str) -> bool:
    needle = phrase.split()
    length = len(needle)
    if length == 0 or length > len(haystack):
        return False
    for index in range(len(haystack) - length + 1):
        window = haystack[index : index + length]
        if window == needle:
            return True
        if window[:-1] == needle[:-1] and (
            window[-1] == needle[-1] + "s" or needle[-1] == window[-1] + "s"
        ):
            return True
    return False


def leaked_owned_phrases(owner: Any, other: Any) -> list[str]:
    """Title phrases owned by ``owner`` that appear in ``other``'s description or AC."""

    phrases = owned_title_phrases(owner)
    if not phrases:
        return []
    body = _story_body_tokens(other)
    if not body:
        return []
    return [phrase for phrase in sorted(phrases) if _contains_phrase_tokens(body, phrase)]


def description_leaks_owned_action(owner: Any, other: Any) -> bool:
    """True when ``other``'s description or AC includes a title phrase owned by ``owner``."""

    return bool(leaked_owned_phrases(owner, other))


def _clip_evidence(text: str, limit: int = 180) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _evidence_for_phrase(item: Any, phrase: str) -> str:
    candidates = [getattr(item, "description", "") or ""]
    candidates.extend(list(getattr(item, "acceptance_criteria", ()) or ()))
    for text in candidates:
        if _contains_phrase_tokens(normalize_source_text(text).split(), phrase):
            return _clip_evidence(text)
    return _clip_evidence(getattr(item, "description", "") or "")


def action_verbs(text: str) -> set[str]:
    normalized = normalize_source_text(text)
    return {verb for verb in _ACTION_VERBS if re.search(rf"\b{re.escape(verb)}\b", normalized)}


def looks_oversized_story(item: Any) -> bool:
    """True when a story appears to bundle multiple independently testable outcomes."""

    fr_paths = set(functional_requirement_paths(item))
    if len(fr_paths) >= 2:
        return True
    criteria = list(getattr(item, "acceptance_criteria", ()) or ())
    if len(criteria) >= 4:
        return True
    blob = " ".join(
        [
            getattr(item, "title", ""),
            getattr(item, "description", ""),
            *criteria,
        ]
    )
    verbs = action_verbs(blob)
    if len(verbs) >= 3:
        return True
    return False


def looks_like_implementation_work(item: Any) -> bool:
    blob = normalize_source_text(
        " ".join(
            [
                getattr(item, "title", ""),
                getattr(item, "description", ""),
                *list(getattr(item, "acceptance_criteria", ()) or ()),
                *list(getattr(item, "test_requirements", ()) or ()),
            ]
        )
    )
    if any(signal in blob for signal in _IMPLEMENTATION_SIGNALS):
        return True
    if len(list(getattr(item, "acceptance_criteria", ()) or ())) >= 2:
        return True
    if len(list(getattr(item, "test_requirements", ()) or ())) >= 2:
        return True
    return False


def task_restates_parent(item: Any, parent: Any | None) -> bool:
    if parent is None:
        return False
    title = normalize_source_text(getattr(item, "title", ""))
    parent_title = normalize_source_text(getattr(parent, "title", ""))
    if not title or not parent_title:
        return False
    if title == parent_title:
        return True
    stripped = _BROAD_PREFIX.sub("", title).strip()
    stripped = re.sub(r"^(the|a|an)\s+", "", stripped)
    return stripped == parent_title


def looks_broad_task(item: Any, parent: Any | None = None) -> bool:
    title = getattr(item, "title", "")
    description = getattr(item, "description", "")
    normalized_title = normalize_source_text(title)
    if any(phrase in normalized_title or phrase in normalize_source_text(description) for phrase in _BROAD_PHRASES):
        return True
    if task_restates_parent(item, parent):
        return True
    words = normalized_title.split()
    if _BROAD_PREFIX.search(normalized_title) and len(words) <= 5:
        return True
    return False


def looks_over_decomposed(task_titles: list[str]) -> bool:
    if len(task_titles) >= 8:
        return True
    if len(task_titles) >= 5 and sum(1 for title in task_titles if _STEP_TITLE.search(title)) >= 3:
        return True
    stripped = [_TRAILING_INDEX.sub("", normalize_source_text(title)).strip() for title in task_titles]
    counts = Counter(title for title in stripped if title)
    if any(count >= 4 for count in counts.values()):
        return True
    return False


def record_granularity_findings(
    _source: NormalizedProjectSource,
    backlog: Any,
    findings: Any,
    *,
    proposal: Any | None = None,
) -> None:
    """Emit heuristic findings. Does not invent effort or fail validation."""

    local_id_by_canonical = _local_id_by_canonical(backlog, proposal)
    by_id = {item.canonical_id: item for item in backlog.items}
    _record_cross_story_findings(list(backlog.items), findings, local_id_by_canonical)
    for item in backlog.items:
        if item.type is WorkItemType.USER_STORY:
            _record_story_granularity(item, by_id, findings)
        elif item.type is WorkItemType.TASK:
            parent = by_id.get(item.parent_id) if item.parent_id else None
            if looks_broad_task(item, parent):
                findings.add(
                    Finding(
                        severity=FindingSeverity.WARNING,
                        code=FindingCode.BROAD_TASK,
                        message=(
                            "Task title or description looks too broad for a few days of one "
                            "contributor's work. Split into concrete implementation or verification "
                            "activities. This is a sizing heuristic, not an estimate."
                        ),
                        canonical_ids=[item.canonical_id],
                        source_references=list(item.provenance),
                    )
                )


def _local_id_by_canonical(backlog: Any, proposal: Any | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if proposal is None:
        return mapping
    for proposed, canonical in zip(proposal.items, backlog.items, strict=True):
        local_id = getattr(proposed, "local_id", "") or ""
        if local_id:
            mapping[canonical.canonical_id] = local_id
    return mapping


def _item_local_id(item: Any, local_id_by_canonical: dict[str, str]) -> str:
    canonical_id = getattr(item, "canonical_id", "") or ""
    mapped = local_id_by_canonical.get(canonical_id, "")
    if mapped:
        return mapped
    return str(getattr(item, "local_id", "") or "")


def _user_stories(items: list[Any]) -> list[Any]:
    return [item for item in items if getattr(item, "type", None) is WorkItemType.USER_STORY]


def _record_cross_story_findings(
    items: list[Any],
    findings: Any,
    local_id_by_canonical: dict[str, str],
) -> None:
    stories = _user_stories(items)
    if len(stories) < 2:
        return
    _record_duplicate_acceptance_criteria(stories, findings, local_id_by_canonical)
    _record_overlapping_responsibilities(stories, findings, local_id_by_canonical)


def _record_duplicate_acceptance_criteria(
    stories: list[Any],
    findings: Any,
    local_id_by_canonical: dict[str, str],
) -> None:
    by_path: dict[str, list[Any]] = defaultdict(list)
    by_text: dict[str, list[Any]] = defaultdict(list)
    for story in stories:
        for path in acceptance_criterion_paths(story):
            by_path[path].append(story)
        seen_text: set[str] = set()
        for criterion in list(getattr(story, "acceptance_criteria", ()) or ()):
            normalized = normalize_source_text(criterion)
            if len(normalized) < 12 or normalized in seen_text:
                continue
            seen_text.add(normalized)
            by_text[normalized].append(story)

    emitted: set[tuple[str, ...]] = set()
    for path, group in by_path.items():
        ids = tuple(sorted({item.canonical_id for item in group}))
        if len(ids) < 2 or ids in emitted:
            continue
        emitted.add(ids)
        findings.add(
            Finding(
                severity=FindingSeverity.WARNING,
                code=FindingCode.DUPLICATE_ACCEPTANCE_CRITERION,
                message=_duplicate_criterion_message(group, ids, local_id_by_canonical, path=path),
                canonical_ids=list(ids),
                local_ids=_local_ids_for(group, ids, local_id_by_canonical),
                source_references=[
                    reference
                    for item in group
                    for reference in list(getattr(item, "provenance", ()) or ())
                ],
            )
        )
    for text, group in by_text.items():
        ids = tuple(sorted({item.canonical_id for item in group}))
        if len(ids) < 2 or ids in emitted:
            continue
        emitted.add(ids)
        findings.add(
            Finding(
                severity=FindingSeverity.WARNING,
                code=FindingCode.DUPLICATE_ACCEPTANCE_CRITERION,
                message=_duplicate_criterion_message(group, ids, local_id_by_canonical, text=text),
                canonical_ids=list(ids),
                local_ids=_local_ids_for(group, ids, local_id_by_canonical),
                source_references=[
                    reference
                    for item in group
                    for reference in list(getattr(item, "provenance", ()) or ())
                ],
            )
        )


def _story_label(item: Any, local_id: str = "") -> str:
    title = getattr(item, "title", "") or "(untitled)"
    identity = f"canonical_id={item.canonical_id}"
    if local_id:
        identity += f", local_id={local_id}"
    return f"{title!r} ({identity})"


def _local_ids_for(
    members: list[Any],
    ids: tuple[str, ...] | list[str],
    local_id_by_canonical: dict[str, str],
) -> list[str]:
    seen: set[str] = set()
    local_ids: list[str] = []
    allowed = set(ids)
    for item in members:
        if item.canonical_id not in allowed or item.canonical_id in seen:
            continue
        seen.add(item.canonical_id)
        local_id = _item_local_id(item, local_id_by_canonical)
        if local_id:
            local_ids.append(local_id)
    return local_ids


def _duplicate_criterion_message(
    members: list[Any],
    ids: tuple[str, ...],
    local_id_by_canonical: dict[str, str],
    *,
    path: str | None = None,
    text: str | None = None,
) -> str:
    unique: list[Any] = []
    seen: set[str] = set()
    for item in members:
        if item.canonical_id not in ids or item.canonical_id in seen:
            continue
        seen.add(item.canonical_id)
        unique.append(item)
    labels = "; ".join(
        _story_label(item, _item_local_id(item, local_id_by_canonical)) for item in unique
    )
    if path:
        excerpts = []
        for item in members:
            for reference in list(getattr(item, "provenance", ()) or ()):
                if getattr(reference, "json_path", "") == path and getattr(reference, "excerpt", None):
                    excerpts.append(_clip_evidence(str(reference.excerpt)))
                    break
            if excerpts:
                break
        criterion = excerpts[0] if excerpts else path
        overlap = f"source path {path} (excerpt: {criterion!r})"
    else:
        overlap = f"normalized text {text!r}"
    return (
        f"Stories {labels} share the same acceptance criterion via {overlap}. "
        f"Canonical IDs: {', '.join(ids)}. "
        "Remediation: keep this criterion on the one story whose outcome it verifies; "
        "remove the duplicate from the other story or stories."
    )


def _overlap_responsibility_message(
    left: Any,
    right: Any,
    leaked_from_left: list[str],
    leaked_from_right: list[str],
    local_id_by_canonical: dict[str, str],
) -> str:
    left_local = _item_local_id(left, local_id_by_canonical)
    right_local = _item_local_id(right, local_id_by_canonical)
    parts: list[str] = [
        f"Stories {_story_label(left, left_local)} and {_story_label(right, right_local)} overlap."
    ]
    if leaked_from_left:
        phrase = leaked_from_left[0]
        parts.append(
            f"{right.title!r} includes action {phrase!r} owned by {left.title!r}. "
            f"Evidence in {right.title!r}: {_evidence_for_phrase(right, phrase)!r}."
        )
    if leaked_from_right:
        phrase = leaked_from_right[0]
        parts.append(
            f"{left.title!r} includes action {phrase!r} owned by {right.title!r}. "
            f"Evidence in {left.title!r}: {_evidence_for_phrase(left, phrase)!r}."
        )
    owner = left if leaked_from_left else right
    leaker = right if leaked_from_left else left
    action = (leaked_from_left or leaked_from_right)[0]
    parts.append(
        f"Remediation: remove {action!r} from {leaker.title!r} description and acceptance "
        f"criteria; keep that outcome on {owner.title!r} only. Split into separate stories "
        "only when the source states independently testable outcomes. Do not invent systems, "
        "local storage, endpoints, or architecture."
    )
    return " ".join(parts)


def _record_overlapping_responsibilities(
    stories: list[Any],
    findings: Any,
    local_id_by_canonical: dict[str, str],
) -> None:
    emitted: set[tuple[str, str]] = set()
    for left in stories:
        for right in stories:
            if left.canonical_id >= right.canonical_id:
                continue
            leaked_from_left = leaked_owned_phrases(left, right)
            leaked_from_right = leaked_owned_phrases(right, left)
            if not leaked_from_left and not leaked_from_right:
                continue
            pair = (left.canonical_id, right.canonical_id)
            if pair in emitted:
                continue
            emitted.add(pair)
            findings.add(
                Finding(
                    severity=FindingSeverity.WARNING,
                    code=FindingCode.OVERLAPPING_STORY_RESPONSIBILITY,
                    message=_overlap_responsibility_message(
                        left,
                        right,
                        leaked_from_left,
                        leaked_from_right,
                        local_id_by_canonical,
                    ),
                    canonical_ids=[left.canonical_id, right.canonical_id],
                    local_ids=_local_ids_for(
                        [left, right],
                        (left.canonical_id, right.canonical_id),
                        local_id_by_canonical,
                    ),
                    source_references=[
                        *list(getattr(left, "provenance", ()) or ()),
                        *list(getattr(right, "provenance", ()) or ()),
                    ],
                )
            )


def _record_story_granularity(item: Any, by_id: dict[str, Any], findings: Any) -> None:
    tasks = [
        by_id[child_id]
        for child_id in item.child_ids
        if child_id in by_id and by_id[child_id].type is WorkItemType.TASK
    ]
    if looks_oversized_story(item):
        findings.add(
            Finding(
                severity=FindingSeverity.WARNING,
                code=FindingCode.OVERSIZED_STORY,
                message=(
                    "User story appears to bundle multiple independently testable outcomes "
                    "or more than a sprint of work. Consider splitting. This is a planning "
                    "heuristic, not an effort estimate."
                ),
                canonical_ids=[item.canonical_id],
                source_references=list(item.provenance),
            )
        )

    if looks_over_decomposed([task.title for task in tasks]):
        findings.add(
            Finding(
                severity=FindingSeverity.INFO,
                code=FindingCode.OVER_DECOMPOSITION,
                message=(
                    "Story has many similarly sliced tasks; combine artificial splits rather "
                    "than tracking duplicate work."
                ),
                canonical_ids=[item.canonical_id],
                source_references=list(item.provenance),
            )
        )

    has_verification = bool(item.acceptance_criteria) or bool(item.test_requirements)
    if not has_verification:
        findings.add(
            Finding(
                severity=FindingSeverity.WARNING,
                code=FindingCode.MISSING_VERIFICATION_COVERAGE,
                message="User story has neither acceptance criteria nor test requirements.",
                canonical_ids=[item.canonical_id],
                source_references=list(item.provenance),
            )
        )

    if tasks:
        return

    if looks_like_implementation_work(item) or looks_oversized_story(item):
        findings.add(
            Finding(
                severity=FindingSeverity.WARNING,
                code=FindingCode.MISSING_IMPLEMENTATION_DETAIL,
                message=(
                    "Story looks like it needs concrete implementation or verification tasks, "
                    "but has none. Record uncertainty instead of inventing filler if the source "
                    "does not say how to build it."
                ),
                canonical_ids=[item.canonical_id],
                source_references=list(item.provenance),
            )
        )
        return

    findings.add(
        Finding(
            severity=FindingSeverity.INFO,
            code=FindingCode.ATOMIC_STORY,
            message=(
                "User story has no child tasks; treated as an atomic implementable slice "
                "rather than incomplete decomposition."
            ),
            canonical_ids=[item.canonical_id],
            source_references=list(item.provenance),
        )
    )
