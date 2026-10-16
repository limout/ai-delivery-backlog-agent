"""Planning heuristics for backlog granularity. Not effort estimates."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from backlog_agent.backlog.ids import normalize_source_text, statement_key
from backlog_agent.backlog.models import WorkItemType
from backlog_agent.backlog.quality_gate import material_hierarchy_message
from backlog_agent.findings.models import Finding, FindingCode, FindingSeverity
from backlog_agent.source.model import NormalizedProjectSource
from backlog_agent.source.request_headings import (
    distinctive_tokens,
    heading_appears_represented,
)

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


_DEPENDENCY_CONTEXT = re.compile(
    r"(?:after|once|when|using|via|through|following|requires|needed for|later|"
    r"depends on|in order to|so that|sourced from|retrieved from|comes from|"
    r"from the|from an|from a|submission to|submitted to|submitting to|submit to|"
    r"to the|via the|using the|through the|against the|integrate with|"
    r"integrates with|integration with)\s+(?:\w+\s+){0,6}"
)


def _phrase_is_dependency_context(text: str, phrase: str) -> bool:
    """True when the phrase appears only as a sequenced dependency, not as owned work."""

    haystack = normalize_source_text(text)
    needle = normalize_source_text(phrase)
    if not haystack or not needle or needle not in haystack:
        return False
    pattern = _DEPENDENCY_CONTEXT.pattern + re.escape(needle)
    return re.search(pattern, haystack) is not None and haystack.count(needle) == len(
        re.findall(pattern, haystack)
    )


def leaked_owned_phrases(owner: Any, other: Any) -> list[str]:
    """Title phrases owned by ``owner`` that appear in ``other``'s description or AC.

    A shared object noun that already appears in both titles is not leaked work.
    Independently testable outcomes may name the same entity with different actions.
    """

    phrases = owned_title_phrases(owner)
    if not phrases:
        return []
    body = _story_body_tokens(other)
    if not body:
        return []
    other_title_tokens = _significant_tokens(getattr(other, "title", "") or "")
    other_text = " ".join(
        [
            getattr(other, "description", "") or "",
            *list(getattr(other, "acceptance_criteria", ()) or ()),
        ]
    )
    leaked: list[str] = []
    for phrase in sorted(phrases):
        if not _contains_phrase_tokens(body, phrase):
            continue
        if _contains_phrase_tokens(other_title_tokens, phrase):
            continue
        if _phrase_is_dependency_context(other_text, phrase):
            continue
        leaked.append(phrase)
    return leaked


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


def _functional_statement_count(item: Any) -> int:
    """Distinct functional statements on a story.

    Paths whose excerpts are the same statement key count once. A path with no
    excerpt stays distinct so a missing quotation is not hidden.
    """

    keyed: set[str] = set()
    unkeyed: set[str] = set()
    for reference in getattr(item, "provenance", ()) or ():
        path = getattr(reference, "json_path", "") or ""
        if not _is_functional_requirement_path(path):
            continue
        excerpt = (getattr(reference, "excerpt", None) or "").strip()
        if excerpt:
            keyed.add(statement_key(excerpt))
        elif path:
            unkeyed.add(path)
    return len(keyed) + len(unkeyed)


def looks_oversized_story(item: Any) -> bool:
    """True when a story appears to bundle multiple independently testable outcomes."""

    if _functional_statement_count(item) >= 2:
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
    if stripped == parent_title:
        return True
    task_tokens = {token for token in _significant_tokens(stripped) if len(token) >= 4}
    parent_tokens = {token for token in _significant_tokens(parent_title) if len(token) >= 4}
    if len(task_tokens) < 2 or len(parent_tokens) < 2:
        return False
    extra = task_tokens - parent_tokens
    return not extra and len(task_tokens & parent_tokens) >= 2


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
    _record_feature_child_alignment(_source, list(backlog.items), findings)
    _record_partial_requirement_coverage(_source, list(backlog.items), findings)
    _record_investigation_task_placement(list(backlog.items), findings)
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


_ALIGNMENT_GENERIC = _GENERIC_CONTEXT_WORDS | {
    "access",
    "application",
    "capabilities",
    "capability",
    "common",
    "core",
    "customer",
    "customers",
    "data",
    "enablement",
    "experience",
    "functionality",
    "functions",
    "general",
    "information",
    "item",
    "items",
    "journey",
    "management",
    "miscellaneous",
    "module",
    "operations",
    "overall",
    "platform",
    "policies",
    "policy",
    "process",
    "processes",
    "product",
    "record",
    "records",
    "request",
    "requests",
    "service",
    "services",
    "shared",
    "solution",
    "system",
    "systems",
    "workflow",
}


def _stem_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
        return token[:-1]
    return token


def _alignment_tokens(text: str) -> set[str]:
    stemmed: set[str] = set()
    for raw in normalize_source_text(text).split():
        token = _stem_token(raw.strip(".,;:!?\"'`()[]{}"))
        if len(token) >= 3 and token not in _ALIGNMENT_GENERIC:
            stemmed.add(token)
    return stemmed


def _alignment_tokens_related(left: str, right: str) -> bool:
    if left == right:
        return True
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    if not long.startswith(short):
        return False
    extra = long[len(short) :]
    if extra == "al" and len(short) >= 5:
        return True
    if extra in {"ing", "ed", "ion", "tion", "ation", "es"} and len(short) >= 6:
        return True
    if extra in {"ment", "ements"} and len(short) >= 3:
        return True
    if extra in {"ication", "ications"} and len(short) >= 5:
        return True
    if extra == "s" and len(short) >= 4:
        return True
    return False


def _alignment_sets_share_term(left: set[str], right: set[str]) -> bool:
    if left & right:
        return True
    return any(_alignment_tokens_related(token, other) for token in left for other in right)


def _child_text(child: Any) -> str:
    return f"{getattr(child, 'title', '')} {getattr(child, 'description', '')}"


def _child_capability_tokens(child: Any) -> set[str]:
    parts = [_child_text(child)]
    for reference in list(getattr(child, "provenance", ()) or ()):
        excerpt = getattr(reference, "excerpt", None) or ""
        path = getattr(reference, "json_path", "") or ""
        if excerpt and _is_functional_requirement_path(path):
            parts.append(excerpt)
    return _alignment_tokens(" ".join(parts))


_TITLE_CLAUSE_SPLIT = re.compile(r"\s*(?:,|/|&|\band\b)\s+", re.I)
_OUTCOME_VERBS = (
    r"access|view|open|see|display|record|submit|capture|review|"
    r"start|complete|create|update|notify|assign|track|sync(?:hron(?:ize|ise))?|"
    r"upload|download|search|approve|export|import|remain|keep|restore"
)
_COORDINATED_OUTCOME = re.compile(
    rf"\b({_OUTCOME_VERBS})\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+and\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:[.?;]|$)",
    re.IGNORECASE,
)
_AND_VERB_CLAUSE = re.compile(
    rf"(.+?)\s+and\s+((?:{_OUTCOME_VERBS})\b.+)",
    re.IGNORECASE,
)
_INVESTIGATION_TASK = re.compile(
    r"\b(investigate|confirm|verify|check|clarify|discover|determine|unspecified|unknown)\b",
    re.IGNORECASE,
)


def requirement_outcomes(text: str) -> tuple[str, ...]:
    """Independently observable outcomes named in one requirement sentence.

    Coordinated objects after one verb become separate outcomes. This does not
    require a story per sentence, only that generated acceptance criteria verify
    each outcome the story claims.
    """

    source = str(text or "").strip()
    if not source:
        return ()
    found: list[str] = []
    seen: set[str] = set()

    def _add(phrase: str) -> None:
        cleaned = re.sub(r"\s+", " ", phrase).strip(" .,;:")
        if len(cleaned) < 8:
            return
        key = normalize_source_text(cleaned)
        if key in seen:
            return
        seen.add(key)
        found.append(cleaned)

    for match in _COORDINATED_OUTCOME.finditer(source):
        verb = match.group(1).lower()
        for raw in (match.group(2), match.group(3)):
            obj = re.sub(r"\s+", " ", raw).strip(" .,;:")
            if len(obj) < 4:
                continue
            _add(f"{verb} {obj}")
    clause = _AND_VERB_CLAUSE.search(source)
    if clause and not found:
        _add(clause.group(1))
        _add(clause.group(2))
    return tuple(found)


def _item_observable_text(item: Any) -> str:
    parts = [
        getattr(item, "title", "") or "",
        getattr(item, "description", "") or "",
        *list(getattr(item, "acceptance_criteria", ()) or ()),
        *list(getattr(item, "test_requirements", ()) or ()),
    ]
    return " ".join(str(part) for part in parts if str(part).strip())


def _item_verification_text(item: Any) -> str:
    """Only what a reviewer could check: acceptance criteria and test requirements."""

    parts = [
        *list(getattr(item, "acceptance_criteria", ()) or ()),
        *list(getattr(item, "test_requirements", ()) or ()),
    ]
    return " ".join(str(part) for part in parts if str(part).strip())


_REQUIREMENT_CONTEXT_TOKENS = frozenset(
    {
        "allow",
        "allows",
        "application",
        "applications",
        "during",
        "field",
        "must",
        "need",
        "needs",
        "remain",
        "support",
        "usable",
        "work",
    }
)


def item_represents_requirement(
    item: Any,
    text: str,
    *,
    background: set[str] | None = None,
    verification_only: bool = False,
) -> bool:
    """True when item wording reflects the requirement, not merely provenance.

    ``background`` tokens (typically other labeled requirements) are subtracted
    so shared domain nouns cannot stand in for a distinct constraint. Generic
    product-context words such as application or support also cannot stand in
    for a material behavior.

    With ``verification_only`` only acceptance criteria and test requirements
    count. A Story that merely mentions a constraint in its description or
    title does not verify it.
    """

    blob = _item_verification_text(item) if verification_only else _item_observable_text(item)
    if not blob.strip() or not str(text or "").strip():
        return False
    outcomes = requirement_outcomes(text)
    if outcomes:
        return all(
            _verification_covers_outcome(outcome, outcomes, blob) for outcome in outcomes
        )
    required = distinctive_tokens(text)
    skip = set(_REQUIREMENT_CONTEXT_TOKENS)
    if background:
        skip |= background
    specific = {
        token
        for token in required
        if token not in skip
        and not any(token == other or token.rstrip("s") == other.rstrip("s") for other in skip)
    }
    if specific:
        required = specific
    if not required:
        return heading_appears_represented(text, blob)
    haystack = distinctive_tokens(blob)
    matched = {
        token
        for token in required
        if token in haystack or any(token.rstrip("s") == other.rstrip("s") for other in haystack)
    }
    needed = 2 if len(required) <= 3 else 3
    return len(matched) >= needed


def _outcome_focus_tokens(outcome: str) -> set[str]:
    """Tokens that distinguish one coordinated outcome from a trailing context clause."""

    core = re.split(
        r"\s+\b(?:from|via|using|against|within|under|onto)\b\s+",
        str(outcome or ""),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return distinctive_tokens(core)


def _verification_covers_outcome(
    outcome: str,
    siblings: tuple[str, ...] | list[str],
    verification: str,
) -> bool:
    """True when verification names this outcome, not only tokens shared with siblings."""

    focus = _outcome_focus_tokens(outcome)
    shared: set[str] = set()
    for sibling in siblings:
        if sibling == outcome:
            continue
        shared |= _outcome_focus_tokens(sibling)
    specific = {
        token
        for token in focus
        if token not in shared
        and not any(token.rstrip("s") == other.rstrip("s") for other in shared)
    }
    if not specific:
        return heading_appears_represented(outcome, verification)
    haystack = distinctive_tokens(verification)
    matched = {
        token
        for token in specific
        if token in haystack or any(token.rstrip("s") == other.rstrip("s") for other in haystack)
    }
    return bool(matched)


def _feature_title_clauses(title: str) -> list[str]:
    parts = [part.strip() for part in _TITLE_CLAUSE_SPLIT.split(str(title or "")) if part.strip()]
    clauses = [part for part in parts if len(_alignment_tokens(part)) >= 1]
    return clauses if len(clauses) >= 2 else []


def _shared_title_hubs(feature: Any, children: list[Any]) -> set[str]:
    title_tokens = _alignment_tokens(
        f"{getattr(feature, 'title', '')} {getattr(feature, 'description', '')}"
    )
    child_sets = [_child_capability_tokens(child) for child in children]
    nonempty = [tokens for tokens in child_sets if tokens]
    if len(nonempty) < 2:
        return set()
    shared = set.intersection(*nonempty)
    return shared & title_tokens


def _majority_title_hubs(feature: Any, children: list[Any]) -> set[str]:
    """Title tokens that appear in more than half of the feature's children.

    Shared workflow nouns in a compound title are not evidence that every child
    belongs to that title's named capability areas.
    """

    title_tokens = _alignment_tokens(
        f"{getattr(feature, 'title', '')} {getattr(feature, 'description', '')} {_feature_excerpt(feature)}"
    )
    child_sets = [_child_capability_tokens(child) for child in children]
    if len(child_sets) < 2:
        return set()
    threshold = len(child_sets) / 2
    hubs: set[str] = set()
    for token in title_tokens:
        count = sum(
            1
            for tokens in child_sets
            if token in tokens
            or any(_alignment_tokens_related(token, other) for other in tokens)
        )
        if count > threshold:
            hubs.add(token)
    return hubs


def _cluster_feature_children(children: list[Any], ignore: set[str] | None = None) -> list[list[Any]]:
    hubs = ignore or set()
    clusters: list[list[Any]] = []
    for child in children:
        tokens = _child_capability_tokens(child) - hubs
        assigned = None
        for cluster in clusters:
            if any(
                _alignment_sets_share_term(tokens, _child_capability_tokens(member) - hubs)
                for member in cluster
            ):
                assigned = cluster
                break
        if assigned is None:
            clusters.append([child])
        else:
            assigned.append(child)
    return clusters


def _unrelated_child_pair_ratio(children: list[Any], ignore: set[str] | None = None) -> float:
    """Fraction of child pairs that share no distinctive capability terms."""

    hubs = ignore or set()
    token_sets = [_child_capability_tokens(child) - hubs for child in children]
    pairs = 0
    unrelated = 0
    for index, left in enumerate(token_sets):
        for right in token_sets[index + 1 :]:
            pairs += 1
            if not _alignment_sets_share_term(left, right):
                unrelated += 1
    if pairs == 0:
        return 0.0
    return unrelated / pairs


def _phrase_in_scope(scope: str, phrase: str) -> bool:
    """True when ``phrase`` appears in ``scope``, allowing a simple plural ``s``."""

    haystack = f" {normalize_source_text(scope)} "
    needle = normalize_source_text(phrase)
    if len(needle) < 8:
        return False
    if f" {needle} " in haystack:
        return True
    if needle.endswith("s") and len(needle) > 8 and f" {needle[:-1]} " in haystack:
        return True
    if f" {needle}s " in haystack:
        return True
    return False


def _child_covered_by_feature_scope(scope: str, child: Any, *, title_only: bool = False) -> bool:
    """True when the feature's stated scope covers this child semantically.

    Coverage is phrase and token based. A child does not have to repeat the
    feature title's distinctive words when the feature title, description, or
    source capability already names that child's outcome.
    """

    if not normalize_source_text(scope):
        return False
    child_tokens = (
        _alignment_tokens(getattr(child, "title", "") or "")
        if title_only
        else _child_capability_tokens(child)
    )
    if _alignment_sets_share_term(_alignment_tokens(scope), child_tokens):
        return True
    title = getattr(child, "title", "") or ""
    tokens = normalize_source_text(title).split()
    for size in (3, 2):
        if len(tokens) < size:
            continue
        for index in range(len(tokens) - size + 1):
            gram = " ".join(tokens[index : index + size])
            if _phrase_in_scope(scope, gram):
                return True
    return False


def _feature_excerpt(feature: Any) -> str:
    for reference in list(getattr(feature, "provenance", ()) or ()):
        excerpt = getattr(reference, "excerpt", None) or ""
        if excerpt:
            return excerpt
    return ""


def _children_split_by_title_clauses(feature: Any, children: list[Any]) -> list[list[Any]] | None:
    """Split when a compound title names distinct child capability areas."""

    clauses = _feature_title_clauses(getattr(feature, "title", "") or "")
    if len(clauses) < 2 or len(children) < 2:
        return None
    buckets: list[list[Any]] = [[] for _ in clauses]
    leftover: list[Any] = []
    for child in children:
        hits = [
            index
            for index, clause in enumerate(clauses)
            if _alignment_sets_share_term(
                _alignment_tokens(clause),
                _child_capability_tokens(child),
            )
        ]
        if len(hits) == 1:
            buckets[hits[0]].append(child)
        else:
            leftover.append(child)
    occupied = [bucket for bucket in buckets if bucket]
    if len(occupied) < 2:
        return None
    if leftover:
        occupied.append(leftover)
    return occupied


def _children_form_distinct_capability_clusters(
    feature: Any,
    children: list[Any],
) -> list[list[Any]] | None:
    """Return child clusters when they do not share one capability area.

    Pairwise overlap among children is the cohesion signal. A shared workflow
    noun in the feature title does not make independent outcomes one capability.
    Compound titles that name two areas are split even with only two children.
    Listing every child capability in the feature description does not create
    overlap. This is not a required feature count.
    """

    titled = _children_split_by_title_clauses(feature, children)
    if titled is not None:
        return titled
    if len(children) < 3:
        return None
    hubs = _shared_title_hubs(feature, children)
    if _unrelated_child_pair_ratio(children, ignore=hubs) <= 0.5:
        return None
    clusters = _cluster_feature_children(children, ignore=hubs)
    if len(clusters) < 2:
        clusters = [[child] for child in children]
    return clusters


_TITLE_FILLER_LEADS = frozenset(
    {"the", "a", "an", "as", "system", "users", "user", "it", "we", "this", "that", "to"}
)


def _leading_verb(title: str) -> str:
    """The action a title opens with, or "" for a sentence-style title with no action lead."""

    words = normalize_source_text(title or "").split()
    if not words or words[0] in _TITLE_FILLER_LEADS:
        return ""
    return _stem_token(words[0])


def _children_form_broad_clusters(children: list[Any]) -> list[list[Any]] | None:
    """Weak evidence that a large feature spans several capability areas.

    Words shared by at least half of the stories (an actor, the product's own subject) hide
    real differences, so they are ignored here. This is weaker than the pairwise cohesion
    signal and is only ever reported as advisory. Stories that all start with the same verb
    are variants of one action and are not treated as broad.
    """

    if len(children) < 4:
        return None
    token_sets = [_child_capability_tokens(child) for child in children]
    half = len(children) / 2
    ubiquitous = {
        token
        for tokens in token_sets
        for token in tokens
        if sum(
            1
            for other in token_sets
            if token in other or any(_alignment_tokens_related(token, item) for item in other)
        )
        >= half
    }
    verbs = Counter(
        verb for verb in (_leading_verb(getattr(child, "title", "") or "") for child in children) if verb
    )
    if verbs and max(verbs.values()) >= half:
        return None
    clusters = _cluster_feature_children(children, ignore=ubiquitous)
    if len(clusters) < 3:
        return None
    if max(len(cluster) for cluster in clusters) > half:
        return None
    return clusters


def _is_material_feature_defect(
    children: list[Any],
    clusters: list[list[Any]],
    title_covered_count: int,
) -> bool:
    """High-confidence only: a clearly unrelated collection of stories under one feature.

    All of these must hold. A weak or partial signal (two clusters, a compound title, one
    outlier story) stays an advisory heuristic.
    """

    if len(children) < 3 or len(clusters) < 3:
        return False
    if max(len(cluster) for cluster in clusters) / len(children) > 0.5:
        return False
    return title_covered_count / len(children) < 0.5


def _record_feature_child_alignment(_source: Any, items: list[Any], findings: Any) -> None:
    """Warn when a feature's children are not one capability area.

    Child-to-child clustering is the cohesion signal. Title and description token
    overlap cannot override distinct child clusters; enumerating those clusters in
    the feature description is not evidence they belong together. Provenance is
    checked separately so a narrow source excerpt is not treated as the whole group.
    Token overlap is a review heuristic, not a hard failure or a feature-count rule.
    """

    by_id = {item.canonical_id: item for item in items}
    for feature in items:
        if getattr(feature, "type", None) is not WorkItemType.FEATURE:
            continue
        children = [
            by_id[child_id]
            for child_id in getattr(feature, "child_ids", ()) or ()
            if child_id in by_id and by_id[child_id].type is WorkItemType.USER_STORY
        ]
        if not children:
            continue
        excerpt = _feature_excerpt(feature)
        title_scope = f"{getattr(feature, 'title', '')} {excerpt}"
        full_scope = (
            f"{getattr(feature, 'title', '')} {getattr(feature, 'description', '')} {excerpt}"
        )
        distinct_clusters = _children_form_distinct_capability_clusters(feature, children)
        majority_hubs = _majority_title_hubs(feature, children)
        distinctive_scope = " ".join(
            token
            for token in _alignment_tokens(title_scope)
            if token not in majority_hubs
        ) or title_scope
        title_covered = [
            child
            for child in children
            if _child_covered_by_feature_scope(
                distinctive_scope,
                child,
                title_only=True,
            )
        ]
        if distinct_clusters is not None and len(title_covered) / len(children) < 0.75:
            cluster_titles = "; ".join(
                ", ".join(repr(child.title) for child in cluster[:3])
                for cluster in distinct_clusters[:4]
            )
            material = _is_material_feature_defect(
                children, distinct_clusters, len(title_covered)
            )
            base = (
                f"Feature {feature.title!r} groups {len(children)} stories that form "
                f"{len(distinct_clusters)} distinct capability clusters ({cluster_titles}). "
                "Listing those areas in the feature description does not make them one "
                "capability. Move related stories under features whose titles describe "
                "those areas. "
            )
            tail = (
                "Token matching is a heuristic, but the evidence here is high confidence: no "
                "cluster holds more than half of the stories and the feature's own title and "
                "source excerpt cover fewer than half of them, so the grouping is not accepted "
                "until it is fixed. This does not require a particular feature count."
                if material
                else "This is a heuristic, not a hard failure and not a requirement to add "
                "extra features."
            )
            message = base + tail
            findings.add(
                Finding(
                    severity=FindingSeverity.WARNING,
                    code=FindingCode.FEATURE_CHILD_MISALIGNMENT,
                    message=material_hierarchy_message(message) if material else message,
                    canonical_ids=[feature.canonical_id, *[child.canonical_id for child in children]],
                    source_references=list(getattr(feature, "provenance", ()) or ()),
                )
            )
            continue
        if distinct_clusters is None and len(title_covered) / len(children) < 0.5:
            broad = _children_form_broad_clusters(children)
            if broad is not None:
                cluster_titles = "; ".join(
                    ", ".join(repr(child.title) for child in cluster[:3]) for cluster in broad[:5]
                )
                findings.add(
                    Finding(
                        severity=FindingSeverity.WARNING,
                        code=FindingCode.FEATURE_CHILD_MISALIGNMENT,
                        message=(
                            f"Feature {feature.title!r} holds {len(children)} stories that fall "
                            f"into {len(broad)} separate capability clusters once words shared by "
                            f"most stories are ignored ({cluster_titles}), and the feature's own "
                            "title names few of them. Consider features whose titles describe "
                            "those capability areas. This is weak token evidence, so it is "
                            "advisory and does not require any particular feature count."
                        ),
                        canonical_ids=[
                            feature.canonical_id,
                            *[child.canonical_id for child in children],
                        ],
                        source_references=list(getattr(feature, "provenance", ()) or ()),
                    )
                )
                continue
        claimed = _alignment_tokens(
            f"{getattr(feature, 'title', '')} {getattr(feature, 'description', '')}"
        )
        provenance_tokens = _alignment_tokens(excerpt)
        if len(claimed) < 2 and len(provenance_tokens) >= 2:
            claimed = provenance_tokens
        if len(claimed) < 2:
            continue
        matched = [
            child for child in children if _child_covered_by_feature_scope(full_scope, child)
        ]
        provenance_matched = [
            child for child in children if _child_covered_by_feature_scope(excerpt, child)
        ]
        coherent = bool(matched) and len(matched) / len(children) >= 0.5
        provenance_grounded = not provenance_tokens or bool(provenance_matched)
        if coherent and provenance_grounded:
            continue
        if not claimed and not matched:
            continue
        outside = [child for child in children if child not in matched]
        outside_titles = ", ".join(repr(child.title) for child in outside[:4])
        if not provenance_grounded:
            detail = (
                "provenance does not share distinctive terms with any child user story. "
                "Cite a source capability the children implement, or move the children."
            )
        else:
            detail = (
                f"title and description share distinctive terms with {len(matched)} of "
                f"{len(children)} child stories. Stories outside that scope: {outside_titles}. "
                "Move those stories to a feature whose source capability covers them. A broad "
                "feature may keep a child that does not repeat the same words."
            )
        findings.add(
            Finding(
                severity=FindingSeverity.WARNING,
                code=FindingCode.FEATURE_CHILD_MISALIGNMENT,
                message=(
                    f"Feature {feature.title!r} {detail} This is a heuristic, not a hard "
                    "failure and not a requirement to add extra features."
                ),
                canonical_ids=[feature.canonical_id, *[child.canonical_id for child in children]],
                source_references=list(getattr(feature, "provenance", ()) or ()),
            )
        )


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
        "Remediation: keep this criterion on the one story whose outcome it verifies. "
        "If the source criterion concatenates independently owned outcomes, replace the "
        "shared copy with generated criteria scoped to each story and never mark that "
        "generated text as a verbatim source quotation."
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


def _prefer_source_wording(outcome: str, source_outcomes: tuple[str, ...]) -> str:
    """Quote the source requirement's own wording when a title fragment names the same outcome."""

    if outcome in source_outcomes:
        return outcome
    focus = _alignment_tokens(outcome)
    if not focus:
        return outcome
    for source_outcome in source_outcomes:
        candidate = _alignment_tokens(source_outcome)
        if all(
            token in candidate or any(_alignment_tokens_related(token, item) for item in candidate)
            for token in focus
        ):
            return source_outcome
    return outcome


def _record_partial_requirement_coverage(
    source: NormalizedProjectSource,
    items: list[Any],
    findings: Any,
) -> None:
    """Warn when a story cites a multi-outcome requirement but verifies only part of it."""

    stories = [
        item
        for item in items
        if getattr(item, "type", None) is WorkItemType.USER_STORY
    ]
    for story in stories:
        cited: list[str] = []
        for reference in list(getattr(story, "provenance", ()) or ()):
            path = getattr(reference, "json_path", "") or ""
            excerpt = getattr(reference, "excerpt", None) or ""
            if not excerpt:
                continue
            if _is_functional_requirement_path(path) or _is_acceptance_criterion_path(path):
                cited.append(excerpt)
        claimed_scope = f"{getattr(story, 'title', '')} {getattr(story, 'description', '')}"
        verification = " ".join(
            [
                *list(getattr(story, "acceptance_criteria", ()) or ()),
                *list(getattr(story, "test_requirements", ()) or ()),
            ]
        )
        claimed_outcomes = requirement_outcomes(getattr(story, "title", "") or "")
        source_outcomes = tuple(
            outcome for excerpt in cited for outcome in requirement_outcomes(excerpt)
        )
        siblings = tuple(dict.fromkeys((*claimed_outcomes, *source_outcomes)))
        missing: list[str] = []
        for outcome in siblings:
            if outcome in source_outcomes and not heading_appears_represented(
                outcome, claimed_scope
            ):
                continue
            if not _verification_covers_outcome(outcome, siblings, verification):
                missing.append(outcome)
        if not missing:
            continue
        unique = list(
            dict.fromkeys(_prefer_source_wording(item, source_outcomes) for item in missing)
        )
        findings.add(
            Finding(
                severity=FindingSeverity.WARNING,
                code=FindingCode.MISSING_VERIFICATION_COVERAGE,
                message=(
                    f"User story {story.title!r} does not verify every outcome in its cited "
                    f"source requirement ({'; '.join(repr(item) for item in unique[:4])}). "
                    "Add acceptance criteria and test requirements that name each missing "
                    "outcome using the source wording, or split the requirement into coherent "
                    "stories. Verifying a related outcome (for example, related data) does not "
                    "verify this one, and provenance alone is not coverage."
                ),
                canonical_ids=[story.canonical_id],
                source_references=list(getattr(story, "provenance", ()) or ()),
            )
        )


_UNRESOLVED_PATH_MARKERS = ("unknown", "open_question", "dependencies", "discovery")


_QUESTION_FRAME_WORDS = frozenset(
    {
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "how",
        "when",
        "where",
        "why",
        "whether",
        "are",
        "does",
        "should",
        "required",
        "require",
        "requires",
        "specific",
        "exact",
        "exactly",
        "detailed",
        "unspecified",
        "unknown",
        "unclear",
        "undefined",
        "confirmed",
        "remaining",
        "clarify",
        "needed",
        "pending",
    }
)


def _investigation_subject_tokens(task: Any) -> set[str]:
    """Tokens of the unresolved subject itself, without shared boilerplate wording.

    Generated clarification tasks repeat the same scaffolding text ("resolve this
    remaining detail ...") and questions share frame words ("which", "required").
    Scoring on either makes unrelated tasks look alike, so prefer the cited
    unknown's own excerpt and drop the frame words.
    """

    return {
        token
        for token in _investigation_raw_tokens(task)
        if token not in _QUESTION_FRAME_WORDS
    }


def _investigation_raw_tokens(task: Any) -> set[str]:
    tokens: set[str] = set()
    for reference in list(getattr(task, "provenance", ()) or ()):
        path = getattr(reference, "json_path", "") or ""
        excerpt = getattr(reference, "excerpt", None) or ""
        if excerpt and any(marker in path for marker in _UNRESOLVED_PATH_MARKERS):
            tokens |= _alignment_tokens(excerpt)
    if tokens:
        return tokens
    tokens = _alignment_tokens(_item_observable_text(task))
    for reference in list(getattr(task, "provenance", ()) or ()):
        excerpt = getattr(reference, "excerpt", None) or ""
        if excerpt:
            tokens |= _alignment_tokens(excerpt)
    return tokens


def _investigation_task_score(task: Any, story: Any) -> int:
    story_tokens = _alignment_tokens(_item_observable_text(story))
    return len(_investigation_subject_tokens(task) & story_tokens)


def _investigation_subjects_overlap(left: set[str], right: set[str]) -> bool:
    """True when two unresolved subjects are essentially the same question."""

    if not left or not right:
        return False
    shared = {
        token
        for token in left
        if token in right or any(_alignment_tokens_related(token, other) for other in right)
    }
    smaller = min(len(left), len(right))
    if smaller <= 3:
        return len(shared) >= 2 and len(shared) == smaller
    return len(shared) >= 3 and len(shared) / smaller >= 0.6


def is_investigation_task(item: Any) -> bool:
    blob = _item_observable_text(item)
    if _INVESTIGATION_TASK.search(blob):
        return True
    return any(
        any(
            marker in (getattr(reference, "json_path", "") or "")
            for marker in ("unknown", "open_question", "dependencies", "discovery")
        )
        for reference in list(getattr(item, "provenance", ()) or ())
    )


def _task_unresolved_paths(task: Any) -> set[str]:
    return {
        getattr(reference, "json_path", "") or ""
        for reference in list(getattr(task, "provenance", ()) or ())
    }


def _record_overlapping_investigation_tasks(tasks: list[Any], findings: Any) -> None:
    """Warn when different source paths raise what is essentially one question.

    The same unknown is often listed twice (for example once as a discovery unknown and
    once as an open question). Same-path duplicates are reported separately.
    """

    reported: set[frozenset[str]] = set()
    subjects = {task.canonical_id: _investigation_subject_tokens(task) for task in tasks}
    for index, left in enumerate(tasks):
        for right in tasks[index + 1 :]:
            if _task_unresolved_paths(left) & _task_unresolved_paths(right):
                continue
            if not _investigation_subjects_overlap(
                subjects[left.canonical_id], subjects[right.canonical_id]
            ):
                continue
            key = frozenset({left.canonical_id, right.canonical_id})
            if key in reported:
                continue
            reported.add(key)
            findings.add(
                Finding(
                    severity=FindingSeverity.WARNING,
                    code=FindingCode.OVER_DECOMPOSITION,
                    message=material_hierarchy_message(
                        f"Overlapping investigation tasks {left.title!r} and {right.title!r} ask "
                        "essentially the same unresolved question from different source "
                        "entries. Keep one well-scoped clarification task under the story whose "
                        "outcome it constrains and cite both source entries on it."
                    ),
                    canonical_ids=[left.canonical_id, right.canonical_id],
                    source_references=[
                        reference
                        for task in (left, right)
                        for reference in list(getattr(task, "provenance", ()) or ())
                    ][:4],
                )
            )


def _is_clearly_misparented(task: Any, parent: Any | None, best: Any, ranked: list[Any]) -> bool:
    """High-confidence only: the task's own subject has nothing in common with its parent
    while exactly one other story clearly shares it.

    Sharing any term with the parent (including a stem variant) makes the assignment
    uncertain, not clearly wrong. A tie between candidate stories is also uncertain.
    """

    if parent is None:
        return False
    subject = _investigation_subject_tokens(task)
    if len(subject) < 2:
        return False
    parent_tokens = _alignment_tokens(_item_observable_text(parent))
    if _alignment_sets_share_term(subject, parent_tokens):
        return False
    best_score = _investigation_task_score(task, best)
    runner_up = max(
        (_investigation_task_score(task, story) for story in ranked if story is not best),
        default=0,
    )
    return best_score >= 2 and best_score > runner_up


def _record_investigation_task_placement(items: list[Any], findings: Any) -> None:
    """Warn when discovery tasks sit on a weaker parent or duplicate the same unknown."""

    by_id = {item.canonical_id: item for item in items}
    stories = _user_stories(items)
    tasks = [
        item
        for item in items
        if getattr(item, "type", None) is WorkItemType.TASK and is_investigation_task(item)
    ]
    by_path: dict[str, list[Any]] = defaultdict(list)
    for task in tasks:
        parent = by_id.get(getattr(task, "parent_id", None) or "")
        if stories:
            ranked = sorted(
                stories,
                key=lambda story: _investigation_task_score(task, story),
                reverse=True,
            )
            best = ranked[0]
            best_score = _investigation_task_score(task, best)
            parent_score = _investigation_task_score(task, parent) if parent is not None else -1
            if (
                best is not parent
                and best_score >= 2
                and best_score > parent_score
            ):
                clearly_unrelated = _is_clearly_misparented(task, parent, best, ranked)
                if clearly_unrelated:
                    message = material_hierarchy_message(
                        f"Investigation task {task.title!r} is attached to "
                        f"{getattr(parent, 'title', 'an unrelated parent')!r}, which shares no "
                        f"part of its unresolved subject, while {best.title!r} clearly does. "
                        "Move it under the story whose outcome the unknown actually constrains."
                    )
                else:
                    message = (
                        f"Investigation task {task.title!r} is attached to "
                        f"{getattr(parent, 'title', 'an unrelated parent')!r} but shares "
                        f"more of its unresolved subject with {best.title!r}. Move it under "
                        "the story whose outcome the unknown actually constrains."
                    )
                findings.add(
                    Finding(
                        severity=FindingSeverity.WARNING,
                        code=FindingCode.AMBIGUOUS_DECOMPOSITION,
                        message=message,
                        canonical_ids=[
                            task.canonical_id,
                            *([parent.canonical_id] if parent is not None else []),
                            best.canonical_id,
                        ],
                        source_references=list(getattr(task, "provenance", ()) or ()),
                    )
                )
        for reference in list(getattr(task, "provenance", ()) or ()):
            path = getattr(reference, "json_path", "") or ""
            if any(marker in path for marker in ("unknown", "open_question", "dependencies")):
                by_path[path].append(task)
    _record_overlapping_investigation_tasks(tasks, findings)
    for path, group in by_path.items():
        if len(group) < 2:
            continue
        findings.add(
            Finding(
                severity=FindingSeverity.WARNING,
                code=FindingCode.OVER_DECOMPOSITION,
                message=material_hierarchy_message(
                    f"{len(group)} investigation tasks cite the same source unknown ({path}). "
                    "Keep one well-scoped discovery task for that uncertainty instead of "
                    "overlapping clarification work."
                ),
                canonical_ids=[item.canonical_id for item in group],
                source_references=[
                    reference
                    for item in group
                    for reference in list(getattr(item, "provenance", ()) or ())
                    if getattr(reference, "json_path", "") == path
                ][:4],
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
                    "but has none. If a source unknown or dependency names a checkable fact, "
                    "add an investigation task that cites it and records an unconfirmed result. "
                    "Otherwise record uncertainty instead of inventing filler."
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
