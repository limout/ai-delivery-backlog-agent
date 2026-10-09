"""Deterministic Copilot envelope → canonical backlog generator.

Mapping (conservative, no invented requirements or estimates):

- Epic: ``$.user_request``
- Feature: in-scope ``$.solution.key_capabilities[*]`` then
  ``$.delivery_plan.workstreams[*]``
- User story: in-scope ``$.requirements.functional_requirements[*]``
- Task: in-scope ``$.sow.deliverables[*]`` then ``$.delivery_plan.delivery_phases[*]``
- Subtask: in-scope ``$.delivery_plan.milestones[*]``

A parent link is stored only when exactly one eligible parent exists.
Otherwise the item is kept as Draft with ``parent_id=None`` and an
``AMBIGUOUS_DECOMPOSITION`` finding. No guessed or synthetic parent is created.

Item IDs are UUID5 of a content-addressed key
``{base_path}|{normalized_text}|{occurrence}``. Provenance ``json_path`` stays
index-based. ``backlog_id`` is an envelope fingerprint, not item identity.

Status is workflow ``todo``. Approval is ``not_approved``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from backlog_agent.backlog.ids import (
    canonical_id_from_source_key,
    item_identity_key,
    normalize_source_text,
)
from backlog_agent.backlog.models import (
    ApprovalState,
    CanonicalBacklogV1,
    Epic,
    Feature,
    SourceReference,
    Subtask,
    Task,
    UserStory,
    WorkItemStatus,
    WorkItemType,
)
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.contracts.versions import COPILOT_WORKFLOW_RESPONSE_V1
from backlog_agent.findings.models import (
    Finding,
    FindingCode,
    FindingSeverity,
    GenerationFindings,
)

_MIN_SOW_LINE_LEN = 12
_JSON_PATH_TOKEN = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]")

_CONSTRUCTORS = {
    WorkItemType.EPIC: Epic,
    WorkItemType.FEATURE: Feature,
    WorkItemType.USER_STORY: UserStory,
    WorkItemType.TASK: Task,
    WorkItemType.SUBTASK: Subtask,
}


@dataclass
class _Draft:
    item_type: WorkItemType
    canonical_id: str
    title: str
    json_path: str
    field_name: str
    excerpt: str
    identity_key: str
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)


class _FindingSink:
    def __init__(self) -> None:
        self.items: list[Finding] = []
        self._seen: set[tuple[Any, ...]] = set()
        self.scope_conflict_texts: set[str] = set()

    def add(self, finding: Finding) -> None:
        key = (finding.code, tuple(sorted(ref.json_path for ref in finding.source_references)))
        if key in self._seen:
            return
        self._seen.add(key)
        self.items.append(finding)


class _IdAllocator:
    def __init__(self, findings: _FindingSink) -> None:
        self._used: dict[str, str] = {}
        self._findings = findings

    def allocate(self, identity_key: str, json_path: str) -> str:
        candidate_key = identity_key
        for attempt in range(1, 33):
            canonical_id = canonical_id_from_source_key(candidate_key)
            owner = self._used.get(canonical_id)
            if owner is None:
                self._used[canonical_id] = identity_key
                if attempt > 1:
                    self._findings.add(
                        Finding(
                            severity=FindingSeverity.ERROR,
                            code=FindingCode.ID_COLLISION,
                            message=(
                                "Canonical ID collision resolved by suffixing the identity key "
                                f"{identity_key!r} (attempt {attempt})"
                            ),
                            source_references=[_source_ref(json_path, None)],
                            canonical_ids=[canonical_id],
                        )
                    )
                return canonical_id
            if owner == identity_key:
                return canonical_id
            candidate_key = f"{identity_key}#collision-{attempt + 1}"
        raise RuntimeError(f"unresolved canonical ID collision for {identity_key!r}")


def generate_canonical_backlog(
    envelope: CopilotWorkflowResponseV1 | Mapping[str, Any],
) -> tuple[CanonicalBacklogV1, GenerationFindings]:
    """Generate a canonical backlog and structured findings from a COMPLETE envelope."""

    parsed = (
        envelope
        if isinstance(envelope, CopilotWorkflowResponseV1)
        else CopilotWorkflowResponseV1.model_validate(envelope)
    )
    original = parsed.original_envelope
    findings = _FindingSink()
    ids = _IdAllocator(findings)

    _record_sow_list_conflicts(original, findings)
    _record_requirement_negations(original, findings)

    epic = _try_source_item(
        original,
        ids,
        findings,
        base_path="$.user_request",
        json_path="$.user_request",
        field_name="user_request",
        item_type=WorkItemType.EPIC,
        text=str(original.get("user_request", "")),
        occurrence=0,
    )

    features = _collect_list_items(
        original,
        ids,
        findings,
        [
            ("$.solution.key_capabilities", "key_capabilities", WorkItemType.FEATURE),
            ("$.delivery_plan.workstreams", "workstreams", WorkItemType.FEATURE),
        ],
    )
    stories = _collect_list_items(
        original,
        ids,
        findings,
        [
            (
                "$.requirements.functional_requirements",
                "functional_requirements",
                WorkItemType.USER_STORY,
            )
        ],
    )
    tasks = _collect_list_items(
        original,
        ids,
        findings,
        [
            ("$.sow.deliverables", "deliverables", WorkItemType.TASK),
            ("$.delivery_plan.delivery_phases", "delivery_phases", WorkItemType.TASK),
        ],
    )
    subtasks = _collect_list_items(
        original,
        ids,
        findings,
        [("$.delivery_plan.milestones", "milestones", WorkItemType.SUBTASK)],
    )

    _link_unique_parent(
        [epic] if epic is not None else [],
        features,
        findings,
        "Features attach to the request epic only when that epic exists.",
    )
    _link_unique_parent(
        features,
        stories,
        findings,
        "User stories were not linked to a parent because Copilot does not name exactly one in-scope feature.",
    )
    _link_unique_parent(
        stories,
        tasks,
        findings,
        "Tasks were not linked to a parent because Copilot does not name exactly one in-scope user story.",
    )
    _link_unique_parent(
        tasks,
        subtasks,
        findings,
        "Milestones were not linked to a parent because Copilot does not name exactly one in-scope task.",
    )

    if len(stories) == 1:
        ac_entries = _list_at(original, "$.requirements.acceptance_criteria")
        if ac_entries:
            stories[0].acceptance_criteria = list(ac_entries)
    elif _list_at(original, "$.requirements.acceptance_criteria") and stories:
        findings.add(
            Finding(
                severity=FindingSeverity.WARNING,
                code=FindingCode.AMBIGUOUS_DECOMPOSITION,
                message=(
                    "Acceptance criteria were not copied onto user stories because Copilot "
                    "does not bind them to a single functional requirement."
                ),
                source_references=[
                    _source_ref("$.requirements.acceptance_criteria", "acceptance_criteria")
                ],
                canonical_ids=[draft.canonical_id for draft in stories],
            )
        )

    drafts: list[_Draft] = []
    if epic is not None:
        drafts.append(epic)
    drafts.extend(features)
    drafts.extend(stories)
    drafts.extend(tasks)
    drafts.extend(subtasks)

    items = [_to_model(draft) for draft in drafts]
    _record_uncovered_requirements(original, items, findings)

    backlog = CanonicalBacklogV1(
        backlog_id=_backlog_id(original),
        items=items,
        source_contract_id=COPILOT_WORKFLOW_RESPONSE_V1,
        source_provenance=[
            _source_ref("$.user_request", "user_request", excerpt=_text_at(original, "$.user_request"))
        ],
    )
    return backlog, GenerationFindings(items=findings.items)


def _backlog_id(original: Mapping[str, Any]) -> str:
    """Envelope fingerprint, not item identity. Any envelope field change alters this ID."""

    canonical = json.dumps(original, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return canonical_id_from_source_key(f"backlog:{digest}")


def _collect_list_items(
    original: Mapping[str, Any],
    ids: _IdAllocator,
    findings: _FindingSink,
    specs: list[tuple[str, str, WorkItemType]],
) -> list[_Draft]:
    collected: list[_Draft] = []
    for base_path, field_name, item_type in specs:
        values = _list_at(original, base_path)
        for index, value in enumerate(values):
            text = value if isinstance(value, str) else str(value)
            occurrence = _occurrence(values, index, text)
            json_path = f"{base_path}[{index}]"
            draft = _try_source_item(
                original,
                ids,
                findings,
                base_path=base_path,
                json_path=json_path,
                field_name=field_name,
                item_type=item_type,
                text=text,
                occurrence=occurrence,
            )
            if draft is not None:
                collected.append(draft)
    return collected


def _occurrence(values: list[Any], index: int, text: str) -> int:
    normalized = normalize_source_text(text)
    return sum(
        1
        for previous in values[:index]
        if isinstance(previous, str) and normalize_source_text(previous) == normalized
    )


def _try_source_item(
    original: Mapping[str, Any],
    ids: _IdAllocator,
    findings: _FindingSink,
    *,
    base_path: str,
    json_path: str,
    field_name: str,
    item_type: WorkItemType,
    text: str,
    occurrence: int,
) -> _Draft | None:
    if not text.strip():
        return None
    classification = _classify_scope(original, text)
    ref = _source_ref(json_path, field_name, excerpt=text)
    if classification == "conflict":
        if normalize_source_text(text) not in findings.scope_conflict_texts:
            findings.add(
                Finding(
                    severity=FindingSeverity.ERROR,
                    code=FindingCode.SCOPE_CONFLICT,
                    message=(
                        "Source text matches both sow.in_scope and sow.out_of_scope; "
                        "no backlog item was generated."
                    ),
                    source_references=[ref, *_scope_refs(original, text)],
                )
            )
        return None
    if classification == "out":
        findings.add(
            Finding(
                severity=FindingSeverity.INFO,
                code=FindingCode.OUT_OF_SCOPE_EXCLUDED,
                message="Source text is explicitly out of scope and was not generated as a backlog item.",
                source_references=[ref, *_scope_refs(original, text, out_only=True)],
            )
        )
        return None
    identity_key = item_identity_key(base_path, text, occurrence)
    return _Draft(
        item_type=item_type,
        canonical_id=ids.allocate(identity_key, json_path),
        title=text,
        json_path=json_path,
        field_name=field_name,
        excerpt=text,
        identity_key=identity_key,
    )


def _link_unique_parent(
    parents: list[_Draft],
    children: list[_Draft],
    findings: _FindingSink,
    message: str,
) -> None:
    if not children:
        return
    if len(parents) == 1:
        parent = parents[0]
        for child in children:
            child.parent_id = parent.canonical_id
            parent.child_ids.append(child.canonical_id)
        return
    findings.add(
        Finding(
            severity=FindingSeverity.WARNING,
            code=FindingCode.AMBIGUOUS_DECOMPOSITION,
            message=message,
            source_references=[
                _source_ref(child.json_path, child.field_name, excerpt=child.excerpt)
                for child in children
            ],
            canonical_ids=[child.canonical_id for child in children],
        )
    )


def _record_uncovered_requirements(
    original: Mapping[str, Any],
    items: list[Any],
    findings: _FindingSink,
) -> None:
    covered_paths: set[str] = set()
    for item in items:
        if item.type is not WorkItemType.USER_STORY:
            continue
        for reference in item.provenance:
            path = reference.json_path
            if not path.startswith("$.requirements.functional_requirements"):
                continue
            resolved = resolve_json_path(original, path)
            if isinstance(resolved, str) and reference.excerpt == resolved:
                covered_paths.add(path)

    values = _list_at(original, "$.requirements.functional_requirements")
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            continue
        json_path = f"$.requirements.functional_requirements[{index}]"
        classification = _classify_scope(original, value)
        if classification in {"out", "conflict"}:
            continue
        if json_path in covered_paths:
            continue
        findings.add(
            Finding(
                severity=FindingSeverity.ERROR,
                code=FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT,
                message="Functional requirement has no traceable canonical user story provenance.",
                source_references=[
                    _source_ref(json_path, "functional_requirements", excerpt=value)
                ],
            )
        )


def _record_sow_list_conflicts(original: Mapping[str, Any], findings: _FindingSink) -> None:
    in_scope = _list_at(original, "$.sow.in_scope")
    out_scope = _list_at(original, "$.sow.out_of_scope")
    for in_index, in_text in enumerate(in_scope):
        if not isinstance(in_text, str):
            continue
        for out_index, out_text in enumerate(out_scope):
            if not isinstance(out_text, str):
                continue
            if normalize_source_text(in_text) != normalize_source_text(out_text):
                continue
            findings.scope_conflict_texts.add(normalize_source_text(in_text))
            findings.add(
                Finding(
                    severity=FindingSeverity.ERROR,
                    code=FindingCode.SCOPE_CONFLICT,
                    message="sow.in_scope and sow.out_of_scope contain the same statement.",
                    source_references=[
                        _source_ref(f"$.sow.in_scope[{in_index}]", "in_scope", excerpt=in_text),
                        _source_ref(
                            f"$.sow.out_of_scope[{out_index}]", "out_of_scope", excerpt=out_text
                        ),
                    ],
                )
            )


def _record_requirement_negations(original: Mapping[str, Any], findings: _FindingSink) -> None:
    requirements = _list_at(original, "$.requirements.functional_requirements")
    for left_index, left in enumerate(requirements):
        if not isinstance(left, str):
            continue
        for right_index, right in enumerate(requirements):
            if right_index <= left_index or not isinstance(right, str):
                continue
            if not _negation_pair(left, right):
                continue
            findings.add(
                Finding(
                    severity=FindingSeverity.ERROR,
                    code=FindingCode.COPILOT_CONTRADICTION,
                    message="Functional requirements contain an explicit shall / shall-not contradiction.",
                    source_references=[
                        _source_ref(
                            f"$.requirements.functional_requirements[{left_index}]",
                            "functional_requirements",
                            excerpt=left,
                        ),
                        _source_ref(
                            f"$.requirements.functional_requirements[{right_index}]",
                            "functional_requirements",
                            excerpt=right,
                        ),
                    ],
                )
            )


def _classify_scope(original: Mapping[str, Any], text: str) -> str:
    in_hits = [
        item
        for item in _list_at(original, "$.sow.in_scope")
        if isinstance(item, str) and _sow_line_applies_to_source(text, item)
    ]
    out_hits = [
        item
        for item in _list_at(original, "$.sow.out_of_scope")
        if isinstance(item, str) and _sow_line_applies_to_source(text, item)
    ]
    if in_hits and out_hits:
        return "conflict"
    if out_hits:
        return "out"
    return "in"


def _scope_refs(
    original: Mapping[str, Any], text: str, *, out_only: bool = False
) -> list[SourceReference]:
    refs: list[SourceReference] = []
    if not out_only:
        for index, item in enumerate(_list_at(original, "$.sow.in_scope")):
            if isinstance(item, str) and _sow_line_applies_to_source(text, item):
                refs.append(_source_ref(f"$.sow.in_scope[{index}]", "in_scope", excerpt=item))
    for index, item in enumerate(_list_at(original, "$.sow.out_of_scope")):
        if isinstance(item, str) and _sow_line_applies_to_source(text, item):
            refs.append(
                _source_ref(f"$.sow.out_of_scope[{index}]", "out_of_scope", excerpt=item)
            )
    return refs


def _sow_line_applies_to_source(source_text: str, sow_line: str) -> bool:
    """True when the SOW line equals the source, or the full SOW line (>=12) occurs in it."""

    normalized_source = normalize_source_text(source_text)
    normalized_sow = normalize_source_text(sow_line)
    if not normalized_source or not normalized_sow:
        return False
    if normalized_source == normalized_sow:
        return True
    return len(normalized_sow) >= _MIN_SOW_LINE_LEN and normalized_sow in normalized_source


def _negation_pair(left: str, right: str) -> bool:
    normalized_left = normalize_source_text(left)
    normalized_right = normalize_source_text(right)
    pairs = (
        ("the system shall not ", "the system shall "),
        ("shall not ", "shall "),
        ("must not ", "must "),
    )
    for negative, positive in pairs:
        if (
            normalized_left.startswith(negative)
            and normalized_right.startswith(positive)
            and normalized_left[len(negative) :] == normalized_right[len(positive) :]
        ):
            return True
        if (
            normalized_right.startswith(negative)
            and normalized_left.startswith(positive)
            and normalized_right[len(negative) :] == normalized_left[len(positive) :]
        ):
            return True
    return False


def _source_ref(json_path: str, field_name: str | None, excerpt: str | None = None) -> SourceReference:
    return SourceReference(
        contract_id=COPILOT_WORKFLOW_RESPONSE_V1,
        json_path=json_path,
        field_name=field_name,
        excerpt=excerpt,
    )


def _list_at(original: Mapping[str, Any], json_path: str) -> list[Any]:
    value = resolve_json_path(original, json_path)
    return value if isinstance(value, list) else []


def _text_at(original: Mapping[str, Any], json_path: str) -> str | None:
    value = resolve_json_path(original, json_path)
    return value if isinstance(value, str) else None


def resolve_json_path(document: Mapping[str, Any] | list[Any], json_path: str) -> Any:
    if not json_path.startswith("$"):
        raise ValueError(f"unsupported JSONPath {json_path!r}")
    cursor: Any = document
    for match in _JSON_PATH_TOKEN.finditer(json_path[1:]):
        name, index = match.group(1), match.group(2)
        if name is not None:
            if not isinstance(cursor, Mapping) or name not in cursor:
                return None
            cursor = cursor[name]
        else:
            position = int(index)
            if not isinstance(cursor, list) or position >= len(cursor):
                return None
            cursor = cursor[position]
    return cursor


def _to_model(draft: _Draft) -> Epic | Feature | UserStory | Task | Subtask:
    constructor = _CONSTRUCTORS[draft.item_type]
    return constructor(
        canonical_id=draft.canonical_id,
        title=draft.title,
        description="",
        status=WorkItemStatus.TODO,
        approval_state=ApprovalState.NOT_APPROVED,
        parent_id=draft.parent_id,
        child_ids=list(draft.child_ids),
        acceptance_criteria=list(draft.acceptance_criteria),
        provenance=[
            _source_ref(draft.json_path, draft.field_name, excerpt=draft.excerpt)
        ],
    )
