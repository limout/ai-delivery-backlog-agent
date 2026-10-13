"""LLM-backed backlog decomposition with deterministic Python validation."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Mapping

from pydantic import ValidationError

from backlog_agent.backlog.generate import (
    _CONSTRUCTORS,
    _FindingSink,
    _IdAllocator,
    _backlog_id,
    _record_requirement_negations,
    _record_sow_list_conflicts,
    _source_ref,
    resolve_json_path,
)
from backlog_agent.backlog.ids import item_identity_key, normalize_source_text
from backlog_agent.backlog.models import (
    ApprovalState,
    CanonicalBacklogV1,
    WorkItemStatus,
    WorkItemType,
)
from backlog_agent.backlog.granularity import (
    functional_requirement_paths,
    looks_like_implementation_work,
    record_granularity_findings,
    task_restates_parent,
)
from backlog_agent.backlog.proposal import (
    BacklogProposal,
    ContentOrigin,
    ProposedItem,
    ProposedProvenance,
    proposal_json_schema,
)
from backlog_agent.findings.models import (
    Finding,
    FindingCode,
    FindingSeverity,
    GenerationFindings,
    LlmCallDiagnostic,
)
from backlog_agent.llm.exceptions import DecompositionValidationError
from backlog_agent.llm.provider import AIProvider
from backlog_agent.source import load_project_source
from backlog_agent.source.model import (
    NormalizedProjectSource,
    claim_content_tokens,
    claims_are_related,
    classify_claim,
    classify_scope,
    fact_texts,
    iter_source_statements,
    looks_like_delivery_process_constraint,
    looks_like_timeline_or_estimate,
    statement_polarity,
)
from backlog_agent.source.request_headings import (
    heading_appears_represented,
    parse_labeled_request_outcomes,
)

_PARENT_TYPE: dict[WorkItemType, WorkItemType | None] = {
    WorkItemType.EPIC: None,
    WorkItemType.FEATURE: WorkItemType.EPIC,
    WorkItemType.USER_STORY: WorkItemType.FEATURE,
    WorkItemType.TASK: WorkItemType.USER_STORY,
    WorkItemType.SUBTASK: WorkItemType.TASK,
}

_LOG = logging.getLogger("backlog_agent.decompose")

_QUALITY_REPAIR_CODES = frozenset(
    {
        FindingCode.OVERSIZED_STORY,
        FindingCode.MISSING_IMPLEMENTATION_DETAIL,
        FindingCode.OVERLAPPING_STORY_RESPONSIBILITY,
        FindingCode.DUPLICATE_ACCEPTANCE_CRITERION,
        FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT,
        FindingCode.FEATURE_CHILD_MISALIGNMENT,
    }
)

_FUNCTIONAL_REQUIREMENT_PATH = "$.requirements.functional_requirements"
_NON_FUNCTIONAL_REQUIREMENT_PATH = "$.requirements.non_functional_requirements"

_GENERIC_TASK_TITLES = frozenset(
    {
        "implementation",
        "implement",
        "development",
        "testing",
        "qa",
        "backend",
        "frontend",
        "misc",
        "other",
        "todo",
        "tbd",
        "placeholder",
        "work",
        "task",
        "subtask",
        "do work",
        "implement feature",
        "implement backend",
        "do testing",
        "handle errors",
        "error handling",
        "backend work",
    }
)

_SENSITIVE_ERROR = re.compile(
    r"(?i)(?:api[_-]?key|secret|password|token|authorization)\s*[:=]\s*\S+"
    r"|bearer\s+\S+"
    r"|AIza[0-9A-Za-z_\-]{8,}"
)

def decompose_backlog(
    envelope: Mapping[str, Any] | object,
    provider: AIProvider,
    *,
    repair: bool = True,
) -> tuple[CanonicalBacklogV1, GenerationFindings]:
    """Propose a canonical backlog via ``provider``, then validate it in Python.

    At most one repair call is made in addition to the initial generation:
    a hard-validation repair if the first proposal fails, or a quality repair
    if it succeeds with actionable quality findings. Never both. The model
    cannot mark the result valid; ``CanonicalBacklogV1`` and coverage checks
    remain deterministic.
    """

    source = load_project_source(envelope)
    schema = proposal_json_schema()
    prompt = _build_prompt(source)
    calls: list[LlmCallDiagnostic] = []
    raw, result = _invoke_provider(provider, prompt, schema, source, calls, "initial")

    if result.ok:
        quality = _actionable_quality_findings(result.findings)
        if repair and quality:
            snapshot = _fr_coverage_snapshot(result.proposal)
            quality_prompt = _build_quality_repair_prompt(
                prompt,
                raw,
                quality,
                snapshot,
                _uncovered_narrative_outcomes(source, result.findings),
            )
            _repaired, second = _invoke_provider(
                provider, quality_prompt, schema, source, calls, "quality_repair"
            )
            if second.ok:
                _mark_accepted(calls, "quality_repair")
                return _accepted(second, calls)
            recovered = _recover_quality_repair(source, second)
            if recovered is not None:
                _mark_accepted(calls, "quality_repair")
                return _accepted(recovered, calls)
            _record_lost_fr_paths(calls, result.proposal, second.proposal)
            _record_quality_repair_rejected(result, second)
            _mark_accepted(calls, "initial")
            return _accepted(result, calls)
        _mark_accepted(calls, "initial")
        return _accepted(result, calls)

    if repair:
        repair_prompt = _build_repair_prompt(prompt, raw, result.errors)
        _repaired, second = _invoke_provider(
            provider, repair_prompt, schema, source, calls, "hard_repair"
        )
        if second.ok:
            _note_dropped_grounded_tasks(source, raw, second)
            _mark_accepted(calls, "hard_repair")
            return _accepted(second, calls)
        _record_lost_fr_paths(calls, result.proposal, second.proposal)
        result = second

    findings = result.findings
    findings.items.append(
        Finding(
            severity=FindingSeverity.ERROR,
            code=FindingCode.LLM_VALIDATION_FAILED,
            message="LLM backlog proposal failed independent validation after at most one repair.",
        )
    )
    raise DecompositionValidationError(
        result.errors, GenerationFindings(items=findings.items, llm_calls=calls)
    )


class _MaterializeResult:
    def __init__(
        self,
        *,
        ok: bool,
        errors: list[str],
        findings: _FindingSink,
        backlog: CanonicalBacklogV1 | None = None,
        parsed: bool = True,
        proposal: BacklogProposal | None = None,
    ) -> None:
        self.ok = ok
        self.errors = errors
        self.findings = findings
        self.backlog = backlog
        self.parsed = parsed
        self.proposal = proposal


def _try_materialize(source: NormalizedProjectSource, raw: Mapping[str, Any]) -> _MaterializeResult:
    findings = _FindingSink()
    original = source.original
    _record_sow_list_conflicts(original, findings)
    _record_requirement_negations(original, findings)
    errors: list[str] = []

    try:
        proposal = BacklogProposal.model_validate(raw)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(part) for part in err.get("loc", ()))
            errors.append(f"{loc}: {err.get('msg')}")
        return _MaterializeResult(
            ok=False,
            errors=errors or ["proposal is not a valid object"],
            findings=findings,
            parsed=False,
        )

    errors.extend(_structural_errors(proposal))
    errors.extend(_provenance_and_scope_errors(source, proposal, findings))
    errors.extend(_execution_quality_errors(source, proposal))
    if not errors:
        _relocate_cross_cutting_constraints(source, proposal)
        if not any(item.type is WorkItemType.TASK for item in proposal.items):
            injected = _attach_grounded_implementation_tasks(source, proposal)
            if injected:
                extra = _execution_quality_errors(source, proposal)
                if extra:
                    drop = {item.local_id for item in injected}
                    proposal.items = [item for item in proposal.items if item.local_id not in drop]
    if errors:
        return _MaterializeResult(ok=False, errors=errors, findings=findings, proposal=proposal)

    try:
        backlog = _to_canonical(source, proposal, findings)
    except (ValidationError, ValueError) as exc:
        errors.append(str(exc))
        return _MaterializeResult(ok=False, errors=errors, findings=findings, proposal=proposal)

    _record_uncovered_functional_requirements(source, backlog.items, findings)
    uncovered = [
        item for item in findings.items if item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT
    ]
    if uncovered:
        for item in uncovered:
            excerpts = [ref.excerpt for ref in item.source_references if ref.excerpt]
            errors.append(
                "Uncovered functional requirement: " + (excerpts[0] if excerpts else item.message)
            )
        return _MaterializeResult(
            ok=False, errors=errors, findings=findings, proposal=proposal, backlog=backlog
        )

    _record_generated_content(source, proposal, backlog, findings)
    _record_uncertainties(proposal, backlog, findings)
    _record_prerequisites(source, proposal, findings)
    _record_requirement_coverage_findings(source, backlog.items, findings)
    record_granularity_findings(source, backlog, findings, proposal=proposal)
    _record_uncovered_request_outcomes(source, backlog, findings, proposal)
    return _MaterializeResult(
        ok=True, errors=[], findings=findings, backlog=backlog, proposal=proposal
    )


def _structural_errors(proposal: BacklogProposal) -> list[str]:
    errors: list[str] = []
    by_id = {item.local_id: item for item in proposal.items}
    if not any(item.type is WorkItemType.EPIC for item in proposal.items):
        errors.append("proposal must include at least one epic")

    for item in proposal.items:
        expected = _PARENT_TYPE[item.type]
        if item.type is WorkItemType.EPIC and item.parent_local_id is not None:
            errors.append(f"{item.local_id}: epics cannot have a parent")
            continue
        if item.parent_local_id is None:
            continue
        parent = by_id.get(item.parent_local_id)
        if parent is None:
            errors.append(
                f"{item.local_id}: parent_local_id {item.parent_local_id!r} does not exist"
            )
            continue
        if expected is not None and parent.type is not expected:
            errors.append(
                f"{item.local_id}: parent must be a {expected.value}, got {parent.type.value}"
            )
        if item.local_id == item.parent_local_id:
            errors.append(f"{item.local_id}: cannot parent itself")

        for dep in item.depends_on_local_ids:
            if dep not in by_id:
                errors.append(f"{item.local_id}: depends_on_local_ids {dep!r} does not exist")
            if dep == item.local_id:
                errors.append(f"{item.local_id}: cannot depend on itself")

    errors.extend(_cycle_errors(proposal.items, parent=True))
    errors.extend(_cycle_errors(proposal.items, parent=False))
    return errors


def _cycle_errors(items: list[ProposedItem], *, parent: bool) -> list[str]:
    by_id = {item.local_id: item for item in items}
    visiting: set[str] = set()
    visited: set[str] = set()
    kind = "parent-child" if parent else "dependency"
    found: list[str] = []

    def walk(item_id: str) -> None:
        if item_id in visited or found:
            return
        if item_id in visiting:
            found.append(f"proposal contains a {kind} cycle")
            return
        visiting.add(item_id)
        item = by_id[item_id]
        nxt = [item.parent_local_id] if parent and item.parent_local_id else list(item.depends_on_local_ids)
        for other in nxt:
            if other in by_id:
                walk(other)
        visiting.remove(item_id)
        visited.add(item_id)

    for item_id in by_id:
        walk(item_id)
    return found


def _path_allowed(source: NormalizedProjectSource, json_path: str) -> bool:
    for prefix in source.allowed_path_prefixes:
        if json_path == prefix or json_path.startswith(prefix + ".") or json_path.startswith(prefix + "["):
            return True
    return False


def _provenance_and_scope_errors(
    source: NormalizedProjectSource,
    proposal: BacklogProposal,
    findings: _FindingSink,
) -> list[str]:
    errors: list[str] = []
    original = source.original
    for item in proposal.items:
        errors.extend(
            _provenance_ref_errors(
                source,
                item.provenance,
                item.local_id,
                findings,
            )
        )

        classification = classify_scope(source, item.title)
        if classification == "out":
            errors.append(
                f"{item.local_id}: title matches an out_of_scope statement and must not appear in the backlog"
            )
            findings.add(
                Finding(
                    severity=FindingSeverity.ERROR,
                    code=FindingCode.OUT_OF_SCOPE_EXCLUDED,
                    message="LLM proposed an out-of-scope item; it was rejected.",
                    source_references=[
                        _source_ref(ref.json_path, ref.field_name, excerpt=ref.excerpt)
                        for ref in item.provenance
                    ],
                )
            )
        elif classification == "conflict":
            errors.append(f"{item.local_id}: title matches both sow.in_scope and sow.out_of_scope")

        if item.title_origin is ContentOrigin.SOURCE:
            excerpts = [ref.excerpt for ref in item.provenance if ref.excerpt]
            if _title_is_exact_source_copy(item.title, excerpts):
                pass
            elif _title_is_grounded_paraphrase(item.title, excerpts):
                # A title that restates a cited excerpt without copying it is generated.
                # Correct the origin and keep the real provenance excerpt.
                item.title_origin = ContentOrigin.GENERATED
            else:
                errors.append(
                    f"{item.local_id}: title_origin=source but title is not equal to any provenance excerpt"
                )

        errors.extend(_origin_value_errors(source, item))

    for index, prerequisite in enumerate(proposal.prerequisites):
        errors.extend(
            _provenance_ref_errors(
                source,
                prerequisite.provenance,
                f"prerequisites[{index}]",
                findings,
            )
        )
    return errors


def _to_canonical(
    source: NormalizedProjectSource,
    proposal: BacklogProposal,
    findings: _FindingSink,
) -> CanonicalBacklogV1:
    ids = _IdAllocator(findings)
    occurrence_counts: dict[tuple[str, str], int] = {}
    local_to_canonical: dict[str, str] = {}
    prepared: list[tuple[ProposedItem, str, list[Any]]] = []

    for item in proposal.items:
        primary = item.provenance[0]
        if item.title_origin is ContentOrigin.SOURCE:
            base_path = _base_path(primary.json_path)
            identity_text = item.title
        else:
            base_path = "$.llm.generated"
            identity_text = item.title
        key = (base_path, identity_text)
        occurrence = occurrence_counts.get(key, 0)
        occurrence_counts[key] = occurrence + 1
        identity_key = item_identity_key(base_path, identity_text, occurrence)
        canonical_id = ids.allocate(identity_key, primary.json_path)
        local_to_canonical[item.local_id] = canonical_id
        provenance = [
            _source_ref(ref.json_path, ref.field_name, excerpt=ref.excerpt) for ref in item.provenance
        ]
        prepared.append((item, canonical_id, provenance))

    children: dict[str, list[str]] = {canonical_id: [] for _item, canonical_id, _p in prepared}
    parent_of: dict[str, str | None] = {}
    for item, canonical_id, _provenance in prepared:
        parent_id = local_to_canonical.get(item.parent_local_id) if item.parent_local_id else None
        parent_of[canonical_id] = parent_id
        if parent_id is not None:
            children[parent_id].append(canonical_id)

    models = []
    for item, canonical_id, provenance in prepared:
        constructor = _CONSTRUCTORS[item.type]
        dependencies = [local_to_canonical[dep] for dep in item.depends_on_local_ids]
        models.append(
            constructor(
                canonical_id=canonical_id,
                title=item.title,
                description=item.description,
                status=WorkItemStatus.TODO,
                approval_state=ApprovalState.NOT_APPROVED,
                parent_id=parent_of[canonical_id],
                child_ids=list(children[canonical_id]),
                acceptance_criteria=list(item.acceptance_criteria),
                test_requirements=list(item.test_requirements),
                dependencies=dependencies,
                provenance=provenance,
            )
        )

    return CanonicalBacklogV1(
        backlog_id=_backlog_id(source.original),
        items=models,
        source_contract_id=source.contract_id,
        source_provenance=[
            _source_ref(
                source.request.json_path,
                source.request.field_name,
                excerpt=source.request.text or None,
            )
        ],
    )


def _base_path(json_path: str) -> str:
    bracket = json_path.find("[")
    return json_path[:bracket] if bracket >= 0 else json_path


def _provenance_ref_errors(
    source: NormalizedProjectSource,
    refs: list[Any],
    owner: str,
    findings: _FindingSink,
) -> list[str]:
    errors: list[str] = []
    original = source.original
    for index, ref in enumerate(refs):
        label = f"{owner}.provenance[{index}]"
        if not _path_allowed(source, ref.json_path):
            errors.append(f"{label}: json_path {ref.json_path!r} is not an allowed source field")
            continue
        resolved = resolve_json_path(original, ref.json_path)
        if resolved is None:
            errors.append(f"{label}: json_path {ref.json_path!r} does not resolve in the envelope")
            findings.add(
                Finding(
                    severity=FindingSeverity.ERROR,
                    code=FindingCode.UNGROUNDED_PROVENANCE,
                    message=f"Provenance path {ref.json_path!r} does not exist on the Copilot envelope.",
                    source_references=[_source_ref(ref.json_path, ref.field_name, ref.excerpt)],
                )
            )
            continue
        if not isinstance(resolved, str):
            if ref.excerpt is not None:
                errors.append(f"{label}: excerpt was provided but {ref.json_path} is not a string")
            continue
        if ref.excerpt is None:
            if "[" in ref.json_path:
                errors.append(
                    f"{label}: array provenance requires an excerpt equal to the source element"
                )
            else:
                ref.excerpt = resolved
            continue
        if ref.excerpt != resolved:
            # Unique scalar fields are identified by path. A $.user_request excerpt
            # may instead be one exact labelled section published as narrative
            # evidence. Any other scalar mismatch is rewritten to the full field.
            # Array elements still require an exact excerpt.
            if "[" not in ref.json_path:
                if _is_exact_narrative_excerpt(source, ref.json_path, ref.excerpt):
                    continue
                ref.excerpt = resolved
                continue
            errors.append(f"{label}: excerpt does not equal the source value at {ref.json_path}")
            findings.add(
                Finding(
                    severity=FindingSeverity.ERROR,
                    code=FindingCode.UNGROUNDED_PROVENANCE,
                    message="Provenance excerpt does not match the Copilot field value.",
                    source_references=[_source_ref(ref.json_path, ref.field_name, excerpt=ref.excerpt)],
                )
            )
    return errors


def _origin_value_errors(source: NormalizedProjectSource, item: ProposedItem) -> list[str]:
    errors: list[str] = []
    source_acs = fact_texts(source.acceptance_criteria)
    for index, criterion in enumerate(item.acceptance_criteria):
        origin = _list_origin(item.acceptance_criteria_origins, index, criterion, source_acs)
        if origin is ContentOrigin.SOURCE and normalize_source_text(criterion) not in source_acs:
            # Edited or scoped criterion text is generated. Keep provenance paths/excerpts;
            # do not treat rewritten text as a source copy.
            item.acceptance_criteria_origins[index] = ContentOrigin.GENERATED
    source_tests: set[str] = set()
    for index, requirement in enumerate(item.test_requirements):
        origin = _list_origin(item.test_requirements_origins, index, requirement, source_tests)
        if origin is ContentOrigin.SOURCE:
            errors.append(
                f"{item.local_id}.test_requirements[{index}]: test requirements are verification activities and cannot use origin=source unless Copilot listed them"
            )
        if normalize_source_text(requirement) in source_acs:
            errors.append(
                f"{item.local_id}.test_requirements[{index}]: test requirement restates an acceptance criterion; keep verification distinct"
            )
    return errors


def _list_origin(
    origins: list[ContentOrigin],
    index: int,
    text: str,
    source_values: set[str],
) -> ContentOrigin:
    if origins:
        return origins[index]
    if normalize_source_text(text) in source_values:
        return ContentOrigin.SOURCE
    return ContentOrigin.GENERATED


def _execution_quality_errors(source: NormalizedProjectSource, proposal: BacklogProposal) -> list[str]:
    errors: list[str] = []
    workstreams = fact_texts(source.workstreams)
    blocking = fact_texts(source.blocking_dependencies)
    timeline_texts = fact_texts(
        tuple(fact for fact in source.non_functional_requirements if fact.is_timeline_or_estimate)
    )
    seen_task_titles: dict[str, str] = {}
    by_local = {item.local_id: item for item in proposal.items}

    for item in proposal.items:
        if item.type is WorkItemType.USER_STORY:
            if not item.description.strip():
                errors.append(f"{item.local_id}: user stories must include a useful description")
            elif _is_title_restatement(item.title, item.description):
                errors.append(
                    f"{item.local_id}: description restates the title; describe actor, behaviour, and outcome"
                )
            paths = [reference.json_path for reference in item.provenance]
            has_functional = any(_is_requirement_path(path, _FUNCTIONAL_REQUIREMENT_PATH) for path in paths)
            has_non_functional = any(
                _is_requirement_path(path, _NON_FUNCTIONAL_REQUIREMENT_PATH) for path in paths
            )
            if has_non_functional and not has_functional:
                errors.append(
                    f"{item.local_id}: user stories must not be created from a non-functional "
                    "requirement alone; attach that constraint's path and excerpt to a story "
                    "that cites a functional requirement"
                )
            fr_paths = sorted(set(functional_requirement_paths(item)))
            if len(fr_paths) >= 2:
                errors.append(
                    f"{item.local_id}: user stories must not combine independently testable "
                    "outcomes from distinct functional requirements; keep one functional "
                    "requirement path per story and cover each remaining requirement with its "
                    "own story. Additional NFR or acceptance-criterion paths on the same story "
                    "are allowed"
                )

        if item.type is WorkItemType.FEATURE and normalize_source_text(item.title) in workstreams:
            errors.append(
                f"{item.local_id}: delivery workstreams must not be copied as features"
            )

        normalized_title = normalize_source_text(item.title)
        if any(
            normalized_title == constraint or constraint in normalized_title
            for constraint in timeline_texts
            if len(constraint) >= 12
        ):
            errors.append(
                f"{item.local_id}: do not convert a timeline or estimate constraint into a committed backlog item"
            )

        if item.type in {WorkItemType.TASK, WorkItemType.SUBTASK}:
            normalized = normalize_source_text(item.title)
            if normalized in _GENERIC_TASK_TITLES or len(normalized) < 12:
                errors.append(f"{item.local_id}: generic or filler task title {item.title!r}")
            parent = by_local.get(item.parent_local_id) if item.parent_local_id else None
            if task_restates_parent(item, parent):
                errors.append(
                    f"{item.local_id}: task restates the parent item; split into concrete engineering work"
                )
            if normalized in workstreams:
                errors.append(
                    f"{item.local_id}: delivery workstreams must not be copied as tasks"
                )
            if normalized in blocking:
                status = classify_claim(item.title, source)
                if status == "established":
                    errors.append(
                        f"{item.local_id}: do not add a task whose purpose is to confirm an "
                        "established source fact"
                    )
                else:
                    errors.append(
                        f"{item.local_id}: source blocking dependency must be an unresolved "
                        "prerequisite, not an implementation task"
                    )
            elif _task_confirms_established_fact(source, item):
                errors.append(
                    f"{item.local_id}: do not add a task whose purpose is to confirm an "
                    "established source fact"
                )
            owner = seen_task_titles.get(normalized)
            if owner is not None:
                errors.append(
                    f"{item.local_id}: duplicate task title of {owner}; combine or specialize the work"
                )
            else:
                seen_task_titles[normalized] = item.local_id

    return errors


def _title_is_exact_source_copy(title: str, excerpts: list[str]) -> bool:
    normalized_title = normalize_source_text(title)
    if not normalized_title:
        return False
    return any(
        title == excerpt or normalized_title == normalize_source_text(excerpt)
        for excerpt in excerpts
        if excerpt
    )


_TITLE_GENERIC = frozenset(
    {
        "allow",
        "allows",
        "customer",
        "customers",
        "eligible",
        "member",
        "members",
        "must",
        "online",
        "shall",
        "system",
        "their",
        "user",
        "users",
    }
)
_DISPLAY_TITLE_TOKENS = frozenset(
    {"browse", "display", "list", "listing", "read", "review", "show", "view"}
)


def _title_stem(token: str) -> str:
    if token.endswith("ies") and len(token) > 5:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 4:
        return token[:-1]
    return token


def _title_content_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in normalize_source_text(text).split():
        token = _title_stem(raw.strip(".,;:!?\"'`()[]{}"))
        if len(token) >= 4 and token not in _TITLE_GENERIC:
            tokens.append(token)
    return tokens


def _title_tokens_related(left: str, right: str) -> bool:
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
    if extra == "s" and len(short) >= 4:
        return True
    return False


def _title_token_covered(token: str, excerpt_tokens: set[str]) -> bool:
    return any(_title_tokens_related(token, other) for other in excerpt_tokens)


def _title_is_grounded_paraphrase(title: str, excerpts: list[str]) -> bool:
    normalized_title = normalize_source_text(title)
    if len(normalized_title) < 12:
        return False
    title_tokens = _title_content_tokens(title)
    for excerpt in excerpts:
        if not excerpt:
            continue
        normalized_excerpt = normalize_source_text(excerpt)
        if not normalized_excerpt:
            continue
        if normalized_title in normalized_excerpt:
            return True
        if len(normalized_excerpt) >= 12 and normalized_excerpt in normalized_title:
            return True
        excerpt_tokens = set(_title_content_tokens(excerpt))
        if len(title_tokens) < 2 or len(excerpt_tokens) < 2:
            continue
        matched = [token for token in title_tokens if _title_token_covered(token, excerpt_tokens)]
        unmatched = [token for token in title_tokens if token not in matched]
        if len(matched) < 2:
            continue
        if not unmatched:
            return True
        excerpt_has_display = any(token in _DISPLAY_TITLE_TOKENS for token in excerpt_tokens)
        if excerpt_has_display and all(token in _DISPLAY_TITLE_TOKENS for token in unmatched):
            return True
    return False


def _is_requirement_path(json_path: str, prefix: str) -> bool:
    return json_path == prefix or json_path.startswith(prefix + "[")


def _is_title_restatement(title: str, description: str) -> bool:
    normalized_title = normalize_source_text(title)
    normalized_description = normalize_source_text(description)
    if not normalized_description or not normalized_title:
        return False
    if normalized_description == normalized_title:
        return True
    prefixes = (
        "user story ",
        "user story to ",
        "user story allowing ",
        "feature ",
        "feature for ",
        "feature allowing ",
        "feature ensuring ",
        "task ",
        "story ",
    )
    for prefix in prefixes:
        remainder = normalized_description[len(prefix) :] if normalized_description.startswith(prefix) else ""
        if remainder and (remainder == normalized_title or normalized_title in remainder):
            return True
    return False


def _record_generated_content(
    source: NormalizedProjectSource,
    proposal: BacklogProposal,
    backlog: CanonicalBacklogV1,
    findings: _FindingSink,
) -> None:
    source_acs = fact_texts(source.acceptance_criteria)
    by_local = {item.local_id: item for item in proposal.items}
    local_ids = list(by_local)
    for index, canonical in enumerate(backlog.items):
        proposed = by_local[local_ids[index]]
        origins: list[str] = []
        if proposed.title_origin is ContentOrigin.GENERATED:
            origins.append("title")
        if proposed.description and proposed.description_origin is ContentOrigin.GENERATED:
            origins.append("description")
        generated_ac = [
            criterion
            for ac_index, criterion in enumerate(proposed.acceptance_criteria)
            if _list_origin(proposed.acceptance_criteria_origins, ac_index, criterion, source_acs)
            is ContentOrigin.GENERATED
        ]
        if generated_ac:
            origins.append("acceptance_criteria")
        if proposed.test_requirements:
            origins.append("test_requirements")
        if not origins:
            continue
        findings.add(
            Finding(
                severity=FindingSeverity.INFO,
                code=FindingCode.GENERATED_CONTENT,
                message="Item contains AI-generated or inferred text: " + ", ".join(origins) + ".",
                canonical_ids=[canonical.canonical_id],
                source_references=list(canonical.provenance),
            )
        )


_VERIFY_VERB = re.compile(
    r"\b(investigate|confirm|verify|check|determine|validate|assess|discover)\b",
    re.IGNORECASE,
)
_EXISTENCE_CHECK = re.compile(
    r"\b(whether|if)\b.{0,80}\b("
    r"available|availability|exists?|existence|access|granted|"
    r"documentation|participate|participation|provided"
    r")\b",
    re.IGNORECASE,
)
_AVAILABILITY_NOUN = re.compile(
    r"\b(available|availability|exists?|existence|access|granted|"
    r"documentation|participate|participation|provided)\b",
    re.IGNORECASE,
)
_REMAINING_DETAIL_TOKENS = frozenset(
    {
        "authentication",
        "authorization",
        "behavior",
        "behaviour",
        "capability",
        "capabilities",
        "credential",
        "credentials",
        "endpoint",
        "environment",
        "idempotency",
        "mapping",
        "oauth",
        "payload",
        "sandbox",
        "schema",
        "sso",
        "token",
        "webhook",
    }
)


def _cited_source_facts(source: NormalizedProjectSource, item: ProposedItem) -> list[Any]:
    by_path = {
        fact.json_path: fact
        for fact in (
            *source.blocking_dependencies,
            *source.unknowns,
            *source.assumptions,
        )
    }
    found: list[Any] = []
    for reference in item.provenance:
        fact = by_path.get(reference.json_path)
        if fact is not None:
            found.append(fact)
    return found


def _remaining_implementation_facts(source: NormalizedProjectSource) -> list[Any]:
    remaining: list[Any] = []
    for fact in (*source.unknowns, *source.blocking_dependencies):
        if classify_claim(fact.text, source, skip_path=fact.json_path) == "established":
            continue
        remaining.append(fact)
    return remaining


_TARGET_STOPWORDS = frozenset(
    {
        "available",
        "customer",
        "customers",
        "existing",
        "let",
        "member",
        "members",
        "must",
        "operator",
        "operators",
        "resident",
        "shall",
        "specified",
        "student",
        "students",
        "system",
        "tenant",
        "unconfirmed",
        "unspecified",
        "user",
        "users",
    }
)


def _distinctive_tokens(text: str) -> set[str]:
    return {token for token in claim_content_tokens(text) if token not in _TARGET_STOPWORDS}


def _token_sets_share_stem(left: set[str], right: set[str]) -> bool:
    left = {token for token in left if token not in _TARGET_STOPWORDS}
    right = {token for token in right if token not in _TARGET_STOPWORDS}
    if left & right:
        return True
    for token in left:
        for other in right:
            if min(len(token), len(other)) < 3:
                continue
            if token.endswith("s") and token[:-1] == other:
                return True
            if other.endswith("s") and other[:-1] == token:
                return True
    return False


def _matched_tokens(left: set[str], right: set[str]) -> set[str]:
    left = {token for token in left if token not in _TARGET_STOPWORDS}
    right = {token for token in right if token not in _TARGET_STOPWORDS}
    matched = set(left & right)
    for token in left:
        for other in right:
            if min(len(token), len(other)) < 3:
                continue
            if token.endswith("s") and token[:-1] == other:
                matched.add(token)
            if other.endswith("s") and other[:-1] == token:
                matched.add(other)
    return matched


def _fact_targets_story(fact: Any, story: ProposedItem) -> bool:
    blob = f"{story.title} {story.description}"
    if claims_are_related(fact.text, blob) or claims_are_related(blob, fact.text):
        return True
    title_hits = _matched_tokens(_distinctive_tokens(fact.text), _distinctive_tokens(story.title))
    if title_hits:
        return True
    blob_hits = _matched_tokens(_distinctive_tokens(fact.text), _distinctive_tokens(blob))
    return len(blob_hits) >= 2 or any(len(token) >= 6 for token in blob_hits)


def _attach_grounded_implementation_tasks(
    source: NormalizedProjectSource, proposal: BacklogProposal
) -> list[ProposedItem]:
    """Add bounded tasks for unresolved source work. Never re-check established facts."""

    stories = [item for item in proposal.items if item.type is WorkItemType.USER_STORY]
    tasked = {
        item.parent_local_id
        for item in proposal.items
        if item.type is WorkItemType.TASK and item.parent_local_id
    }
    cited_paths = {
        reference.json_path
        for item in proposal.items
        if item.type is WorkItemType.TASK
        for reference in item.provenance
    }
    used_ids = {item.local_id for item in proposal.items}
    used_titles = {normalize_source_text(item.title) for item in proposal.items}
    injected: list[ProposedItem] = []
    for index, fact in enumerate(_remaining_implementation_facts(source)):
        if fact.json_path in cited_paths:
            continue
        candidates = [story for story in stories if _fact_targets_story(fact, story)]
        if not candidates:
            continue
        titled = [
            story
            for story in candidates
            if _token_sets_share_stem(_distinctive_tokens(fact.text), _distinctive_tokens(story.title))
        ]
        pool = titled or candidates
        impl = [story for story in pool if looks_like_implementation_work(story)]
        parent = next((story for story in impl if story.local_id not in tasked), None)
        if parent is None:
            parent = next((story for story in pool if story.local_id not in tasked), None)
        if parent is None:
            continue
        detail = " ".join(str(fact.text).split())
        title = f"Clarify remaining implementation detail: {detail}"
        if len(title) > 180:
            title = title[:177].rstrip() + "..."
        if normalize_source_text(title) in used_titles or len(normalize_source_text(title)) < 12:
            continue
        local_id = f"task-grounded-{index + 1}"
        suffix = 2
        while local_id in used_ids:
            local_id = f"task-grounded-{index + 1}-{suffix}"
            suffix += 1
        item = ProposedItem(
            local_id=local_id,
            type=WorkItemType.TASK,
            title=title,
            description=(
                "Resolve this remaining source detail for the parent story. "
                "Do not treat a confirmed source fact as still unknown."
            ),
            parent_local_id=parent.local_id,
            provenance=[
                ProposedProvenance(
                    json_path=fact.json_path,
                    field_name=fact.field_name,
                    excerpt=fact.text,
                )
            ],
            title_origin=ContentOrigin.GENERATED,
            description_origin=ContentOrigin.GENERATED,
        )
        if task_restates_parent(item, parent) or _task_confirms_established_fact(source, item):
            continue
        proposal.items.append(item)
        injected.append(item)
        used_ids.add(local_id)
        used_titles.add(normalize_source_text(title))
        tasked.add(parent.local_id)
        cited_paths.add(fact.json_path)
    return injected


def _task_is_existence_or_availability_check(text: str) -> bool:
    if _EXISTENCE_CHECK.search(text):
        return True
    return bool(_VERIFY_VERB.search(text) and _AVAILABILITY_NOUN.search(text))


def _related_statement_polarities(source: NormalizedProjectSource, text: str) -> set[str]:
    polarities: set[str] = set()
    for _path, statement in iter_source_statements(source):
        polarity = statement_polarity(statement)
        if polarity is None or not claims_are_related(text, statement):
            continue
        polarities.add(polarity)
    return polarities


def _task_has_unresolved_remaining_purpose(
    source: NormalizedProjectSource,
    item: ProposedItem,
    cited: list[Any],
) -> bool:
    combined = f"{item.title} {item.description}"
    extra_detail = bool(claim_content_tokens(combined) & _REMAINING_DETAIL_TOKENS)
    if not extra_detail:
        return False
    for fact in cited:
        if classify_claim(fact.text, source, skip_path=fact.json_path) != "unresolved":
            continue
        if not (claim_content_tokens(fact.text) & _REMAINING_DETAIL_TOKENS):
            continue
        if claims_are_related(fact.text, combined) or claims_are_related(combined, fact.text):
            return True
    return False


def _task_confirms_established_fact(source: NormalizedProjectSource, item: ProposedItem) -> bool:
    combined = f"{item.title} {item.description}"
    cited = _cited_source_facts(source, item)
    if _task_is_existence_or_availability_check(combined):
        if not _task_has_unresolved_remaining_purpose(source, item, cited):
            polarities = _related_statement_polarities(source, combined)
            if "established" in polarities and "unresolved" not in polarities:
                return True
    task_tokens = claim_content_tokens(combined)
    extra_detail = bool(task_tokens & _REMAINING_DETAIL_TOKENS)
    for fact in cited:
        if classify_claim(fact.text, source, skip_path=fact.json_path) != "established":
            continue
        if extra_detail and not (claim_content_tokens(fact.text) & _REMAINING_DETAIL_TOKENS):
            continue
        if _VERIFY_VERB.search(combined) and (
            _EXISTENCE_CHECK.search(combined) or claims_are_related(fact.text, item.title)
        ):
            return True
        if (
            claims_are_related(fact.text, item.title)
            and claims_are_related(item.title, fact.text)
            and not extra_detail
        ):
            return True
    return False


def _record_prerequisites(
    source: NormalizedProjectSource, proposal: BacklogProposal, findings: _FindingSink
) -> None:
    for prerequisite in proposal.prerequisites:
        statuses = [
            classify_claim(ref.excerpt or prerequisite.text, source, skip_path=ref.json_path)
            for ref in prerequisite.provenance
        ]
        if statuses and all(status == "established" for status in statuses):
            continue
        findings.add(
            Finding(
                severity=FindingSeverity.WARNING,
                code=FindingCode.UNRESOLVED_PREREQUISITE,
                message=prerequisite.text,
                source_references=[
                    _source_ref(ref.json_path, ref.field_name, excerpt=ref.excerpt)
                    for ref in prerequisite.provenance
                ],
            )
        )


def _provenance_covers_fact(
    items: list[Any],
    fact,
    *,
    types: set[WorkItemType] | None = None,
) -> bool:
    for item in items:
        if types is not None and item.type not in types:
            continue
        for reference in item.provenance:
            if reference.json_path == fact.json_path and reference.excerpt == fact.text:
                return True
    return False


def _record_uncovered_functional_requirements(
    source: NormalizedProjectSource,
    items: list[Any],
    findings: _FindingSink,
) -> None:
    for fact in source.functional_requirements:
        if classify_scope(source, fact.text) in {"out", "conflict"}:
            continue
        if _provenance_covers_fact(items, fact, types={WorkItemType.USER_STORY}):
            continue
        findings.add(
            Finding(
                severity=FindingSeverity.ERROR,
                code=FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT,
                message="Functional requirement has no traceable canonical user story provenance.",
                source_references=[
                    _source_ref(fact.json_path, fact.field_name, excerpt=fact.text)
                ],
            )
        )


def _is_exact_narrative_excerpt(source: NormalizedProjectSource, json_path: str, excerpt: str) -> bool:
    if json_path != source.request.json_path or not excerpt:
        return False
    return any(excerpt == section.excerpt for section in source.narrative_sections)


def _narrative_excerpt_cited(backlog: CanonicalBacklogV1, excerpt: str) -> bool:
    for item in backlog.items:
        for reference in item.provenance:
            if reference.json_path == "$.user_request" and reference.excerpt == excerpt:
                return True
    return False


def _record_uncovered_request_outcomes(
    source: NormalizedProjectSource,
    backlog: CanonicalBacklogV1,
    findings: _FindingSink,
    proposal: BacklogProposal | None = None,
) -> None:
    """Warn when labelled narrative sections are not represented. Heuristic only.

    Structured functional and non-functional coverage is recorded separately.
    Citing a narrative excerpt does not satisfy that contract.
    """

    structured_blob = " ".join(
        fact.text
        for fact in (
            *source.functional_requirements,
            *source.non_functional_requirements,
            *source.acceptance_criteria,
        )
    )
    story_blob = " ".join(
        " ".join(
            [
                item.title,
                item.description,
                *list(item.acceptance_criteria or ()),
                *list(item.test_requirements or ()),
            ]
        )
        for item in backlog.items
        if item.type in {WorkItemType.FEATURE, WorkItemType.USER_STORY, WorkItemType.TASK}
    )
    uncertainty_blob = ""
    if proposal is not None:
        notes = list(proposal.uncertainties)
        notes.extend(note for item in proposal.items for note in item.uncertainties)
        notes.extend(item.text for item in proposal.prerequisites)
        uncertainty_blob = " ".join(notes)
    covered_labels: set[str] = set()
    for section in source.narrative_sections:
        if section.evidence_kind == "context" or not section.label:
            continue
        covered_labels.add(section.label)
        if looks_like_timeline_or_estimate(section.title):
            continue
        cited = _narrative_excerpt_cited(backlog, section.excerpt)
        represented = cited or heading_appears_represented(section.title, structured_blob)
        if section.evidence_kind == "unconfirmed":
            represented = represented or heading_appears_represented(
                section.title, f"{story_blob} {uncertainty_blob}"
            )
            if represented:
                continue
            message = (
                f"Heuristic: narrative section {section.label}: {section.title!r} says the detail is "
                "unconfirmed. This is not a structured requirement. Do not invent the missing "
                "rule, target, or mechanism. Record an uncertainty, or a discovery task when a "
                "source unknown supports a check, and cite this exact $.user_request excerpt."
            )
        else:
            represented = represented or heading_appears_represented(section.title, story_blob)
            if represented:
                continue
            kind_word = "constraint" if section.evidence_kind == "constraint" else "outcome"
            message = (
                f"Heuristic: narrative {kind_word} {section.label}: {section.title!r} is not "
                "represented by a generated story, acceptance criterion, or exact provenance "
                "excerpt. This is labelled text inside $.user_request, not a structured "
                "functional or non-functional requirement. Token matching does not prove "
                "semantic coverage."
            )
        findings.add(
            Finding(
                severity=FindingSeverity.WARNING,
                code=FindingCode.UNCOVERED_REQUEST_OUTCOME,
                message=message,
                source_references=[
                    _source_ref("$.user_request", "user_request", excerpt=section.excerpt),
                ],
            )
        )
    for heading in parse_labeled_request_outcomes(source.request.text):
        if heading.label in covered_labels or looks_like_timeline_or_estimate(heading.title):
            continue
        if heading_appears_represented(heading.title, structured_blob):
            continue
        if heading_appears_represented(heading.title, f"{story_blob} {uncertainty_blob}"):
            continue
        findings.add(
            Finding(
                severity=FindingSeverity.WARNING,
                code=FindingCode.UNCOVERED_REQUEST_OUTCOME,
                message=(
                    f"Heuristic: labeled request heading {heading.label}: {heading.title!r} is not "
                    "clearly reflected in structured requirements or in any story title, description, "
                    "or acceptance criterion. Token matching does not prove semantic coverage. Do not "
                    "invent a json_path for this heading."
                ),
                source_references=[
                    _source_ref("$.user_request", "user_request", excerpt=None),
                ],
            )
        )


def _record_requirement_coverage_findings(
    source: NormalizedProjectSource,
    items: list[Any],
    findings: _FindingSink,
) -> None:
    for fact in source.non_functional_requirements:
        if fact.is_timeline_or_estimate or looks_like_delivery_process_constraint(fact.text):
            findings.add(
                Finding(
                    severity=FindingSeverity.INFO,
                    code=FindingCode.PLANNING_CONSTRAINT,
                    message=(
                        "Timeline, estimate, or cross-cutting delivery-process constraint is not "
                        "a committed backlog item and must not be copied onto every story."
                    ),
                    source_references=[
                        _source_ref(fact.json_path, fact.field_name, excerpt=fact.text)
                    ],
                )
            )
            continue
        if classify_scope(source, fact.text) in {"out", "conflict"}:
            continue
        if _provenance_covers_fact(items, fact):
            continue
        findings.add(
            Finding(
                severity=FindingSeverity.WARNING,
                code=FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT,
                message=(
                    "Non-functional requirement is not traced on any backlog item. "
                    "A related Feature or Story without this source path does not count as coverage."
                ),
                source_references=[
                    _source_ref(fact.json_path, fact.field_name, excerpt=fact.text)
                ],
            )
        )


def _uncertainty_key(note: str) -> str:
    return normalize_source_text(note).strip(".,;:")


def _uncertainty_repeats(candidate: str, existing: str) -> bool:
    if candidate == existing:
        return True
    short, long = (candidate, existing) if len(candidate) <= len(existing) else (existing, candidate)
    return len(short) >= 40 and short in long


def _record_uncertainties(
    proposal: BacklogProposal,
    backlog: CanonicalBacklogV1,
    findings: _FindingSink,
) -> None:
    local_ids = [item.local_id for item in proposal.items]
    by_canonical = {local_ids[index]: backlog.items[index].canonical_id for index in range(len(local_ids))}
    notes: list[tuple[str, list[str]]] = []
    for note in proposal.uncertainties:
        if note.strip():
            notes.append((note, []))
    for item in proposal.items:
        for note in item.uncertainties:
            if note.strip():
                notes.append((note, [by_canonical[item.local_id]]))
    kept: list[str] = []
    for note, canonical_ids in notes:
        key = _uncertainty_key(note)
        if not key or any(_uncertainty_repeats(key, previous) for previous in kept):
            continue
        kept.append(key)
        findings.add(
            Finding(
                severity=FindingSeverity.WARNING,
                code=FindingCode.AMBIGUOUS_DECOMPOSITION,
                message=note,
                canonical_ids=canonical_ids,
            )
        )


def _build_prompt(source: NormalizedProjectSource) -> str:
    return (
        "You decompose normalized project facts into a tracker-independent software backlog. "
        "Return JSON only, matching the supplied schema. Do not assume a particular industry.\n\n"
        "Sizing heuristics (planning guidance, not estimates; do not invent story points, hours, "
        "or delivery commitments):\n"
        "- Epic: a substantial business outcome spanning multiple stories.\n"
        "- Feature: a coherent product capability grouping related stories.\n"
        "- User Story: one meaningful, independently testable user or system outcome, normally "
        "about one sprint of work. Treat two sprints as a warning threshold, not a hard rule. "
        "If a requirement bundles independent outcomes, split into multiple stories. Each story "
        "must still cite the relevant functional requirement path and excerpt.\n"
        "- Task: concrete implementation or verification that one contributor can normally finish "
        "in a few days. Split journey-sized tasks. Do not copy the parent story title as the task.\n"
        "- Subtask: optional, only when it improves execution tracking.\n"
        "- Consider implementation, integration, error handling, security, testing, observability, "
        "and operations only when the source or solution context makes them relevant. Do not invent "
        "architecture or filler work. If the source is too thin to decompose safely, record "
        "uncertainties or prerequisites instead.\n"
        "- Do not force a Task under every Story. Atomic stories remain valid when the story is "
        "already a concrete slice. Do not artificially split work into duplicate or numbered filler.\n"
        "- Trace non-functional requirements by putting their exact source path and full excerpt "
        "on an existing item that implements or verifies that constraint, alongside a functional "
        "requirement when the item is a user_story. A story that only discusses the idea is not "
        "coverage. Do not create a user_story whose provenance contains a non-functional "
        "requirement path but no functional requirement path. Attach an NFR to a story only when "
        "that constraint applies to that story's outcome. Cross-cutting delivery-process "
        "constraints such as incremental or phased release belong on the Epic, not on every Story. "
        "Constraints such as not replacing an existing system, or an external system remaining "
        "the source of truth, belong on the FR-backed story that is bound by them.\n\n"
        "Product tree:\n"
        "- One concise Epic title. Do not paste the entire request as the title.\n"
        "- Features are in-scope product capabilities. Do not copy delivery workstreams into Features. "
        "Group stories whose outcomes the feature title and description actually describe. Cite "
        "provenance for a source capability that supports that group. A child does not have to repeat "
        "the feature's wording. Do not create extra features to reach a count. When several "
        "functional requirements describe distinct capability areas, keep those areas as separate "
        "features. Do not collapse them into one catch-all feature whose title is only an actor, "
        "management, core, platform, or capabilities label. Integration stories belong with the "
        "capability they serve, or in their own feature when the source treats that integration as "
        "a separate capability.\n"
        "- Every item in required_functional_coverage needs its own user_story whose provenance "
        "json_path and excerpt equal that entry. The initial proposal is invalid if any path is "
        "missing. A narrative section does not cover a functional requirement. Never place two "
        "distinct functional-requirement paths on one user_story. A story may also cite the "
        "acceptance criterion that verifies its outcome and an NFR that actually constrains that "
        "outcome. Do not attach a global delivery NFR to every story merely because it exists.\n"
        "- Story descriptions must explain actor, behaviour, and outcome. Do not restate the title.\n"
        "- Bind each source acceptance criterion to the single story whose outcome that criterion "
        "actually verifies, and mark acceptance_criteria_origins as source. Do not attach a source "
        "criterion to an authentication, integration, notification, or discovery story merely to "
        "obtain a provenance path when the criterion describes a different user outcome. Do not "
        "reuse an authentication or authorization criterion as a generic criterion on unrelated "
        "stories. Additional story-specific criteria may be proposed with origin generated. If a "
        "source acceptance criterion concatenates independently owned outcomes, such as listing "
        "records and reviewing terms, do not copy that compound sentence into any story's "
        "acceptance_criteria. The listing story gets a generated criterion for list visibility "
        "only. The terms story gets a generated criterion for reviewing terms only. You may cite "
        "the compound criterion path on at most one story, and its provenance excerpt must equal "
        "the full source field. The acceptance criterion text must still be the scoped generated "
        "sentence, with origin generated, never the compound sentence marked as source. Do not "
        "put that source path on both stories. Create generated acceptance "
        "criteria scoped to each story's owned outcome. Keep source excerpts and json_path accurate "
        "to the original compound criterion or the matching functional requirement. Never mark "
        "rewritten or shortened criterion text as origin=source. Never represent generated "
        "acceptance criteria as verbatim source text. A customer submission story and the story "
        "that integrates with a downstream system stay distinct: the submission story may say the "
        "request is sent to that system, and that mention is a dependency, not shared ownership "
        "of the integration story's acceptance criteria. Each functional requirement must remain "
        "covered on its own story. Do not invent additional systems, endpoints, databases, "
        "workflows, authentication protocols, or acceptance criteria the source does not state. "
        "If binding is uncertain, say so in uncertainties.\n"
        "- Acceptance criteria are observable outcomes. test_requirements are verification activities "
        "(tests to run, data to assert, integrations to stub). They must not copy acceptance criteria. "
        "Mark test_requirements generated.\n"
        "- Add Tasks only when they are concrete, independently trackable implementation, integration, "
        "testing, operational, or investigation work. Do not force a Task under every Story and do "
        "not require any minimum number of Tasks, Features, or Stories. If the Story is already an "
        "atomic implementable slice and no checkable unknown applies, leave it without child tasks. "
        "When a story names an external integration, mapping, or verification that the source "
        "supports as distinct work, add bounded implementation or verification tasks with "
        "observable results. Do not invent endpoints, payloads, or provider behavior. "
        "Add an investigation task when a source unknown, open question, or unresolved blocking "
        "dependency names a verifiable objective that is not already established elsewhere in the "
        "source, for example confirming whether API documentation is available and complete when "
        "the source does not already state that, verifying whether a sandbox or test environment "
        "is available, or determining whether a documented status lookup or recovery from an "
        "uncertain outcome can be exercised where documentation or an environment exists. Each "
        "investigation task needs a bounded objective, a verifiable result such as available, "
        "incomplete, unavailable, or not yet verifiable, and provenance equal to that unknown or "
        "dependency path and excerpt. State that the capability is not confirmed. Do not add a "
        "task whose purpose is to confirm that a fact exists, is available, or that named people "
        "can participate when evidence_status is established. Confirmation of access or "
        "documentation does not establish every capability, sandbox, authentication behavior, "
        "schema, or integration detail; keep those remaining unknowns visible. Do not treat an "
        "assumption as a verified capability. Do not invent endpoints, idempotency support, "
        "schemas, databases, frameworks, authentication protocols, or integration guarantees. Do "
        "not add generic filler such as implement backend, do testing, or handle errors. If a "
        "question cannot yet be checked, record an uncertainty or unresolved prerequisite instead. "
        "Zero tasks is valid.\n"
        "- Treat listed deliverables as inspiration for necessary work, not a 1:1 task dump.\n"
        "- Subtasks only when they add tracking value as independently completable slices. Do not copy "
        "every milestone or phase as a Subtask.\n"
        "- depends_on_local_ids only for a defensible blocking relationship (for example a confirmation "
        "step that cannot run until a submit step succeeds). Do not add dependencies to fill the tree.\n"
        "- Put unverified blocking conditions (external access, unfinalized staffing, unconfirmed "
        "standards) in prerequisites with status unresolved. Do not list a condition the source "
        "already establishes unless a related statement contradicts it; keep genuine conflicts "
        "visible. Do not create a task that claims unresolved conditions are already met. "
        "blocking_dependencies.evidence_status is established, unresolved, or conflict; do not "
        "treat every listed dependency as an unknown.\n"
        "- Honour in-scope and out-of-scope statements. Never create items for excluded work.\n"
        "- Do not invent effort, architecture decisions, or external approvals. Do not convert a "
        "timeline or estimate constraint into a committed milestone, date, or work item.\n"
        "- Trace a non-functional requirement only by putting its exact source path and excerpt on an "
        "item that implements or verifies that constraint. A related Feature or Story without that "
        "path does not count as coverage. Attach source-of-truth and do-not-replace-existing-system "
        "constraints to the FR-backed integration or portal story they constrain. Do not promote an "
        "NFR into its own user_story. Timeline and estimate statements are planning constraints, "
        "not committed work items.\n"
        "- title_origin=source only when the title string equals a provenance excerpt. A grounded "
        "paraphrase, including a shorter or reordered wording of the same fact, is "
        "title_origin=generated and must still cite the real excerpt. Do not mark a paraphrase as "
        "source. Array provenance excerpts must equal the source element. A $.user_request excerpt "
        "must be the full field or an exact narrative_sections or narrative_context excerpt. Do not "
        "cite a summary. Record each remaining uncertainty once. Do not copy the same unknown onto "
        "the proposal and again onto an item.\n"
        "- local_id values are your own stable keys (not tracker keys, not cbl_ ids).\n"
        "- Hierarchy is epic→feature→user_story→task→subtask.\n"
        "- Provenance json_path must be the Copilot envelope path on each fact (the fact's json_path "
        "value). Do not cite prompt-object keys or adapter attribute names such as $.contract_id, "
        "$.capabilities, $.functional_requirements, $.acceptance_criteria, "
        "$.non_functional_requirements, or $.blocking_dependencies. Array facts require the indexed "
        "envelope path (for example $.requirements.functional_requirements[0]). Never invent paths "
        "such as $.user_request.FR-01; labeled request headings are context, not structured facts.\n"
        "- Feature provenance should describe the same capability as its child stories. Do not attach "
        "a feature to a capability excerpt that none of its children implement.\n"
        "- narrative_sections are exact labelled spans inside $.user_request. They are not "
        "structured requirements and their json_path is only $.user_request. evidence_kind "
        "outcome: when the section is independently testable and no structured requirement "
        "already states it, add a generated user story, or a generated acceptance criterion on "
        "the story that owns that responsibility. evidence_kind constraint: attach it to the "
        "functional story it constrains; do not add a story that has only a constraint. "
        "evidence_kind unconfirmed: do not invent the missing rule, target, or mechanism; record "
        "an uncertainty, or one discovery task when a source unknown supports a check. Cite the "
        "exact excerpt. Titles and criteria for this evidence are generated. Independently "
        "testable examples include eligibility, duplicate submission, uncertain results, status "
        "visibility, operational visibility, and auditability. Combine outcomes that are one "
        "testable responsibility. Do not add one item per paragraph or per heading. "
        "narrative_context is grounding only. labeled_request_outcomes remain headings, not paths.\n\n"
        "Normalized project facts:\n"
        f"{json.dumps(source.prompt_facts, ensure_ascii=False, indent=2)}\n"
    )


def _actionable_quality_findings(findings: _FindingSink) -> list[Finding]:
    return [item for item in findings.items if item.code in _QUALITY_REPAIR_CODES]


def _proposal_item_counts(proposal: BacklogProposal | None) -> dict[str, int]:
    if proposal is None:
        return {}
    counts = Counter(item.type.value for item in proposal.items)
    return {name: counts[name] for name in sorted(counts)}


def _covered_fr_paths(proposal: BacklogProposal | None) -> list[str]:
    if proposal is None:
        return []
    paths: set[str] = set()
    for item in proposal.items:
        if item.type is not WorkItemType.USER_STORY:
            continue
        for reference in item.provenance:
            if _is_requirement_path(reference.json_path, _FUNCTIONAL_REQUIREMENT_PATH):
                paths.add(reference.json_path)
    return sorted(paths)


def _fr_coverage_snapshot(proposal: BacklogProposal | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    if proposal is None:
        return rows
    for item in proposal.items:
        if item.type is not WorkItemType.USER_STORY:
            continue
        for reference in item.provenance:
            if not _is_requirement_path(reference.json_path, _FUNCTIONAL_REQUIREMENT_PATH):
                continue
            key = (item.local_id, reference.json_path)
            if key in seen:
                continue
            seen.add(key)
            excerpt = (reference.excerpt or "")[:240]
            rows.append(
                {
                    "local_id": item.local_id,
                    "json_path": reference.json_path,
                    "excerpt": excerpt,
                }
            )
    return rows


def _record_lost_fr_paths(
    calls: list[LlmCallDiagnostic],
    initial: BacklogProposal | None,
    repaired: BacklogProposal | None,
) -> None:
    lost = [path for path in _covered_fr_paths(initial) if path not in set(_covered_fr_paths(repaired))]
    if not lost:
        return
    bounded = lost[:20]
    for index, event in enumerate(calls):
        if event.kind in {"quality_repair", "hard_repair"} and not event.accepted:
            calls[index] = event.model_copy(update={"lost_fr_paths": bounded})


def _diagnostic(
    kind: str,
    result: _MaterializeResult | None,
    *,
    usable_proposal: bool | None = None,
) -> LlmCallDiagnostic:
    parsed = result.parsed if result is not None else False
    ok = result.ok if result is not None else False
    errors = list(result.errors) if result is not None and not ok else []
    proposal = result.proposal if result is not None else None
    event = LlmCallDiagnostic(
        kind=kind,  # type: ignore[arg-type]
        usable_proposal=parsed if usable_proposal is None else usable_proposal,
        validation_passed=ok,
        accepted=False,
        validation_errors=_public_validation_errors(errors),
        item_counts=_proposal_item_counts(proposal),
        covered_fr_paths=_covered_fr_paths(proposal),
    )
    _LOG.info(
        "llm_call kind=%s usable_proposal=%s validation_passed=%s accepted=%s "
        "error_count=%s item_counts=%s covered_fr=%s",
        event.kind,
        event.usable_proposal,
        event.validation_passed,
        event.accepted,
        len(event.validation_errors),
        event.item_counts,
        event.covered_fr_paths,
    )
    return event


def _public_validation_errors(errors: list[str]) -> list[str]:
    """Keep short deterministic messages. Never include prompts, payloads, or secrets."""

    cleaned: list[str] = []
    for error in errors[:20]:
        text = " ".join(error.split())
        text = _SENSITIVE_ERROR.sub("[redacted]", text)
        lowered = text.lower()
        if "normalized project facts" in lowered or "generate_json" in lowered:
            text = "validation error omitted because it contained prompt or payload text"
        if len(text) > 300:
            text = text[:297] + "..."
        cleaned.append(text)
    return cleaned


def _mark_accepted(calls: list[LlmCallDiagnostic], kind: str) -> None:
    for index, event in enumerate(calls):
        if event.kind == kind:
            updated = event.model_copy(update={"accepted": True})
            calls[index] = updated
            _LOG.info(
                "llm_call kind=%s usable_proposal=%s validation_passed=%s accepted=%s",
                updated.kind,
                updated.usable_proposal,
                updated.validation_passed,
                updated.accepted,
            )
            return


def _error_owner(error: str) -> str:
    return error.split(":", 1)[0].split(".", 1)[0].strip()


def _relocate_cross_cutting_constraints(
    source: NormalizedProjectSource, proposal: BacklogProposal
) -> None:
    """Move delivery-process NFRs from stories onto the Epic. Does not drop FR coverage."""

    process_paths = {
        fact.json_path
        for fact in source.non_functional_requirements
        if looks_like_delivery_process_constraint(fact.text)
    }
    if not process_paths:
        return
    epics = [item for item in proposal.items if item.type is WorkItemType.EPIC]
    moved: list[Any] = []
    for item in proposal.items:
        if item.type is WorkItemType.EPIC:
            continue
        keep = [ref for ref in item.provenance if ref.json_path not in process_paths]
        if keep and len(keep) < len(item.provenance):
            moved.extend(ref for ref in item.provenance if ref.json_path in process_paths)
            item.provenance = keep
    if not epics or not moved:
        return
    seen = {ref.json_path for ref in epics[0].provenance}
    for reference in moved:
        if reference.json_path in seen:
            continue
        epics[0].provenance.append(reference)
        seen.add(reference.json_path)


def _recover_quality_repair(
    source: NormalizedProjectSource, second: _MaterializeResult
) -> _MaterializeResult | None:
    """Keep a repaired tree when only new Tasks failed validation. Never merge two trees."""

    if second.ok or second.proposal is None:
        return None
    task_ids = {
        item.local_id
        for item in second.proposal.items
        if item.type in {WorkItemType.TASK, WorkItemType.SUBTASK}
    }
    bad_tasks: set[str] = set()
    for error in second.errors:
        owner = _error_owner(error)
        if owner in task_ids:
            bad_tasks.add(owner)
        else:
            return None
    if not bad_tasks:
        return None
    drop = set(bad_tasks)
    raw = second.proposal.model_dump(mode="json")
    changed = True
    while changed:
        changed = False
        for item in raw.get("items") or []:
            local_id = str(item.get("local_id") or "")
            parent = item.get("parent_local_id")
            if local_id in drop or parent not in drop:
                continue
            drop.add(local_id)
            changed = True
    raw["items"] = [
        item
        for item in raw.get("items") or []
        if str(item.get("local_id") or "") not in drop
    ]
    for item in raw["items"]:
        item["depends_on_local_ids"] = [
            dep for dep in item.get("depends_on_local_ids") or [] if dep not in drop
        ]
    recovered = _try_materialize(source, raw)
    if not recovered.ok:
        return None
    initial_fr = set(_covered_fr_paths(second.proposal))
    if not initial_fr <= set(_covered_fr_paths(recovered.proposal)):
        return None
    dropped: list[str] = []
    for local_id in sorted(bad_tasks):
        title = next(
            (item.title for item in second.proposal.items if item.local_id == local_id),
            local_id,
        )
        dropped.append(title)
    recovered.findings.add(
        Finding(
            severity=FindingSeverity.INFO,
            code=FindingCode.QUALITY_REPAIR_REJECTED,
            message=(
                "Quality repair included invalid tasks that were dropped without discarding the "
                f"remaining repaired tree: {', '.join(repr(title) for title in dropped[:8])}. "
                "The initial and repaired trees were not merged."
            ),
        )
    )
    return recovered


def _record_quality_repair_rejected(
    initial: _MaterializeResult, second: _MaterializeResult
) -> None:
    reasons = _public_validation_errors(second.errors)[:5]
    summary = "; ".join(reasons) if reasons else "validation failed"
    initial.findings.add(
        Finding(
            severity=FindingSeverity.WARNING,
            code=FindingCode.QUALITY_REPAIR_REJECTED,
            message=(
                "Quality repair was rejected; the initial valid proposal was kept. "
                f"Repair failed because: {summary}. Remaining quality findings were not silently "
                "cleared; they still describe the accepted tree."
            ),
        )
    )


def _accepted(result: _MaterializeResult, calls: list[LlmCallDiagnostic]) -> tuple[CanonicalBacklogV1, GenerationFindings]:
    assert result.backlog is not None
    return result.backlog, GenerationFindings(items=result.findings.items, llm_calls=calls)


def _invoke_provider(
    provider: AIProvider,
    prompt: str,
    schema: Mapping[str, Any],
    source: NormalizedProjectSource,
    calls: list[LlmCallDiagnostic],
    kind: str,
) -> tuple[Any, _MaterializeResult]:
    try:
        raw = provider.generate_json(prompt, schema)
    except Exception:
        calls.append(_diagnostic(kind, None, usable_proposal=False))
        raise
    result = _try_materialize(source, raw)
    calls.append(_diagnostic(kind, result))
    return raw, result


def _uncovered_narrative_outcomes(
    source: NormalizedProjectSource,
    findings: _FindingSink,
) -> list[dict[str, str]]:
    """Outcome sections the proposal left unrepresented.

    Constraints and unconfirmed sections stay warnings. They do not start a
    repair and are not returned here. The excerpt is the verbatim request span.
    """

    by_excerpt = {
        section.excerpt: section
        for section in source.narrative_sections
        if section.evidence_kind == "outcome"
    }
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in findings.items:
        if item.code is not FindingCode.UNCOVERED_REQUEST_OUTCOME:
            continue
        for reference in item.source_references:
            section = by_excerpt.get(reference.excerpt or "")
            if section is None or section.label in seen:
                continue
            seen.add(section.label)
            rows.append(
                {
                    "label": section.label,
                    "title": section.title,
                    "json_path": "$.user_request",
                    "excerpt": section.excerpt,
                }
            )
    return rows


def _quality_finding_payload(finding: Finding) -> dict[str, Any]:
    return {
        "code": finding.code.value,
        "message": finding.message,
        "canonical_ids": list(finding.canonical_ids),
        "local_ids": list(finding.local_ids),
        "source_paths": [reference.json_path for reference in finding.source_references],
    }


def _build_quality_repair_prompt(
    original_prompt: str,
    previous: Mapping[str, Any],
    findings: list[Finding],
    coverage_snapshot: list[dict[str, str]] | None = None,
    narrative_outcomes: list[dict[str, str]] | None = None,
) -> str:
    listed = json.dumps(
        [_quality_finding_payload(item) for item in findings],
        ensure_ascii=False,
        indent=2,
    )
    snapshot = json.dumps(coverage_snapshot or [], ensure_ascii=False, indent=2)
    narrative = json.dumps(narrative_outcomes or [], ensure_ascii=False, indent=2)
    return (
        f"{original_prompt}\n\n"
        "The previous JSON passed deterministic validation but has actionable quality "
        "findings. Return a corrected JSON object for the same project facts. Do not "
        "mark the output as valid yourself. Resolve the named pairwise overlaps in the "
        "findings; do not broadly rewrite unrelated items.\n"
        "Repair guidance:\n"
        "- Canonical IDs identify persisted review items only. Proposal items use local_id. "
        "Match each overlap finding to previous JSON items by local_ids in the finding, then "
        "by title if a local_id is absent. Never treat a canonical UUID or cbl_ id as a "
        "proposal local_id.\n"
        "- Use each overlap finding's message (titles, canonical IDs, proposal local_ids, "
        "overlapping action, evidence, and remediation) to locate the named pair. Fix that "
        "overlap pair only.\n"
        "- Uncovered narrative outcomes are listed separately. For each one, add a generated "
        "user story or a generated acceptance criterion on the story that owns that behavior. "
        "Copy its excerpt exactly onto provenance whose json_path is $.user_request. Do not "
        "mark the title or criterion as source. Do not treat the excerpt as structured "
        "functional-requirement coverage. Skip the entry when the same behavior is already "
        "represented. Do not add stories for unconfirmed sections, and do not invent API "
        "endpoints, protocols, schemas, numeric targets, or audit fields.\n"
        "- When one story owns an action named in a finding, remove that leaked action from "
        "the non-owning story's description and acceptance criteria. Keep both stories when "
        "each is independently required for functional-requirement coverage.\n"
        "- If a source acceptance criterion concatenates independently owned outcomes, replace "
        "the compound copy with generated acceptance criteria scoped to each story. Keep "
        "source excerpts and json_path accurate. Never mark edited criterion text as "
        "origin=source.\n"
        "- Split a Story only when source evidence supports independently testable outcomes.\n"
        "- Remove duplicated responsibilities and acceptance criteria across Stories. Bind each "
        "source acceptance criterion to the story whose outcome it verifies; do not copy a "
        "viewing or submission criterion onto an authentication or integration story merely "
        "to obtain provenance.\n"
        "- Add concrete child Tasks only when the source supports distinct implementation, "
        "verification, or discovery work. Do not invent endpoints or filler tasks.\n"
        "- Narrative sections are exact $.user_request excerpts, not structured requirements. An "
        "independently testable outcome may be a generated criterion or its own generated story "
        "when no structured requirement states it. Cite the exact excerpt. That citation does not "
        "cover required_functional_coverage.\n"
        "- Narrative headings are context, not structured requirements. For eligibility, duplicate "
        "submission, uncertain or unresolved outcomes, status visibility, customer-service or "
        "operational visibility, and auditability: when the request states a testable behavior, add "
        "a generated acceptance criterion on the existing functional story that owns it. When the "
        "request says the rules, states, permissions, fields, or mechanism are unconfirmed, add one "
        "bounded discovery task under that story, or an uncertainty, citing the unknown or "
        "dependency. The task result is whether the fact is available, incomplete, unavailable, or "
        "not yet verifiable. Do not invent API endpoint paths, payload schemas, idempotency "
        "support, audit fields, retention periods, source paths, or quotations. Do not add stories "
        "or tasks merely to increase the count.\n"
        "- A feature groups stories that implement the capability in its title and description. "
        "Its provenance must support that capability. A child may use different wording. When "
        "child stories implement distinct source capabilities, split them into separate features "
        "instead of one catch-all. Do not add features to reach a count. title_origin=source only "
        "when the title equals a provenance excerpt; a paraphrase is generated.\n"
        "- If a finding names missing implementation detail on an integration or verification "
        "story, add concrete child tasks the source supports. Do not add a task that only "
        "investigates whether an established source fact is true.\n"
        "- Attach delivery-process NFRs to the Epic. Do not copy them onto every Story.\n"
        "- Treat non-functional requirements as constraints on relevant work, not "
        "automatically as separate technical stories. Never add a user_story whose only "
        "requirement provenance is a non-functional requirement; attach that exact path and "
        "excerpt to related functional work instead. Never merge distinct functional "
        "requirements onto one user_story.\n"
        "- Preserve previously covered functional requirements. The coverage snapshot lists "
        "local_id, json_path, and excerpt pairs that must remain covered by some user_story "
        "after repair. You may move a path to a different story when splitting overlap, but "
        "you must not drop a previously covered path. Independent validation will reject a "
        "repair that loses coverage. Do not merge the initial and repaired trees.\n"
        "- Preserve source provenance, requirement coverage, valid dependencies, scope, "
        "and local_id values for unchanged items.\n"
        "- If implementation detail is unavailable, record a specific uncertainty or "
        "prerequisite rather than inventing architecture, local databases, endpoints, "
        "API contracts, payload formats, or business rules.\n"
        f"Previously covered functional requirements:\n{snapshot}\n"
        f"Quality findings:\n{listed}\n"
        f"Uncovered narrative outcomes:\n{narrative}\n\n"
        "Previous JSON:\n"
        f"{json.dumps(previous, ensure_ascii=False)}\n"
    )


def _repair_task_snapshot(previous: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Bounded task list for the repair prompt. Does not copy tasks into the result."""

    items = previous.get("items")
    if not isinstance(items, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or item.get("type") not in {"task", "subtask"}:
            continue
        provenance: list[dict[str, str]] = []
        for reference in item.get("provenance") or []:
            if not isinstance(reference, dict):
                continue
            provenance.append(
                {
                    "json_path": str(reference.get("json_path") or "")[:180],
                    "excerpt": str(reference.get("excerpt") or "")[:240],
                }
            )
        rows.append(
            {
                "local_id": str(item.get("local_id") or "")[:80],
                "title": str(item.get("title") or "")[:180],
                "parent_local_id": item.get("parent_local_id"),
                "provenance": provenance[:4],
            }
        )
        if len(rows) >= 12:
            break
    return rows


_GROUNDED_TASK_PATH_MARKERS = ("unknown", "open_question", "dependencies", "discovery")


def _note_dropped_grounded_tasks(
    source: NormalizedProjectSource, previous: Mapping[str, Any], result: _MaterializeResult
) -> None:
    """Warn when the only valid repair omits a source-backed investigation task.

    The initial tree is not spliced back in. The finding names the omitted task so
    the gap stays visible after a hard repair that otherwise validates.
    """

    if result.backlog is None:
        return
    kept_paths = {
        reference.json_path
        for item in result.backlog.items
        if item.type is WorkItemType.TASK
        for reference in item.provenance
    }
    for item in previous.get("items") or []:
        if not isinstance(item, dict) or item.get("type") != "task":
            continue
        grounded = [
            reference
            for reference in item.get("provenance") or []
            if isinstance(reference, dict)
            and any(
                marker in str(reference.get("json_path") or "")
                for marker in _GROUNDED_TASK_PATH_MARKERS
            )
        ]
        if not grounded or any(str(reference.get("json_path") or "") in kept_paths for reference in grounded):
            continue
        if all(
            classify_claim(
                str(reference.get("excerpt") or ""),
                source,
                skip_path=str(reference.get("json_path") or "") or None,
            )
            == "established"
            for reference in grounded
        ):
            continue
        title = str(item.get("title") or "investigation task")
        result.findings.add(
            Finding(
                severity=FindingSeverity.WARNING,
                code=FindingCode.MISSING_IMPLEMENTATION_DETAIL,
                message=(
                    f"Hard repair omitted grounded investigation task {title!r}. "
                    "The accepted tree was not merged with the initial proposal. "
                    "Restore an equivalent task only when the same source path and excerpt still apply."
                ),
                source_references=[
                    _source_ref(
                        str(reference.get("json_path") or ""),
                        str(reference.get("field_name") or "") or None,
                        excerpt=str(reference.get("excerpt") or "") or None,
                    )
                    for reference in grounded[:2]
                ],
            )
        )


def _build_repair_prompt(
    original_prompt: str,
    previous: Mapping[str, Any],
    errors: list[str],
) -> str:
    listed = "\n".join(f"- {error}" for error in errors)
    tasks = json.dumps(_repair_task_snapshot(previous), ensure_ascii=False, indent=2)
    return (
        f"{original_prompt}\n\n"
        "The previous JSON failed deterministic validation. Return a corrected JSON object "
        "for the same project facts. Do not mark the output as valid yourself. This is the only "
        "repair. Do not drop a functional requirement that the previous JSON already covered. "
        "A narrative excerpt on $.user_request does not cover a required functional requirement. "
        "Do not merge the previous tree with a new one; return one corrected tree.\n"
        "While correcting the validation errors, also apply these rules when the source supports them:\n"
        "- If a story's acceptance criterion is a compound source sentence that also states another "
        "story's outcome, replace that criterion with a generated sentence scoped to this story. "
        "List visibility and renewal-term review are separate outcomes. Keep any cited provenance "
        "excerpt equal to the full source field, and never mark the scoped sentence as source.\n"
        "- Attach each in-scope non-functional requirement's exact path and excerpt to an existing "
        "FR-backed item it constrains. Do not add a story that has only an NFR path.\n"
        "- Previous tasks are listed below. Preserve each task whose provenance cites an unknown, "
        "open question, unresolved or conflicting dependency, or discovery fact, or replace it "
        "with an equivalent task that uses the same json_path and excerpt, a bounded objective, "
        "and an unconfirmed result. Do not preserve a task whose only purpose is to confirm a fact "
        "the source already establishes. Do not delete an unresolved investigation task while "
        "fixing titles, story splits, or other errors. A generic filler task may be removed. Do "
        "not invent endpoints or capabilities to replace it.\n"
        "- When a labelled section is narrative evidence, cite its exact $.user_request excerpt. "
        "Represent a testable outcome as a generated criterion or, when no structured requirement "
        "states it, a generated story. Record an uncertainty or investigation task when the section "
        "is unconfirmed. Do not invent a json_path or treat the excerpt as functional-requirement coverage.\n"
        "- title_origin=source only when the title string equals a provenance excerpt. A grounded "
        "paraphrase is title_origin=generated and keeps the real excerpt. Correct that validation "
        "error by setting generated. Do not rewrite the title into the full source sentence, and "
        "do not drop features or investigation tasks to make the origin check pass.\n"
        "- Keep feature groupings that match a source capability. Do not collapse distinct "
        "capability areas into one catch-all feature. Different wording between a feature and a "
        "child is fine when they implement the same capability.\n"
        "- Preserve each grounded investigation task listed below when its unknown, open question, "
        "or dependency is still unresolved or conflicting. Do not add investigation tasks the "
        "source does not support, do not confirm established facts, and do not require a task "
        "under every story.\n"
        "- Feature provenance should describe the capability its child stories implement.\n"
        f"Validation errors:\n{listed}\n\n"
        f"Previous tasks to preserve or replace when they are evidence-supported:\n{tasks}\n\n"
        "Previous JSON:\n"
        f"{json.dumps(previous, ensure_ascii=False)}\n"
    )
