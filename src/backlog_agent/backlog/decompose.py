"""LLM-backed backlog decomposition with deterministic Python validation."""

from __future__ import annotations

import json
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
from backlog_agent.backlog.granularity import record_granularity_findings, task_restates_parent
from backlog_agent.backlog.proposal import (
    BacklogProposal,
    ContentOrigin,
    ProposedItem,
    proposal_json_schema,
)
from backlog_agent.findings.models import Finding, FindingCode, FindingSeverity, GenerationFindings
from backlog_agent.llm.exceptions import DecompositionValidationError
from backlog_agent.llm.provider import AIProvider
from backlog_agent.source import load_project_source
from backlog_agent.source.model import (
    NormalizedProjectSource,
    classify_scope,
    fact_texts,
)

_PARENT_TYPE: dict[WorkItemType, WorkItemType | None] = {
    WorkItemType.EPIC: None,
    WorkItemType.FEATURE: WorkItemType.EPIC,
    WorkItemType.USER_STORY: WorkItemType.FEATURE,
    WorkItemType.TASK: WorkItemType.USER_STORY,
    WorkItemType.SUBTASK: WorkItemType.TASK,
}

_QUALITY_REPAIR_CODES = frozenset(
    {
        FindingCode.OVERSIZED_STORY,
        FindingCode.MISSING_IMPLEMENTATION_DETAIL,
        FindingCode.OVERLAPPING_STORY_RESPONSIBILITY,
        FindingCode.DUPLICATE_ACCEPTANCE_CRITERION,
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
    }
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
    raw = provider.generate_json(prompt, schema)
    result = _try_materialize(source, raw)

    if result.ok:
        quality = _actionable_quality_findings(result.findings)
        if repair and quality:
            quality_prompt = _build_quality_repair_prompt(prompt, raw, quality)
            repaired = provider.generate_json(quality_prompt, schema)
            second = _try_materialize(source, repaired)
            if second.ok:
                return second.backlog, GenerationFindings(items=second.findings.items)
            return result.backlog, GenerationFindings(items=result.findings.items)
        return result.backlog, GenerationFindings(items=result.findings.items)

    if repair:
        repair_prompt = _build_repair_prompt(prompt, raw, result.errors)
        repaired = provider.generate_json(repair_prompt, schema)
        second = _try_materialize(source, repaired)
        if second.ok:
            return second.backlog, GenerationFindings(items=second.findings.items)
        result = second

    findings = result.findings
    findings.items.append(
        Finding(
            severity=FindingSeverity.ERROR,
            code=FindingCode.LLM_VALIDATION_FAILED,
            message="LLM backlog proposal failed independent validation after at most one repair.",
        )
    )
    raise DecompositionValidationError(result.errors, GenerationFindings(items=findings.items))


class _MaterializeResult:
    def __init__(
        self,
        *,
        ok: bool,
        errors: list[str],
        findings: _FindingSink,
        backlog: CanonicalBacklogV1 | None = None,
    ) -> None:
        self.ok = ok
        self.errors = errors
        self.findings = findings
        self.backlog = backlog


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
        return _MaterializeResult(ok=False, errors=errors or ["proposal is not a valid object"], findings=findings)

    errors.extend(_structural_errors(proposal))
    errors.extend(_provenance_and_scope_errors(source, proposal, findings))
    errors.extend(_execution_quality_errors(source, proposal))
    if errors:
        return _MaterializeResult(ok=False, errors=errors, findings=findings)

    try:
        backlog = _to_canonical(source, proposal, findings)
    except (ValidationError, ValueError) as exc:
        errors.append(str(exc))
        return _MaterializeResult(ok=False, errors=errors, findings=findings)

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
        return _MaterializeResult(ok=False, errors=errors, findings=findings)

    _record_generated_content(source, proposal, backlog, findings)
    _record_uncertainties(proposal, backlog, findings)
    _record_prerequisites(proposal, findings)
    _record_requirement_coverage_findings(source, backlog.items, findings)
    record_granularity_findings(source, backlog, findings, proposal=proposal)
    return _MaterializeResult(ok=True, errors=[], findings=findings, backlog=backlog)


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
                # Concise epic titles drawn from a longer source field are generated,
                # not source copies. Correct the origin; keep provenance grounding.
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
            # Unique scalar fields are identified by path. Array elements still
            # require an exact excerpt so the wrong index cannot be papered over.
            if "[" not in ref.json_path:
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
            errors.append(
                f"{item.local_id}.acceptance_criteria[{index}]: origin=source but text is not a Copilot acceptance criterion"
            )
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
                errors.append(
                    f"{item.local_id}: source blocking dependency must be an unresolved prerequisite, not an implementation task"
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


def _title_is_grounded_paraphrase(title: str, excerpts: list[str]) -> bool:
    normalized_title = normalize_source_text(title)
    if len(normalized_title) < 12:
        return False
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


def _record_prerequisites(proposal: BacklogProposal, findings: _FindingSink) -> None:
    for prerequisite in proposal.prerequisites:
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


def _record_requirement_coverage_findings(
    source: NormalizedProjectSource,
    items: list[Any],
    findings: _FindingSink,
) -> None:
    for fact in source.non_functional_requirements:
        if fact.is_timeline_or_estimate:
            findings.add(
                Finding(
                    severity=FindingSeverity.INFO,
                    code=FindingCode.PLANNING_CONSTRAINT,
                    message="Timeline or estimate constraint is not a committed backlog item.",
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


def _record_uncertainties(
    proposal: BacklogProposal,
    backlog: CanonicalBacklogV1,
    findings: _FindingSink,
) -> None:
    local_ids = [item.local_id for item in proposal.items]
    by_canonical = {local_ids[index]: backlog.items[index].canonical_id for index in range(len(local_ids))}
    for note in proposal.uncertainties:
        if not note.strip():
            continue
        findings.add(
            Finding(
                severity=FindingSeverity.WARNING,
                code=FindingCode.AMBIGUOUS_DECOMPOSITION,
                message=note,
            )
        )
    for item in proposal.items:
        for note in item.uncertainties:
            if not note.strip():
                continue
            findings.add(
                Finding(
                    severity=FindingSeverity.WARNING,
                    code=FindingCode.AMBIGUOUS_DECOMPOSITION,
                    message=note,
                    canonical_ids=[by_canonical[item.local_id]],
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
        "- Trace non-functional requirements by putting their exact source path on the item that "
        "implements or verifies them. Do not create a user_story whose provenance contains a "
        "non-functional requirement path but no functional requirement path. NFRs are constraints "
        "on related functional work, not standalone stories.\n\n"
        "Product tree:\n"
        "- One concise Epic title. Do not paste the entire request as the title.\n"
        "- Features are in-scope product capabilities. Do not copy delivery workstreams into Features.\n"
        "- Every in-scope functional requirement needs at least one user_story whose provenance "
        "json_path points at that requirement and whose excerpt equals the source text.\n"
        "- Story descriptions must explain actor, behaviour, and outcome. Do not restate the title.\n"
        "- Bind source acceptance criteria onto the story they most clearly verify and mark "
        "acceptance_criteria_origins as source. Additional story-specific criteria may be proposed "
        "with origin generated. If a source acceptance criterion concatenates independently owned "
        "outcomes, do not copy that entire criterion onto every involved story. Create generated "
        "acceptance criteria scoped to each story's owned outcome. Keep source_excerpt and json_path "
        "accurate to the original compound criterion or the matching functional requirement. Never "
        "mark rewritten or shortened criterion text as origin=source. Both functional requirements "
        "must remain covered. Do not invent additional systems, endpoints, databases, workflows, "
        "or acceptance criteria the source does not state. If binding is uncertain, say so in "
        "uncertainties.\n"
        "- Acceptance criteria are observable outcomes. test_requirements are verification activities "
        "(tests to run, data to assert, integrations to stub). They must not copy acceptance criteria. "
        "Mark test_requirements generated.\n"
        "- Add Tasks only when they are concrete, independently trackable implementation, integration, "
        "testing, or operational work. Do not force a Task under every Story. If the Story is already "
        "an atomic implementable slice, leave it without child tasks. If implementation work is needed "
        "but unknown, record an uncertainty instead of inventing filler.\n"
        "- Treat listed deliverables as inspiration for necessary work, not a 1:1 task dump.\n"
        "- Subtasks only when they add tracking value as independently completable slices. Do not copy "
        "every milestone or phase as a Subtask.\n"
        "- depends_on_local_ids only for a defensible blocking relationship (for example a confirmation "
        "step that cannot run until a submit step succeeds). Do not add dependencies to fill the tree.\n"
        "- Put unverified blocking conditions (external access, unfinalized staffing, unconfirmed "
        "standards) in prerequisites with status unresolved. Do not create a task that claims those "
        "conditions are already met.\n"
        "- Honour in-scope and out-of-scope statements. Never create items for excluded work.\n"
        "- Do not invent effort, architecture decisions, or external approvals. Do not convert a "
        "timeline or estimate constraint into a committed milestone, date, or work item.\n"
        "- Trace a non-functional requirement only by putting its exact source path and excerpt on an "
        "item that implements or verifies that constraint. A related Feature or Story without that "
        "path does not count as coverage. Do not promote an NFR into its own user_story. Timeline "
        "and estimate statements are planning constraints, not committed work items.\n"
        "- title_origin=source only when title equals a provenance excerpt. A concise epic title "
        "taken from a longer request or heading is generated, not source. Provenance excerpts must "
        "equal the source field. Preserve remaining uncertainty in uncertainties.\n"
        "- local_id values are your own stable keys (not tracker keys, not cbl_ ids).\n"
        "- Hierarchy is epic→feature→user_story→task→subtask.\n"
        "- Provenance json_path must be the Copilot envelope path on each fact (the fact's json_path "
        "value). Do not cite prompt-object keys or adapter attribute names such as $.contract_id, "
        "$.capabilities, $.functional_requirements, $.acceptance_criteria, "
        "$.non_functional_requirements, or $.blocking_dependencies. Array facts require the indexed "
        "envelope path (for example $.requirements.functional_requirements[0]).\n\n"
        "Normalized project facts:\n"
        f"{json.dumps(source.prompt_facts, ensure_ascii=False, indent=2)}\n"
    )


def _actionable_quality_findings(findings: _FindingSink) -> list[Finding]:
    return [item for item in findings.items if item.code in _QUALITY_REPAIR_CODES]


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
) -> str:
    listed = json.dumps(
        [_quality_finding_payload(item) for item in findings],
        ensure_ascii=False,
        indent=2,
    )
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
        "- Use each finding's message (titles, canonical IDs, proposal local_ids, overlapping "
        "action, evidence, and remediation) to locate the named pair. Fix that pair only.\n"
        "- When one story owns an action named in a finding, remove that leaked action from "
        "the non-owning story's description and acceptance criteria. Keep both stories when "
        "each is independently required for functional-requirement coverage.\n"
        "- If a source acceptance criterion concatenates independently owned outcomes, replace "
        "the compound copy with generated acceptance criteria scoped to each story. Keep "
        "source excerpts and json_path accurate. Never mark edited criterion text as "
        "origin=source.\n"
        "- Split a Story only when source evidence supports independently testable outcomes.\n"
        "- Remove duplicated responsibilities and acceptance criteria across Stories.\n"
        "- Add concrete child Tasks only when the source supports distinct implementation "
        "or verification work.\n"
        "- Treat non-functional requirements as constraints on relevant work, not "
        "automatically as separate technical stories. Never add a user_story whose only "
        "requirement provenance is a non-functional requirement; attach that path to related "
        "functional work instead.\n"
        "- Preserve source provenance, requirement coverage, valid dependencies, scope, "
        "and local_id values for unchanged items.\n"
        "- If implementation detail is unavailable, record a specific uncertainty or "
        "prerequisite rather than inventing architecture, local databases, endpoints, "
        "API contracts, payload formats, or business rules.\n"
        f"Quality findings:\n{listed}\n\n"
        "Previous JSON:\n"
        f"{json.dumps(previous, ensure_ascii=False)}\n"
    )


def _build_repair_prompt(
    original_prompt: str,
    previous: Mapping[str, Any],
    errors: list[str],
) -> str:
    listed = "\n".join(f"- {error}" for error in errors)
    return (
        f"{original_prompt}\n\n"
        "The previous JSON failed deterministic validation. Return a corrected JSON object "
        "for the same project facts. Do not mark the output as valid yourself.\n"
        f"Validation errors:\n{listed}\n\n"
        "Previous JSON:\n"
        f"{json.dumps(previous, ensure_ascii=False)}\n"
    )
