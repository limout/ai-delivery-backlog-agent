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
    _classify_scope,
    _record_requirement_negations,
    _record_sow_list_conflicts,
    _record_uncovered_requirements,
    _source_ref,
    resolve_json_path,
)
from backlog_agent.backlog.ids import item_identity_key
from backlog_agent.backlog.models import (
    ApprovalState,
    CanonicalBacklogV1,
    WorkItemStatus,
    WorkItemType,
)
from backlog_agent.backlog.proposal import (
    BacklogProposal,
    ContentOrigin,
    ProposedItem,
    proposal_json_schema,
)
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.contracts.versions import COPILOT_WORKFLOW_RESPONSE_V1
from backlog_agent.findings.models import Finding, FindingCode, FindingSeverity, GenerationFindings
from backlog_agent.llm.exceptions import DecompositionValidationError
from backlog_agent.llm.provider import AIProvider

_PARENT_TYPE: dict[WorkItemType, WorkItemType | None] = {
    WorkItemType.EPIC: None,
    WorkItemType.FEATURE: WorkItemType.EPIC,
    WorkItemType.USER_STORY: WorkItemType.FEATURE,
    WorkItemType.TASK: WorkItemType.USER_STORY,
    WorkItemType.SUBTASK: WorkItemType.TASK,
}

_ALLOWED_PATH_PREFIXES = (
    "$.user_request",
    "$.executive_summary",
    "$.requirements",
    "$.solution",
    "$.delivery_plan",
    "$.sow",
)


def decompose_backlog(
    envelope: CopilotWorkflowResponseV1 | Mapping[str, Any],
    provider: AIProvider,
    *,
    repair: bool = True,
) -> tuple[CanonicalBacklogV1, GenerationFindings]:
    """Propose a canonical backlog via ``provider``, then validate it in Python.

    At most one repair call is made when the first proposal fails validation.
    The model cannot mark the result valid; ``CanonicalBacklogV1`` and coverage
    checks remain deterministic.
    """

    parsed = (
        envelope
        if isinstance(envelope, CopilotWorkflowResponseV1)
        else CopilotWorkflowResponseV1.model_validate(envelope)
    )
    original = parsed.original_envelope
    schema = proposal_json_schema()
    prompt = _build_prompt(original)
    raw = provider.generate_json(prompt, schema)
    result = _try_materialize(original, raw)

    if result.ok:
        return result.backlog, GenerationFindings(items=result.findings.items)

    if repair:
        repair_prompt = _build_repair_prompt(prompt, raw, result.errors)
        repaired = provider.generate_json(repair_prompt, schema)
        second = _try_materialize(original, repaired)
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


def _try_materialize(original: Mapping[str, Any], raw: Mapping[str, Any]) -> _MaterializeResult:
    findings = _FindingSink()
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
    errors.extend(_provenance_and_scope_errors(original, proposal, findings))
    if errors:
        return _MaterializeResult(ok=False, errors=errors, findings=findings)

    try:
        backlog = _to_canonical(original, proposal, findings)
    except (ValidationError, ValueError) as exc:
        errors.append(str(exc))
        return _MaterializeResult(ok=False, errors=errors, findings=findings)

    _record_uncovered_requirements(original, backlog.items, findings)
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

    _record_generated_content(proposal, backlog, findings)
    _record_uncertainties(proposal, backlog, findings)
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


def _path_allowed(json_path: str) -> bool:
    for prefix in _ALLOWED_PATH_PREFIXES:
        if json_path == prefix or json_path.startswith(prefix + ".") or json_path.startswith(prefix + "["):
            return True
    return False


def _provenance_and_scope_errors(
    original: Mapping[str, Any],
    proposal: BacklogProposal,
    findings: _FindingSink,
) -> list[str]:
    errors: list[str] = []
    for item in proposal.items:
        for index, ref in enumerate(item.provenance):
            label = f"{item.local_id}.provenance[{index}]"
            if not _path_allowed(ref.json_path):
                errors.append(f"{label}: json_path {ref.json_path!r} is not an allowed Copilot field")
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
            if ref.excerpt is not None:
                if isinstance(resolved, str):
                    if ref.excerpt != resolved:
                        errors.append(
                            f"{label}: excerpt does not equal the source value at {ref.json_path}"
                        )
                        findings.add(
                            Finding(
                                severity=FindingSeverity.ERROR,
                                code=FindingCode.UNGROUNDED_PROVENANCE,
                                message="Provenance excerpt does not match the Copilot field value.",
                                source_references=[
                                    _source_ref(ref.json_path, ref.field_name, excerpt=ref.excerpt)
                                ],
                            )
                        )
                elif not isinstance(resolved, str):
                    errors.append(f"{label}: excerpt was provided but {ref.json_path} is not a string")

        classification = _classify_scope(original, item.title)
        if classification == "out":
            errors.append(
                f"{item.local_id}: title matches sow.out_of_scope and must not appear in the backlog"
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
            if item.title not in excerpts:
                errors.append(
                    f"{item.local_id}: title_origin=source but title is not equal to any provenance excerpt"
                )
    return errors


def _to_canonical(
    original: Mapping[str, Any],
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
        backlog_id=_backlog_id(original),
        items=models,
        source_contract_id=COPILOT_WORKFLOW_RESPONSE_V1,
        source_provenance=[
            _source_ref(
                "$.user_request",
                "user_request",
                excerpt=original.get("user_request") if isinstance(original.get("user_request"), str) else None,
            )
        ],
    )


def _base_path(json_path: str) -> str:
    bracket = json_path.find("[")
    return json_path[:bracket] if bracket >= 0 else json_path


def _record_generated_content(
    proposal: BacklogProposal,
    backlog: CanonicalBacklogV1,
    findings: _FindingSink,
) -> None:
    by_local = {item.local_id: item for item in proposal.items}
    local_ids = list(by_local)
    for index, canonical in enumerate(backlog.items):
        proposed = by_local[local_ids[index]]
        origins = []
        if proposed.title_origin is ContentOrigin.GENERATED:
            origins.append("title")
        if proposed.description and proposed.description_origin is ContentOrigin.GENERATED:
            origins.append("description")
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


def _build_prompt(original: Mapping[str, Any]) -> str:
    compact = {
        "user_request": original.get("user_request"),
        "executive_summary": original.get("executive_summary"),
        "requirements": original.get("requirements"),
        "solution": original.get("solution"),
        "delivery_plan": original.get("delivery_plan"),
        "sow": original.get("sow"),
    }
    return (
        "You decompose a completed Copilot workflow_response into a tracker-independent "
        "software backlog. Return JSON only, matching the supplied schema.\n\n"
        "Rules:\n"
        "- Propose one concise project Epic title. Do not use the entire Markdown user_request as the title.\n"
        "- Features are product capabilities (solution.key_capabilities that are in scope), "
        "not delivery workstreams. Do not copy every workstream into a Feature.\n"
        "- Every in-scope functional requirement must be covered by at least one user_story "
        "whose provenance json_path points at that requirement and whose excerpt equals it.\n"
        "- Attach acceptance_criteria and test_requirements to stories when the source supports it. "
        "If Copilot does not bind criteria to a requirement, put them on the most relevant story "
        "and list the ambiguity in uncertainties. Do not invent customer commitments.\n"
        "- Honour sow.in_scope and sow.out_of_scope. Do not create items for out-of-scope text.\n"
        "- Delivery phases, workstreams, milestones, and deliverables may become tasks or subtasks "
        "or dependencies. Use meaningful parent-child links (epic→feature→user_story→task→subtask).\n"
        "- local_id values are your own stable keys (not tracker keys, not cbl_ ids).\n"
        "- title_origin=source only when title equals a provenance excerpt copied from Copilot. "
        "Otherwise title_origin=generated.\n"
        "- Provenance excerpts must equal the Copilot field at json_path. Do not paraphrase excerpts.\n"
        "- Preserve uncertainty in uncertainties. Do not silently resolve material ambiguity.\n"
        "- Do not invent effort, estimates, or customer commitments that are not in the source.\n\n"
        "Copilot envelope (relevant sections):\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}\n"
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
        "for the same Copilot envelope. Do not mark the output as valid yourself.\n"
        f"Validation errors:\n{listed}\n\n"
        "Previous JSON:\n"
        f"{json.dumps(previous, ensure_ascii=False)}\n"
    )
