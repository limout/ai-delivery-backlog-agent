from __future__ import annotations

import copy

import pytest

from backlog_agent.backlog.decompose import _build_prompt, decompose_backlog
from backlog_agent.source import UnsupportedInputError, load_project_source
from backlog_agent.backlog.models import WorkItemType
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import FindingCode
from backlog_agent.llm.exceptions import DecompositionValidationError
from backlog_agent.llm.mock import MockProvider
from tests.copilot_envelopes import library_holds_envelope, warehouse_picking_envelope


def _src(path: str, field: str, excerpt: str) -> dict[str, str]:
    return {"json_path": path, "field_name": field, "excerpt": excerpt}


def _item(
    local_id: str,
    type_name: str,
    title: str,
    *,
    parent: str | None,
    provenance: list[dict[str, str]],
    description: str = "Actor uses the capability and observes a concrete outcome.",
    depends: list[str] | None = None,
    ac: list[str] | None = None,
    tests: list[str] | None = None,
    title_origin: str = "source",
    ac_origins: list[str] | None = None,
) -> dict:
    payload = {
        "local_id": local_id,
        "type": type_name,
        "title": title,
        "description": description if type_name == "user_story" else "",
        "parent_local_id": parent,
        "depends_on_local_ids": depends or [],
        "acceptance_criteria": ac or [],
        "test_requirements": tests or [],
        "provenance": provenance,
        "title_origin": title_origin,
        "description_origin": "generated",
        "uncertainties": [],
    }
    if ac_origins is not None:
        payload["acceptance_criteria_origins"] = ac_origins
    if tests:
        payload["test_requirements_origins"] = ["generated"] * len(tests)
    return payload


def _library_proposal(*, cover_notify: bool = True, trace_nfr: bool = True) -> dict:
    payload = library_holds_envelope()
    fr = payload["requirements"]["functional_requirements"]
    ac = payload["requirements"]["acceptance_criteria"][0]
    caps = payload["solution"]["key_capabilities"]
    nfr = payload["requirements"]["non_functional_requirements"][0]
    items = [
        _item(
            "epic-1",
            "epic",
            "Member catalog and holds",
            parent=None,
            provenance=[_src("$.user_request", "user_request", payload["user_request"])],
            title_origin="generated",
            description="",
        ),
        _item(
            "feat-search",
            "feature",
            caps[0],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[0]", "key_capabilities", caps[0])],
        ),
        _item(
            "feat-hold",
            "feature",
            caps[1],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[1]", "key_capabilities", caps[1])],
        ),
        _item(
            "story-search",
            "user_story",
            fr[0],
            parent="feat-search",
            provenance=[_src("$.requirements.functional_requirements[0]", "functional_requirements", fr[0])],
            description=(
                "A signed-in member enters a title or author and receives matching holdings "
                "without paging through the full collection."
            ),
            ac=[ac],
            ac_origins=["source"],
            tests=["Run a catalog search fixture and assert matching holdings appear in ranked order."],
        ),
        _item(
            "story-hold",
            "user_story",
            fr[1],
            parent="feat-hold",
            provenance=[
                _src("$.requirements.functional_requirements[1]", "functional_requirements", fr[1]),
                *([_src("$.requirements.non_functional_requirements[0]", "non_functional_requirements", nfr)] if trace_nfr else []),
            ],
            description=(
                "A member selects an available copy and places a hold so the copy is reserved "
                "until pickup rather than remaining on the open shelf."
                + (
                    " Catalog search p95 latency must stay under 400 milliseconds."
                    if trace_nfr
                    else ""
                )
            ),
            tests=[
                "Contract test: hold request against an available copy returns reserved status.",
                *(
                    ["Assert catalog search p95 latency stays under 400 milliseconds."]
                    if trace_nfr
                    else []
                ),
            ],
        ),
    ]
    if cover_notify:
        items.append(
            _item(
                "story-notify",
                "user_story",
                fr[2],
                parent="feat-hold",
                provenance=[_src("$.requirements.functional_requirements[2]", "functional_requirements", fr[2])],
                description=(
                    "When a reserved copy reaches the pickup shelf, the member receives a notice "
                    "so they can collect it before the hold expires."
                ),
                tests=["Assert a pickup-ready event sends one notice to the holding member."],
                depends=["story-hold"],
            )
        )
    items.extend(
        [
            _item(
                "task-hold-api",
                "task",
                "Implement hold reservation against inventory availability",
                parent="story-hold",
                provenance=[_src("$.sow.in_scope[1]", "in_scope", payload["sow"]["in_scope"][1])],
                title_origin="generated",
            ),
            _item(
                "task-hold-ui",
                "task",
                "Add place-hold action on the holding detail screen",
                parent="story-hold",
                provenance=[_src("$.sow.in_scope[1]", "in_scope", payload["sow"]["in_scope"][1])],
                title_origin="generated",
            ),
            _item(
                "task-hold-notice",
                "task",
                "Queue pickup-ready notice after a successful hold",
                parent="story-hold",
                provenance=[_src("$.sow.deliverables[1]", "deliverables", payload["sow"]["deliverables"][1])],
                title_origin="generated",
            ),
        ]
    )
    return {
        "items": items,
        "prerequisites": [
            {
                "text": "Bibliographic vendor MARC feed access is not yet granted.",
                "status": "unresolved",
                "provenance": [
                    _src(
                        "$.delivery_plan.dependencies[0]",
                        "dependencies",
                        payload["delivery_plan"]["dependencies"][0],
                    )
                ],
            }
        ],
        "uncertainties": ["Identity provider for member login is not specified."],
    }


def _warehouse_proposal(*, omit_reprint: bool = False) -> dict:
    payload = warehouse_picking_envelope()
    fr = payload["requirements"]["functional_requirements"]
    ac = payload["requirements"]["acceptance_criteria"][0]
    caps = payload["solution"]["key_capabilities"]
    items = [
        _item(
            "epic-1",
            "epic",
            "Handheld warehouse picking",
            parent=None,
            provenance=[_src("$.user_request", "user_request", payload["user_request"])],
            title_origin="generated",
        ),
        _item(
            "feat-play",
            "feature",
            caps[0],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[0]", "key_capabilities", caps[0])],
        ),
        _item(
            "feat-scan",
            "feature",
            caps[1],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[1]", "key_capabilities", caps[1])],
        ),
        _item(
            "story-next",
            "user_story",
            fr[0],
            parent="feat-play",
            provenance=[_src("$.requirements.functional_requirements[0]", "functional_requirements", fr[0])],
            description=(
                "An operator opening an assigned list sees the next bin and SKU so they walk "
                "directly to the location instead of reading a paper ticket."
            ),
            tests=["Render the next incomplete pick and assert bin and SKU are visible."],
        ),
        _item(
            "story-scan",
            "user_story",
            fr[1],
            parent="feat-scan",
            provenance=[_src("$.requirements.functional_requirements[1]", "functional_requirements", fr[1])],
            description=(
                "An operator scans the presented bin and the pick is completed only when the "
                "scan matches the assigned location."
            ),
            ac=[ac],
            ac_origins=["source"],
            tests=["Scan the expected bin to complete; scan a different bin to keep the pick open."],
        ),
        _item(
            "task-scan-decode",
            "task",
            "Decode handheld barcode scans against the assigned bin",
            parent="story-scan",
            provenance=[_src("$.sow.in_scope[0]", "in_scope", payload["sow"]["in_scope"][0])],
            title_origin="generated",
        ),
        _item(
            "task-scan-complete",
            "task",
            "Mark the pick complete after a matching bin scan",
            parent="story-scan",
            provenance=[_src("$.sow.in_scope[0]", "in_scope", payload["sow"]["in_scope"][0])],
            title_origin="generated",
        ),
    ]
    if not omit_reprint:
        items.append(
            _item(
                "story-reprint",
                "user_story",
                fr[2],
                parent="feat-play",
                provenance=[_src("$.requirements.functional_requirements[2]", "functional_requirements", fr[2])],
                description=(
                    "A supervisor can reprint a completed pick list for audit without changing "
                    "the completed pick state."
                ),
                tests=["Reprint a completed list and assert pick statuses remain complete."],
            )
        )
    return {
        "items": items,
        "prerequisites": [
            {
                "text": "Barcode symbology for existing bins is unconfirmed.",
                "status": "unresolved",
                "provenance": [_src("$.sow.dependencies[0]", "dependencies", payload["sow"]["dependencies"][0])],
            }
        ],
        "uncertainties": ["Wi-Fi packet-loss target has no agreed measurement method."],
    }


def test_library_project_covers_frs_atomic_story_and_multi_task_story() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(library_holds_envelope())
    backlog, findings = decompose_backlog(envelope, MockProvider(_library_proposal()))

    stories = [item for item in backlog.items if item.type is WorkItemType.USER_STORY]
    search = next(item for item in stories if "search the catalog" in item.title.lower())
    hold = next(item for item in stories if "place a hold" in item.title.lower())
    assert search.child_ids == []
    assert len(hold.child_ids) == 3
    assert any(item.code is FindingCode.ATOMIC_STORY and search.canonical_id in item.canonical_ids for item in findings.items)
    assert not any(item.code is FindingCode.ATOMIC_STORY and hold.canonical_id in item.canonical_ids for item in findings.items)
    assert any(item.code is FindingCode.UNRESOLVED_PREREQUISITE for item in findings.items)
    assert any(item.code is FindingCode.AMBIGUOUS_DECOMPOSITION for item in findings.items)
    assert any(item.code is FindingCode.PLANNING_CONSTRAINT for item in findings.items)
    assert not any(item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT for item in findings.items)
    assert not any(item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT for item in findings.items)
    assert any(item.dependencies for item in backlog.items)
    generated = [item for item in findings.items if item.code is FindingCode.GENERATED_CONTENT]
    assert generated
    titles = {item.title for item in backlog.items}
    assert "Interlibrary loan routing" not in titles
    assert "Member experience" not in titles


def test_related_story_without_nfr_path_is_not_coverage() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(library_holds_envelope())
    proposal = _library_proposal(trace_nfr=False)
    _backlog, findings = decompose_backlog(envelope, MockProvider(proposal), repair=False)

    uncovered = [
        item for item in findings.items if item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT
    ]
    assert len(uncovered) == 1
    assert "400 milliseconds" in uncovered[0].source_references[0].excerpt


def test_timeline_constraint_is_not_a_work_item() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(library_holds_envelope())
    proposal = _library_proposal()
    constraint = library_holds_envelope()["requirements"]["non_functional_requirements"][1]
    proposal["items"].append(
        _item(
            "feat-deadline",
            "feature",
            constraint,
            parent="epic-1",
            provenance=[
                _src("$.requirements.non_functional_requirements[1]", "non_functional_requirements", constraint)
            ],
        )
    )
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(envelope, MockProvider(proposal), repair=False)
    assert any("timeline or estimate" in error for error in exc.value.errors)


def test_warehouse_project_rejects_uncovered_requirement() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(warehouse_picking_envelope())
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(envelope, MockProvider(_warehouse_proposal(omit_reprint=True)), repair=False)
    assert any("Uncovered functional requirement" in error for error in exc.value.errors)
    assert any(
        item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT for item in exc.value.findings.items
    )


def test_warehouse_atomic_and_multi_task_stories_with_generated_provenance() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(warehouse_picking_envelope())
    backlog, findings = decompose_backlog(envelope, MockProvider(_warehouse_proposal()), repair=False)

    stories = [item for item in backlog.items if item.type is WorkItemType.USER_STORY]
    next_pick = next(item for item in stories if "next pick" in item.title.lower())
    scan = next(item for item in stories if "confirm a pick" in item.title.lower())
    reprint = next(item for item in stories if "reprint" in item.title.lower())
    assert next_pick.child_ids == []
    assert reprint.child_ids == []
    assert len(scan.child_ids) == 2
    assert any("test_requirements" in item.message for item in findings.items if item.code is FindingCode.GENERATED_CONTENT)
    assert any(item.code is FindingCode.UNRESOLVED_PREREQUISITE for item in findings.items)
    nfr_uncovered = [
        item for item in findings.items if item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT
    ]
    assert nfr_uncovered
    assert "packet loss" in nfr_uncovered[0].source_references[0].excerpt.lower()
    titles = {item.title for item in backlog.items}
    assert "Packing optimization" not in titles


def test_out_of_scope_item_is_rejected_on_non_insurance_project() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(warehouse_picking_envelope())
    proposal = _warehouse_proposal()
    excluded = warehouse_picking_envelope()["sow"]["out_of_scope"][0]
    proposal["items"].append(
        _item(
            "feat-pack",
            "feature",
            excluded,
            parent="epic-1",
            provenance=[_src("$.sow.out_of_scope[0]", "out_of_scope", excluded)],
        )
    )
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(envelope, MockProvider(proposal), repair=False)
    assert any(item.code is FindingCode.OUT_OF_SCOPE_EXCLUDED for item in exc.value.findings.items)


def test_decompose_rejects_unsupported_input() -> None:
    with pytest.raises(UnsupportedInputError):
        decompose_backlog({"random": True}, MockProvider(_library_proposal()))


def test_invented_user_request_fr_path_is_rejected() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(library_holds_envelope())
    proposal = _library_proposal()
    story = next(item for item in proposal["items"] if item["local_id"] == "story-search")
    story["provenance"].append(
        _src("$.user_request.FR-04", "user_request", "Customer Authentication")
    )
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(envelope, MockProvider(proposal), repair=False)
    assert any("does not resolve" in error for error in exc.value.errors)


def test_user_request_subsection_excerpt_rewrites_to_full_field() -> None:
    payload = library_holds_envelope()
    envelope = CopilotWorkflowResponseV1.model_validate(payload)
    proposal = _library_proposal()
    epic = next(item for item in proposal["items"] if item["local_id"] == "epic-1")
    epic["provenance"][0]["excerpt"] = "### FR-04: Customer Authentication"
    backlog, _findings = decompose_backlog(envelope, MockProvider(proposal), repair=False)
    stored = next(item for item in backlog.items if item.type is WorkItemType.EPIC)
    assert stored.provenance[0].excerpt == payload["user_request"]
    assert "FR-04" not in (stored.provenance[0].excerpt or "")


def test_policy_listing_heading_matches_review_policies_story() -> None:
    payload = library_holds_envelope()
    payload["user_request"] += "\n\n### FR-02: Policy Listing\n"
    envelope = CopilotWorkflowResponseV1.model_validate(payload)
    proposal = _library_proposal()
    search = next(item for item in proposal["items"] if item["local_id"] == "story-search")
    search["title"] = "Review Personal Policies"
    search["title_origin"] = "generated"
    search["description"] = "A member reviews personal policies and inspects the listed coverage."
    search["acceptance_criteria"] = ["A member can review the policies on the account."]
    search["acceptance_criteria_origins"] = ["generated"]
    _backlog, findings = decompose_backlog(envelope, MockProvider(proposal), repair=False)
    assert not any(
        item.code is FindingCode.UNCOVERED_REQUEST_OUTCOME and "Policy Listing" in item.message
        for item in findings.items
    )


def test_repeated_uncertainty_notes_collapse_and_distinct_notes_remain() -> None:
    proposal = _library_proposal()
    repeated = "Availability and completeness of the catalog vendor API documentation"
    proposal["uncertainties"] = [
        "Identity provider for member login is not specified.",
        repeated,
        repeated + " remain unconfirmed.",
        "Pickup notice timing is unspecified.",
    ]
    search = next(item for item in proposal["items"] if item["local_id"] == "story-search")
    search["uncertainties"] = [repeated]
    _backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(library_holds_envelope()),
        MockProvider(proposal),
        repair=False,
    )
    ambiguous = [
        item.message
        for item in findings.items
        if item.code is FindingCode.AMBIGUOUS_DECOMPOSITION
    ]
    assert sum(repeated in message for message in ambiguous) == 1
    assert any("Pickup notice timing" in message for message in ambiguous)
    assert any("Identity provider" in message for message in ambiguous)


def test_distinct_uncovered_headings_are_reported_separately() -> None:
    payload = library_holds_envelope()
    payload["user_request"] += (
        "\n\n### FR-90: Completely Unique Widget Enumeration\n"
        "### NFR-91: Unverified Orbital Calibration\n"
    )
    envelope = CopilotWorkflowResponseV1.model_validate(payload)
    _backlog, findings = decompose_backlog(
        envelope, MockProvider(_library_proposal()), repair=False
    )
    warnings = [
        item.message
        for item in findings.items
        if item.code is FindingCode.UNCOVERED_REQUEST_OUTCOME
    ]
    assert any("FR-90" in message for message in warnings)
    assert any("NFR-91" in message for message in warnings)


def test_labeled_heading_warns_when_tokens_are_absent() -> None:
    payload = library_holds_envelope()
    payload["user_request"] = (
        payload["user_request"]
        + "\n\n### FR-99: Completely Unique Widget Enumeration\n"
        + "The portal must expose a widget enumeration listed only in this heading.\n"
        + "### NFR-99: Launch within 6 weeks pending feasibility\n"
    )
    envelope = CopilotWorkflowResponseV1.model_validate(payload)
    _backlog, findings = decompose_backlog(
        envelope, MockProvider(_library_proposal()), repair=False
    )
    warnings = [
        item for item in findings.items if item.code is FindingCode.UNCOVERED_REQUEST_OUTCOME
    ]
    assert len(warnings) == 1
    assert "FR-99" in warnings[0].message
    assert "heuristic" in warnings[0].message.lower()
    assert "does not prove" in warnings[0].message.lower()
    assert warnings[0].source_references[0].json_path == "$.user_request"
    excerpt = warnings[0].source_references[0].excerpt or ""
    assert excerpt.startswith("### FR-99:")
    assert excerpt in payload["user_request"]
    assert "$.user_request.FR-99" not in excerpt


def test_labeled_heading_skips_outcomes_already_in_structured_facts() -> None:
    payload = library_holds_envelope()
    payload["user_request"] = (
        payload["user_request"] + "\n\n### FR-01: Catalog Search\nMembers search holdings.\n"
    )
    envelope = CopilotWorkflowResponseV1.model_validate(payload)
    _backlog, findings = decompose_backlog(
        envelope, MockProvider(_library_proposal()), repair=False
    )
    assert not any(
        item.code is FindingCode.UNCOVERED_REQUEST_OUTCOME for item in findings.items
    )


def test_generation_prompt_covers_ownership_discovery_and_narrative_limits() -> None:
    prompt = _build_prompt(load_project_source(library_holds_envelope())).lower()
    assert "merely to obtain a provenance" in prompt
    assert "never invent paths" in prompt
    assert "zero tasks is valid" in prompt
    assert "confirming whether api documentation" in prompt
    assert "not confirmed" in prompt
    assert "source of truth" in prompt
    assert "list visibility" in prompt
    assert "do not create extra features" in prompt
    assert "do not put that source path on both stories" in prompt
    assert "auditability" in prompt
    assert "implement backend" in prompt
    assert "labeled_request_outcomes" in prompt or "labeled fr/nfr" in prompt


def test_authentication_heading_is_covered_by_story_wording_or_feature_title() -> None:
    payload = library_holds_envelope()
    payload["user_request"] += "\n\n### FR-01: Customer Authentication\n"
    envelope = CopilotWorkflowResponseV1.model_validate(payload)
    by_story = _library_proposal()
    search = next(item for item in by_story["items"] if item["local_id"] == "story-search")
    search["description"] = (
        "A signed-in member can authenticate and then search holdings by title or author."
    )
    _backlog, findings = decompose_backlog(envelope, MockProvider(by_story), repair=False)
    assert not any(
        item.code is FindingCode.UNCOVERED_REQUEST_OUTCOME and "FR-01" in item.message
        for item in findings.items
    )

    by_feature = _library_proposal()
    feature = next(item for item in by_feature["items"] if item["local_id"] == "feat-search")
    feature["title"] = "Customer Authentication"
    feature["title_origin"] = "generated"
    _backlog, feature_findings = decompose_backlog(envelope, MockProvider(by_feature), repair=False)
    assert not any(
        item.code is FindingCode.UNCOVERED_REQUEST_OUTCOME and "FR-01" in item.message
        for item in feature_findings.items
    )


def test_scoped_list_criterion_does_not_claim_term_review() -> None:
    payload = library_holds_envelope()
    compound = "A member can view holdings and review lending terms."
    payload["requirements"]["acceptance_criteria"] = [compound]
    envelope = CopilotWorkflowResponseV1.model_validate(payload)
    proposal = _library_proposal()
    search = next(item for item in proposal["items"] if item["local_id"] == "story-search")
    search["title"] = "View Catalog Holdings"
    search["title_origin"] = "generated"
    search["acceptance_criteria"] = ["A member can view the listed holdings."]
    search["acceptance_criteria_origins"] = ["generated"]
    search["provenance"].append(_src("$.requirements.acceptance_criteria[0]", "acceptance_criteria", compound))
    terms = next(item for item in proposal["items"] if item["local_id"] == "story-notify")
    terms["title"] = "Review Lending Terms"
    terms["title_origin"] = "generated"
    terms["description"] = "A member reviews lending terms before requesting a copy."
    terms["acceptance_criteria"] = ["A member can review the lending terms before requesting a copy."]
    terms["acceptance_criteria_origins"] = ["generated"]
    backlog, findings = decompose_backlog(envelope, MockProvider(proposal), repair=False)
    stored = next(item for item in backlog.items if item.title == "View Catalog Holdings")
    assert stored.acceptance_criteria == ["A member can view the listed holdings."]
    assert compound not in stored.acceptance_criteria
    assert any(reference.excerpt == compound for reference in stored.provenance)
    generated = [
        item
        for item in findings.items
        if item.code is FindingCode.GENERATED_CONTENT and stored.canonical_id in item.canonical_ids
    ]
    assert generated
    assert "acceptance_criteria" in generated[0].message
    assert not any(item.code is FindingCode.OVERLAPPING_STORY_RESPONSIBILITY for item in findings.items)
    assert not any(item.code is FindingCode.DUPLICATE_ACCEPTANCE_CRITERION for item in findings.items)


def test_hard_repair_keeps_equivalent_investigation_task() -> None:
    payload = library_holds_envelope()
    dependency = payload["delivery_plan"]["dependencies"][0]
    initial = _library_proposal()
    next(item for item in initial["items"] if item["local_id"] == "story-search")["description"] = ""
    task = _item(
        "task-docs",
        "task",
        "Confirm whether bibliographic vendor documentation is available",
        parent="story-hold",
        provenance=[_src("$.delivery_plan.dependencies[0]", "dependencies", dependency)],
        title_origin="generated",
    )
    initial["items"].append(task)
    repaired = _library_proposal()
    repaired["items"].append(copy.deepcopy(task))
    provider = MockProvider(responses=[initial, repaired])
    backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload), provider
    )
    prompt = provider.calls[1][0]
    assert len(provider.calls) == 2
    assert "task-docs" in prompt
    assert "bibliographic vendor documentation" in prompt
    assert "$.delivery_plan.dependencies[0]" in prompt
    assert "previous tasks to preserve" in prompt.lower()
    assert "do not merge the previous tree" in prompt.lower()
    assert [call.kind for call in findings.llm_calls] == ["initial", "hard_repair"]
    assert findings.llm_calls[1].accepted is True
    assert any("documentation is available" in item.title for item in backlog.items)
    assert not any("omitted grounded investigation task" in item.message for item in findings.items)


def test_invalid_title_origin_triggers_one_hard_repair_that_keeps_coverage() -> None:
    payload = library_holds_envelope()
    dependency = payload["delivery_plan"]["dependencies"][0]
    initial = _library_proposal()
    search = next(item for item in initial["items"] if item["local_id"] == "story-search")
    search["title"] = "Unrelated quarterly billing export"
    search["title_origin"] = "source"
    task = _item(
        "task-docs",
        "task",
        "Confirm whether bibliographic vendor documentation is available",
        parent="story-hold",
        provenance=[_src("$.delivery_plan.dependencies[0]", "dependencies", dependency)],
        title_origin="generated",
    )
    initial["items"].append(task)
    repaired = _library_proposal()
    repaired["items"].append(copy.deepcopy(task))
    provider = MockProvider(responses=[initial, repaired, _library_proposal()])
    backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload), provider
    )
    prompt = provider.calls[1][0]
    assert len(provider.calls) == 2
    assert "actionable quality findings" not in prompt
    assert "title_origin=source only when the title string equals" in prompt
    assert "do not require a task under every story" in prompt.lower()
    assert "do not add investigation tasks the source does not support" in prompt.lower()
    assert [call.kind for call in findings.llm_calls] == ["initial", "hard_repair"]
    assert findings.llm_calls[0].validation_passed is False
    assert findings.llm_calls[0].accepted is False
    assert findings.llm_calls[1].validation_passed is True
    assert findings.llm_calls[1].accepted is True
    covered = {
        reference.json_path
        for item in backlog.items
        for reference in item.provenance
        if reference.json_path.startswith("$.requirements.functional_requirements")
    }
    assert covered == {f"$.requirements.functional_requirements[{index}]" for index in range(3)}
    assert any("documentation is available" in item.title for item in backlog.items)


def _repair_that_moves_notify_onto_a_feature() -> dict:
    payload = library_holds_envelope()
    repaired = _library_proposal(cover_notify=False)
    feature = next(item for item in repaired["items"] if item["local_id"] == "feat-hold")
    feature["provenance"].append(
        _src(
            "$.requirements.functional_requirements[2]",
            "functional_requirements",
            payload["requirements"]["functional_requirements"][2],
        )
    )
    return repaired


def _repair_with_workstream_feature_and_combined_story() -> dict:
    payload = library_holds_envelope()
    proposal = _library_proposal(cover_notify=False)
    fr = payload["requirements"]["functional_requirements"]
    hold = next(item for item in proposal["items"] if item["local_id"] == "story-hold")
    hold["provenance"].append(
        _src("$.requirements.functional_requirements[2]", "functional_requirements", fr[2])
    )
    hold["parent_local_id"] = "feat-stream"
    proposal["items"].append(
        _item(
            "feat-stream",
            "feature",
            "Member experience",
            parent="epic-1",
            provenance=[
                _src(
                    "$.delivery_plan.workstreams[0]",
                    "workstreams",
                    payload["delivery_plan"]["workstreams"][0],
                )
            ],
            title_origin="source",
        )
    )
    return proposal


def test_hard_repair_corrects_workstream_feature_and_combined_requirements() -> None:
    payload = library_holds_envelope()
    initial = _library_proposal()
    search = next(item for item in initial["items"] if item["local_id"] == "story-search")
    search["title"] = "Unrelated quarterly billing export"
    search["title_origin"] = "source"
    provider = MockProvider(responses=[initial, _repair_with_workstream_feature_and_combined_story()])

    backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload), provider
    )

    assert len(provider.calls) == 2
    prompt = provider.calls[1][0]
    assert "When a feature title equals a delivery workstream" in prompt
    assert "one user_story per path" in prompt
    assert findings.llm_calls[1].validation_passed is True
    assert not any(
        item.type is WorkItemType.FEATURE and item.title == "Member experience" for item in backlog.items
    )
    stories = [item for item in backlog.items if item.type is WorkItemType.USER_STORY]
    covered: dict[str, int] = {}
    for story in stories:
        paths = [
            reference.json_path
            for reference in story.provenance
            if reference.json_path.startswith("$.requirements.functional_requirements")
        ]
        assert len(paths) == 1
        covered[paths[0]] = covered.get(paths[0], 0) + 1
    assert set(covered) == {f"$.requirements.functional_requirements[{index}]" for index in range(3)}
    assert not any(item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT for item in findings.items)


def test_execution_structure_repair_does_not_bypass_other_validation() -> None:
    payload = library_holds_envelope()
    initial = _library_proposal()
    search = next(item for item in initial["items"] if item["local_id"] == "story-search")
    search["title"] = "Unrelated quarterly billing export"
    search["title_origin"] = "source"
    repaired = _repair_with_workstream_feature_and_combined_story()
    repaired["items"] = [
        item
        for item in repaired["items"]
        if item["local_id"] not in {"story-search", "feat-search"}
    ]
    provider = MockProvider(responses=[initial, repaired])

    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(CopilotWorkflowResponseV1.model_validate(payload), provider)

    assert len(provider.calls) == 2
    assert any("Uncovered functional requirement" in error for error in exc.value.errors)
    assert not any("must not combine independently testable" in error for error in exc.value.errors)
    assert not any("workstreams must not be copied as features" in error for error in exc.value.errors)
    assert exc.value.findings.llm_calls[1].validation_passed is False
    assert exc.value.findings.llm_calls[1].accepted is False


def test_hard_repair_restores_user_story_coverage_moved_onto_a_feature() -> None:
    payload = library_holds_envelope()
    initial = _library_proposal()
    search = next(item for item in initial["items"] if item["local_id"] == "story-search")
    search["title"] = "Unrelated quarterly billing export"
    search["title_origin"] = "source"
    provider = MockProvider(responses=[initial, _repair_that_moves_notify_onto_a_feature()])

    backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload), provider
    )

    assert len(provider.calls) == 2
    prompt = provider.calls[1][0]
    assert "must remain on a user_story" in prompt
    assert "$.requirements.functional_requirements[2]" in prompt
    assert "Citing the path only on an epic, feature, or task does not preserve coverage" in prompt
    assert [call.kind for call in findings.llm_calls] == ["initial", "hard_repair"]
    assert findings.llm_calls[1].validation_passed is True
    assert findings.llm_calls[1].lost_fr_paths == []
    stories = [
        item
        for item in backlog.items
        if item.type is WorkItemType.USER_STORY
        and any(
            reference.json_path == "$.requirements.functional_requirements[2]"
            for reference in item.provenance
        )
    ]
    assert len(stories) == 1
    assert not any(
        item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT for item in findings.items
    )


def test_hard_repair_does_not_keep_an_invalid_restored_story() -> None:
    payload = library_holds_envelope()
    initial = _library_proposal()
    search = next(item for item in initial["items"] if item["local_id"] == "story-search")
    search["title"] = "Unrelated quarterly billing export"
    search["title_origin"] = "source"
    notify = next(item for item in initial["items"] if item["local_id"] == "story-notify")
    notify["title"] = "Unrelated quarterly billing export"
    notify["title_origin"] = "source"
    provider = MockProvider(responses=[initial, _repair_that_moves_notify_onto_a_feature()])

    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(CopilotWorkflowResponseV1.model_validate(payload), provider)

    assert len(provider.calls) == 2
    assert [call.kind for call in exc.value.findings.llm_calls] == ["initial", "hard_repair"]
    assert exc.value.findings.llm_calls[1].validation_passed is False
    assert "$.requirements.functional_requirements[2]" in exc.value.findings.llm_calls[1].lost_fr_paths
    diagnostic = exc.value.coverage_diagnostic
    assert diagnostic["uncovered_fr_paths"] == ["$.requirements.functional_requirements[2]"]
    assert diagnostic["non_story_citations"] == [
        {"path": "$.requirements.functional_requirements[2]", "item_types": ["feature"]}
    ]


def test_hard_repair_does_not_splice_back_a_dropped_task() -> None:
    payload = library_holds_envelope()
    dependency = payload["delivery_plan"]["dependencies"][0]
    initial = _library_proposal()
    next(item for item in initial["items"] if item["local_id"] == "story-search")["description"] = ""
    initial["items"].append(
        _item(
            "task-docs",
            "task",
            "Confirm whether bibliographic vendor documentation is available",
            parent="story-hold",
            provenance=[_src("$.delivery_plan.dependencies[0]", "dependencies", dependency)],
            title_origin="generated",
        )
    )
    provider = MockProvider(responses=[initial, _library_proposal()])
    backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload), provider
    )
    assert len(provider.calls) == 2
    assert findings.llm_calls[1].accepted is True
    assert not any("documentation is available" in item.title for item in backlog.items)
    omitted = [
        item
        for item in findings.items
        if item.code is FindingCode.MISSING_IMPLEMENTATION_DETAIL
        and "omitted grounded investigation task" in item.message
    ]
    assert len(omitted) == 1
    assert "bibliographic vendor documentation" in omitted[0].message


def test_generic_filler_tasks_are_rejected() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(library_holds_envelope())
    proposal = _library_proposal()
    proposal["items"].append(
        _item(
            "task-filler",
            "task",
            "Implement backend",
            parent="story-hold",
            provenance=[
                _src(
                    "$.sow.in_scope[1]",
                    "in_scope",
                    library_holds_envelope()["sow"]["in_scope"][1],
                )
            ],
            title_origin="generated",
        )
    )
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(envelope, MockProvider(proposal), repair=False)
    assert any("generic or filler" in error for error in exc.value.errors)


def test_grounded_discovery_task_from_blocking_unknown_is_accepted() -> None:
    payload = library_holds_envelope()
    envelope = CopilotWorkflowResponseV1.model_validate(payload)
    proposal = _library_proposal()
    proposal["items"].append(
        _item(
            "task-discover-marc",
            "task",
            "Confirm whether bibliographic vendor MARC feed documentation is available",
            parent="story-hold",
            provenance=[
                _src(
                    "$.delivery_plan.dependencies[0]",
                    "dependencies",
                    payload["delivery_plan"]["dependencies"][0],
                )
            ],
            title_origin="generated",
        )
    )
    backlog, _findings = decompose_backlog(envelope, MockProvider(proposal), repair=False)
    assert any(
        "MARC feed documentation" in item.title for item in backlog.items
    )


def test_nfr_counts_only_when_its_path_is_on_an_item() -> None:
    payload = library_holds_envelope()
    constraint = "The catalog service must remain the authoritative source of truth for holdings."
    payload["requirements"]["non_functional_requirements"].append(constraint)
    envelope = CopilotWorkflowResponseV1.model_validate(payload)
    missing = _library_proposal()
    _backlog, uncovered_findings = decompose_backlog(envelope, MockProvider(missing), repair=False)
    uncovered = [
        item
        for item in uncovered_findings.items
        if item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT
        and item.source_references
        and item.source_references[0].excerpt == constraint
    ]
    assert len(uncovered) == 1

    attached = _library_proposal()
    search = next(item for item in attached["items"] if item["local_id"] == "story-search")
    search["provenance"].append(
        _src(
            "$.requirements.non_functional_requirements[2]",
            "non_functional_requirements",
            constraint,
        )
    )
    search["description"] = (
        search["description"]
        + " The catalog service remains the authoritative source of truth for holdings."
    )
    search["test_requirements"] = [
        *search["test_requirements"],
        "Assert the catalog service remains the authoritative source of truth for holdings.",
    ]
    search["test_requirements_origins"] = ["generated"] * len(search["test_requirements"])
    _covered, covered_findings = decompose_backlog(envelope, MockProvider(attached), repair=False)
    assert not any(
        item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT
        and item.source_references
        and item.source_references[0].excerpt == constraint
        for item in covered_findings.items
    )


def test_unverifiable_unknowns_may_yield_zero_tasks() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(library_holds_envelope())
    proposal = _library_proposal()
    proposal["items"] = [item for item in proposal["items"] if item["type"] != "task"]
    backlog, findings = decompose_backlog(envelope, MockProvider(proposal), repair=False)
    assert not any(item.type is WorkItemType.TASK for item in backlog.items)
    assert any(item.code is FindingCode.UNRESOLVED_PREREQUISITE for item in findings.items)
    assert not any(item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT for item in findings.items)
