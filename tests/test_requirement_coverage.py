"""Coverage, grouping, and investigation-task placement — domain-independent."""

from __future__ import annotations

from types import SimpleNamespace

from backlog_agent.backlog.decompose import decompose_backlog
from backlog_agent.backlog.granularity import (
    item_represents_requirement,
    record_granularity_findings,
    requirement_outcomes,
)
from backlog_agent.backlog.models import WorkItemType
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import FindingCode
from backlog_agent.llm.mock import MockProvider
from tests.copilot_envelopes import clinic_rounds_envelope, field_inspection_envelope
from tests.test_decompose_generic import _item, _src
from tests.test_story_overlap import _Sink, _codes, _ref, _story


def _codes_of(findings) -> set[FindingCode]:
    return {item.code for item in findings.items}


def _field_payload():
    return field_inspection_envelope()


def _compound_inspection_proposal(*, include_sync_work: bool = False, split_features: bool = False) -> dict:
    payload = _field_payload()
    fr = payload["requirements"]["functional_requirements"]
    ac = payload["requirements"]["acceptance_criteria"]
    nfr = payload["requirements"]["non_functional_requirements"][0]
    unknown = payload["requirements"]["open_questions"][0]
    caps = payload["solution"]["key_capabilities"]
    if split_features:
        features = [
            _item(
                "feat-access",
                "feature",
                "Assigned installation context",
                parent="epic-1",
                provenance=[_src("$.solution.key_capabilities[0]", "key_capabilities", caps[0])],
                title_origin="generated",
            ),
            _item(
                "feat-context",
                "feature",
                caps[0],
                parent="epic-1",
                provenance=[_src("$.solution.key_capabilities[0]", "key_capabilities", caps[0])],
                title_origin="generated",
            ),
            _item(
                "feat-defect",
                "feature",
                caps[1],
                parent="epic-1",
                provenance=[_src("$.solution.key_capabilities[1]", "key_capabilities", caps[1])],
                title_origin="generated",
            ),
            _item(
                "feat-report",
                "feature",
                caps[2],
                parent="epic-1",
                provenance=[_src("$.solution.key_capabilities[2]", "key_capabilities", caps[2])],
                title_origin="generated",
            ),
        ]
        parents = ("feat-access", "feat-context", "feat-defect", "feat-report")
        if include_sync_work:
            features.append(
                _item(
                    "feat-sync",
                    "feature",
                    caps[3],
                    parent="epic-1",
                    provenance=[_src("$.solution.key_capabilities[3]", "key_capabilities", caps[3])],
                    title_origin="generated",
                )
            )
    else:
        features = [
            _item(
                "feat-exec",
                "feature",
                "Inspection Execution and Defect Recording",
                parent="epic-1",
                provenance=[_src("$.solution.key_capabilities[0]", "key_capabilities", caps[0])],
                title_origin="generated",
                description="Covers starting inspections, recording defects, and opening installation context.",
            ),
            _item(
                "feat-report",
                "feature",
                "Inspection Reporting and Follow-up Actions",
                parent="epic-1",
                provenance=[_src("$.solution.key_capabilities[2]", "key_capabilities", caps[2])],
                title_origin="generated",
            ),
        ]
        parents = ("feat-exec", "feat-exec", "feat-exec", "feat-report")
    stories = [
        _item(
            "story-access",
            "user_story",
            fr[0],
            parent=parents[0],
            provenance=[
                _src("$.requirements.functional_requirements[0]", "functional_requirements", fr[0]),
                _src("$.requirements.acceptance_criteria[0]", "acceptance_criteria", ac[0]),
            ],
            description="A technician opens the assigned installation and reads its current details.",
            ac=[ac[0]],
            ac_origins=["source"],
            tests=["Open an assigned installation and assert current details are visible."],
        ),
        _item(
            "story-complete",
            "user_story",
            fr[1],
            parent=parents[1],
            provenance=[
                _src("$.requirements.functional_requirements[1]", "functional_requirements", fr[1]),
                _src("$.requirements.acceptance_criteria[1]", "acceptance_criteria", ac[1]),
            ],
            description=(
                "A technician starts a field inspection, keeps captured work with the visit, "
                "and completes it after the site work."
            ),
            ac=[ac[1]],
            ac_origins=["source"],
            tests=["Start then complete an inspection and assert the visit is closed."],
        ),
        _item(
            "story-defect",
            "user_story",
            fr[2],
            parent=parents[2],
            provenance=[
                _src("$.requirements.functional_requirements[2]", "functional_requirements", fr[2]),
                _src("$.requirements.acceptance_criteria[2]", "acceptance_criteria", ac[2]),
            ],
            description="A technician records an observed defect against the installation.",
            ac=[ac[2]],
            ac_origins=["source"],
            tests=["Record a defect and assert it is stored against the installation."],
        ),
        _item(
            "story-report",
            "user_story",
            fr[3],
            parent=parents[3],
            provenance=[
                _src("$.requirements.functional_requirements[3]", "functional_requirements", fr[3]),
                _src("$.requirements.acceptance_criteria[3]", "acceptance_criteria", ac[3]),
            ],
            description="A technician submits the inspection report together with follow-up actions.",
            ac=[ac[3]],
            ac_origins=["source"],
            tests=["Submit a report and assert follow-up actions are included."],
        ),
    ]
    items = [
        _item(
            "epic-1",
            "epic",
            "Field installation inspections",
            parent=None,
            provenance=[_src("$.user_request", "user_request", payload["user_request"])],
            title_origin="generated",
        ),
        *features,
        *stories,
        _item(
            "task-sync-retry",
            "task",
            "Clarify remaining unspecified retry and conflict behavior when synchronizing captured work",
            parent="story-report",
            provenance=[_src("$.requirements.open_questions[0]", "open_questions", unknown)],
            title_origin="generated",
        ),
        _item(
            "task-sync-overlap",
            "task",
            "Investigate unspecified conflict handling for synchronizing captured work",
            parent="story-report",
            provenance=[_src("$.requirements.open_questions[0]", "open_questions", unknown)],
            title_origin="generated",
        ),
    ]
    if include_sync_work:
        items.append(
            _item(
                "story-sync",
                "user_story",
                "Keep captured inspection work usable until connectivity is restored and then synchronize it",
                parent="feat-sync" if split_features else parents[1],
                provenance=[
                    _src("$.requirements.functional_requirements[1]", "functional_requirements", fr[1]),
                    _src(
                        "$.requirements.non_functional_requirements[0]",
                        "non_functional_requirements",
                        nfr,
                    ),
                ],
                title_origin="generated",
                description=(
                    "A technician continues an in-progress inspection during unreliable "
                    "connectivity and the captured work synchronizes when connectivity returns."
                ),
                ac=[
                    "Captured inspection work remains usable during unreliable connectivity "
                    "and synchronizes when connectivity is restored."
                ],
                ac_origins=["generated"],
                tests=["Interrupt connectivity, capture work, restore the link, and assert sync."],
            )
        )
    return {"items": items, "prerequisites": [], "uncertainties": [unknown]}


def test_requirement_outcomes_split_coordinated_objects() -> None:
    outcomes = requirement_outcomes(
        "The system shall let a technician access installation details and previous inspection reports."
    )
    blob = " ".join(outcomes).lower()
    assert "installation" in blob
    assert "previous" in blob or "report" in blob
    assert len(outcomes) >= 2


def test_item_representation_is_not_provenance() -> None:
    story = SimpleNamespace(
        title="Access installation details",
        description="A technician opens current installation details.",
        acceptance_criteria=["Current installation details are visible."],
        test_requirements=[],
    )
    requirement = (
        "The system shall let a technician access installation details and previous inspection reports."
    )
    assert item_represents_requirement(story, requirement) is False


def test_missing_connectivity_behavior_is_uncovered_capability() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(field_inspection_envelope())
    backlog, findings = decompose_backlog(
        envelope,
        MockProvider(_compound_inspection_proposal()),
        repair=False,
    )
    uncovered = [
        item
        for item in findings.items
        if item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT
    ]
    assert uncovered
    excerpt = uncovered[0].source_references[0].excerpt.lower()
    assert "connectiv" in excerpt or "synchron" in excerpt
    blob = " ".join(
        f"{item.title} {item.description} {' '.join(item.acceptance_criteria)}"
        for item in backlog.items
        if item.type is WorkItemType.USER_STORY
    ).lower()
    assert "offline" not in blob
    assert "synchron" not in blob
    assert "connectiv" not in blob


def test_ambiguity_note_is_not_connectivity_coverage() -> None:
    payload = field_inspection_envelope()
    proposal = _compound_inspection_proposal()
    proposal["uncertainties"] = [
        payload["requirements"]["non_functional_requirements"][0]
        + " Mechanism is unspecified."
    ]
    _backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload),
        MockProvider(proposal),
        repair=False,
    )
    assert any(
        item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT for item in findings.items
    )


def test_connectivity_story_covers_the_required_capability() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(field_inspection_envelope())
    _backlog, findings = decompose_backlog(
        envelope,
        MockProvider(_compound_inspection_proposal(include_sync_work=True, split_features=True)),
        repair=False,
    )
    assert not any(
        item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT for item in findings.items
    )


def test_compound_feature_title_is_misaligned() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(field_inspection_envelope())
    _backlog, findings = decompose_backlog(
        envelope,
        MockProvider(_compound_inspection_proposal()),
        repair=False,
    )
    misaligned = [
        item for item in findings.items if item.code is FindingCode.FEATURE_CHILD_MISALIGNMENT
    ]
    assert misaligned
    joined = " ".join(item.message for item in misaligned).lower()
    assert "and" in joined
    assert "inspection" in joined or "defect" in joined or "reporting" in joined


def test_split_features_are_not_catch_alls() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(field_inspection_envelope())
    _backlog, findings = decompose_backlog(
        envelope,
        MockProvider(_compound_inspection_proposal(include_sync_work=True, split_features=True)),
        repair=False,
    )
    assert not any(
        item.code is FindingCode.FEATURE_CHILD_MISALIGNMENT for item in findings.items
    )


def test_partial_acceptance_criteria_are_flagged() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(field_inspection_envelope())
    _backlog, findings = decompose_backlog(
        envelope,
        MockProvider(_compound_inspection_proposal()),
        repair=False,
    )
    assert any(item.code is FindingCode.MISSING_VERIFICATION_COVERAGE for item in findings.items)
    access = next(
        item
        for item in _backlog.items
        if item.type is WorkItemType.USER_STORY and "previous inspection reports" in item.title.lower()
    )
    assert any(
        finding.code is FindingCode.MISSING_VERIFICATION_COVERAGE
        and access.canonical_id in finding.canonical_ids
        for finding in findings.items
    )


def test_duplicate_and_misplaced_investigation_tasks() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(field_inspection_envelope())
    _backlog, findings = decompose_backlog(
        envelope,
        MockProvider(_compound_inspection_proposal()),
        repair=False,
    )
    assert any(
        item.code is FindingCode.OVER_DECOMPOSITION
        and "investigation" in item.message.lower()
        for item in findings.items
    )
    assert any(
        item.code is FindingCode.AMBIGUOUS_DECOMPOSITION
        and "investigation task" in item.message.lower()
        for item in findings.items
    )


def test_confirmed_access_is_not_treated_as_unknown() -> None:
    payload = field_inspection_envelope()
    proposal = _compound_inspection_proposal()
    dep = payload["delivery_plan"]["dependencies"][0]
    proposal["items"].append(
        _item(
            "task-exist",
            "task",
            "Verify that API access and technical documentation are available",
            parent="story-complete",
            provenance=[_src("$.delivery_plan.dependencies[0]", "dependencies", dep)],
            title_origin="generated",
        )
    )
    from backlog_agent.llm.exceptions import DecompositionValidationError
    import pytest

    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(
            CopilotWorkflowResponseV1.model_validate(payload),
            MockProvider(proposal),
            repair=False,
        )
    assert any("established source fact" in error for error in exc.value.errors)


def test_clinic_rounds_unrelated_domain_same_rules() -> None:
    payload = clinic_rounds_envelope()
    fr = payload["requirements"]["functional_requirements"]
    ac = payload["requirements"]["acceptance_criteria"]
    nfr = payload["requirements"]["non_functional_requirements"][0]
    unknown = payload["requirements"]["open_questions"][0]
    proposal = {
        "items": [
            _item(
                "epic-1",
                "epic",
                "Bedside rounds",
                parent=None,
                provenance=[_src("$.user_request", "user_request", payload["user_request"])],
                title_origin="generated",
            ),
            _item(
                "feat-board",
                "feature",
                "Care Visibility and Shift Follow-up",
                parent="epic-1",
                provenance=[
                    _src(
                        "$.solution.key_capabilities[0]",
                        "key_capabilities",
                        payload["solution"]["key_capabilities"][0],
                    )
                ],
                title_origin="generated",
            ),
            _item(
                "story-plan",
                "user_story",
                fr[0],
                parent="feat-board",
                provenance=[
                    _src("$.requirements.functional_requirements[0]", "functional_requirements", fr[0]),
                    _src("$.requirements.acceptance_criteria[0]", "acceptance_criteria", ac[0]),
                ],
                description="A nurse opens the current care plan for the assigned patient.",
                ac=[ac[0]],
                ac_origins=["source"],
                tests=["Open the care plan and assert it is visible."],
            ),
            _item(
                "story-observe",
                "user_story",
                fr[1],
                parent="feat-board",
                provenance=[
                    _src("$.requirements.functional_requirements[1]", "functional_requirements", fr[1]),
                    _src("$.requirements.acceptance_criteria[1]", "acceptance_criteria", ac[1]),
                ],
                description="A nurse records a bedside observation.",
                ac=[ac[1]],
                ac_origins=["source"],
                tests=["Record an observation and assert it is stored."],
            ),
            _item(
                "story-follow",
                "user_story",
                fr[2],
                parent="feat-board",
                provenance=[
                    _src("$.requirements.functional_requirements[2]", "functional_requirements", fr[2]),
                    _src("$.requirements.acceptance_criteria[2]", "acceptance_criteria", ac[2]),
                ],
                description="A nurse raises a follow-up for the next shift.",
                ac=[ac[2]],
                ac_origins=["source"],
                tests=["Raise a follow-up and assert the next shift can see it."],
            ),
            _item(
                "task-conflict-a",
                "task",
                "Clarify unspecified conflict handling when synchronizing overlapping notes",
                parent="story-follow",
                provenance=[_src("$.requirements.open_questions[0]", "open_questions", unknown)],
                title_origin="generated",
            ),
            _item(
                "task-conflict-b",
                "task",
                "Investigate unspecified conflict handling for synchronizing overlapping notes",
                parent="story-follow",
                provenance=[_src("$.requirements.open_questions[0]", "open_questions", unknown)],
                title_origin="generated",
            ),
        ],
        "prerequisites": [],
        "uncertainties": [nfr],
    }
    _backlog, findings = decompose_backlog(
        CopilotWorkflowResponseV1.model_validate(payload),
        MockProvider(proposal),
        repair=False,
    )
    codes = _codes_of(findings)
    assert FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT in codes
    assert FindingCode.MISSING_VERIFICATION_COVERAGE in codes
    assert FindingCode.FEATURE_CHILD_MISALIGNMENT in codes
    assert FindingCode.OVER_DECOMPOSITION in codes


def test_unit_partial_ac_on_multi_outcome_requirement() -> None:
    requirement = (
        "The system shall let a member view current loans and previous overdue notices."
    )
    story = _story(
        "cbl_loans",
        "View current loans and previous overdue notices",
        description="A member opens current loans.",
        acceptance_criteria=["Current loans are listed."],
        provenance=[_ref("$.requirements.functional_requirements[0]", requirement)],
    )
    sink = _Sink()
    record_granularity_findings(
        SimpleNamespace(functional_requirements=(), non_functional_requirements=()),
        SimpleNamespace(items=[story]),
        sink,
    )
    assert FindingCode.MISSING_VERIFICATION_COVERAGE in _codes(sink)
    assert "previous" in sink.items[0].message.lower() or "overdue" in sink.items[0].message.lower()


def test_shared_system_tokens_do_not_cover_sibling_outcome() -> None:
    requirement = (
        "The system shall let a technician access installation details and previous "
        "inspection reports from the asset registry."
    )
    story = _story(
        "cbl_access",
        "Access Installation Details and Previous Reports",
        description="A technician accesses installation details from the asset registry.",
        acceptance_criteria=[
            "A technician can view installation details retrieved from the asset registry.",
            "Verify retrieval and display of installation records from the asset registry API.",
        ],
        provenance=[_ref("$.requirements.functional_requirements[0]", requirement)],
    )
    sink = _Sink()
    record_granularity_findings(
        SimpleNamespace(functional_requirements=(), non_functional_requirements=()),
        SimpleNamespace(items=[story]),
        sink,
    )
    assert FindingCode.MISSING_VERIFICATION_COVERAGE in _codes(sink)


def test_generic_product_wording_does_not_cover_material_constraint() -> None:
    nfr = (
        "The application shall support field work in remote sites where mobile "
        "connectivity may be unreliable."
    )
    story = SimpleNamespace(
        title="View Assigned Sites and Inspection Jobs",
        description="As a field technician, I want to view my assigned sites in the mobile application.",
        acceptance_criteria=[
            "A technician can view a list of their assigned sites and inspection jobs within the mobile application."
        ],
        test_requirements=["Verify that the interface lists assigned sites and jobs."],
    )
    assert item_represents_requirement(story, nfr, background={"technician", "assigned", "sites"}) is False


def test_completeness_gaps_trigger_quality_repair_and_remain_if_unfixed() -> None:
    envelope = CopilotWorkflowResponseV1.model_validate(field_inspection_envelope())
    incomplete = _compound_inspection_proposal()
    still_incomplete = _compound_inspection_proposal()
    provider = MockProvider(responses=[incomplete, still_incomplete])
    _backlog, findings = decompose_backlog(envelope, provider)
    assert len(provider.calls) == 2
    assert FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT.value in provider.calls[1][0]
    assert FindingCode.MISSING_VERIFICATION_COVERAGE.value in provider.calls[1][0]
    assert any(item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT for item in findings.items)
    assert any(item.code is FindingCode.MISSING_VERIFICATION_COVERAGE for item in findings.items)
    assert [call.kind for call in findings.llm_calls] == ["initial", "quality_repair"]
    # Acceptance contract: the repair did not resolve the blocking defects, so nothing is accepted.
    assert [call.accepted for call in findings.llm_calls] == [False, False]
    assert all(call.validation_passed for call in findings.llm_calls)
    assert all(call.blocking_quality_findings >= 1 for call in findings.llm_calls)
    assert any(item.code is FindingCode.QUALITY_REPAIR_REJECTED for item in findings.items)


def test_two_related_stories_may_share_one_feature() -> None:
    submit = _story(
        "cbl_submit",
        "Submit Completed Inspection Reports",
        description="A technician submits the completed inspection report.",
        acceptance_criteria=["The completed inspection report is submitted."],
    )
    follow = _story(
        "cbl_follow",
        "Create Follow-Up Maintenance Actions",
        description="A technician creates a follow-up maintenance action for an identified defect.",
        acceptance_criteria=["A follow-up maintenance action is created for the defect."],
    )
    feature = SimpleNamespace(
        type=WorkItemType.FEATURE,
        canonical_id="cbl_feat_report",
        title="Inspection reporting",
        description="Technicians submit reports and raise the follow-up work those reports identify.",
        child_ids=(submit.canonical_id, follow.canonical_id),
        provenance=(_ref("$.solution.key_capabilities[0]", "Inspection reporting"),),
        acceptance_criteria=[],
        test_requirements=[],
        parent_id=None,
        dependencies=[],
    )
    sink = _Sink()
    record_granularity_findings(
        SimpleNamespace(functional_requirements=(), non_functional_requirements=()),
        SimpleNamespace(items=[feature, submit, follow]),
        sink,
    )
    assert FindingCode.FEATURE_CHILD_MISALIGNMENT not in _codes(sink)
