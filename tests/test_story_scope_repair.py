"""Shared acceptance criteria are scoped, and leftover blocking defects are not accepted."""

from __future__ import annotations

from backlog_agent.backlog.decompose import decompose_backlog
from backlog_agent.backlog.quality_gate import blocking_quality_findings
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import FindingCode
from backlog_agent.llm.mock import MockProvider
from tests.copilot_envelopes import _complete
from tests.test_decompose_generic import _item, _src
from tests.test_decompose_granularity import _base_items, _envelope, _proposal


def _delivery_envelope(*, integrations: bool = True) -> dict:
    requirements = [
        "The system shall let a reviewer review incoming work.",
        "The system shall let a reviewer track delivery status.",
    ]
    criteria = ["Review incoming work and track delivery status."]
    capabilities = ["Review incoming work.", "Track delivery status."]
    if integrations:
        requirements.extend(
            [
                "The release shall integrate with the project management system.",
                "The release shall integrate with the source-code repository.",
            ]
        )
        criteria.append(
            "The release integrates with the project management system and the source-code repository."
        )
        capabilities.extend(
            ["Project management integration.", "Source-code repository integration."]
        )
    return _complete(
        user_request=(
            "Review incoming work and track delivery status. Integrate with the project "
            "management system and the source-code repository."
        ),
        executive_summary="Incoming work is reviewed and delivery status is tracked.",
        requirements={
            "functional_requirements": requirements,
            "acceptance_criteria": criteria,
            "non_functional_requirements": [],
        },
        solution={"key_capabilities": capabilities},
        sow={
            "in_scope": ["Incoming work review", "Delivery status", "Integrations"],
            "out_of_scope": ["Billing"],
            "deliverables": ["Review", "Status", "Integrations"],
        },
    )


def _story(
    local_id: str,
    title: str,
    parent: str,
    fr_index: int,
    fr_text: str,
    *,
    description: str,
    acceptance: list[str],
    ac_paths: list[tuple[int, str]] | None = None,
) -> dict:
    provenance = [
        _src(
            f"$.requirements.functional_requirements[{fr_index}]",
            "functional_requirements",
            fr_text,
        )
    ]
    for index, excerpt in ac_paths or []:
        provenance.append(
            _src(
                f"$.requirements.acceptance_criteria[{index}]",
                "acceptance_criteria",
                excerpt,
            )
        )
    return _item(
        local_id,
        "user_story",
        title,
        parent=parent,
        provenance=provenance,
        title_origin="generated",
        description=description,
        ac=acceptance,
        ac_origins=["source"] * len(acceptance),
        tests=[f"Verify {title}."],
    )


def _tree(payload: dict, stories: list[dict]) -> dict:
    caps = payload["solution"]["key_capabilities"]
    items = [
        _item(
            "epic-1",
            "epic",
            "Work delivery",
            parent=None,
            provenance=[_src("$.user_request", "user_request", payload["user_request"])],
            title_origin="generated",
        )
    ]
    for index, capability in enumerate(caps):
        items.append(
            _item(
                f"feat-{index}",
                "feature",
                capability,
                parent="epic-1",
                provenance=[
                    _src(f"$.solution.key_capabilities[{index}]", "key_capabilities", capability)
                ],
                title_origin="generated",
            )
        )
    items.extend(stories)
    return {"items": items, "prerequisites": [], "uncertainties": []}


def _invalid(payload: dict) -> dict:
    return {
        "items": [
            _item(
                "story-only",
                "user_story",
                "Review Incoming Work",
                parent=None,
                provenance=[
                    _src(
                        "$.requirements.functional_requirements[0]",
                        "functional_requirements",
                        payload["requirements"]["functional_requirements"][0],
                    )
                ],
                title_origin="generated",
                description="A reviewer reviews incoming work.",
            )
        ],
        "prerequisites": [],
        "uncertainties": [],
    }


def _by_title(backlog):
    return {item.title: item for item in backlog.items}


def test_shared_compound_criterion_is_scoped_per_story() -> None:
    payload = _delivery_envelope(integrations=False)
    fr = payload["requirements"]["functional_requirements"]
    shared = payload["requirements"]["acceptance_criteria"]
    stories = [
        _story(
            "story-review",
            "Review Incoming Work",
            "feat-0",
            0,
            fr[0],
            description="A reviewer reviews incoming work and track delivery status.",
            acceptance=[shared[0]],
            ac_paths=[(0, shared[0])],
        ),
        _story(
            "story-status",
            "Track Delivery Status",
            "feat-1",
            1,
            fr[1],
            description="A reviewer tracks delivery status and review incoming work.",
            acceptance=[shared[0]],
            ac_paths=[(0, shared[0])],
        ),
    ]
    provider = MockProvider(responses=[_invalid(payload), _tree(payload, stories)])
    backlog, findings = decompose_backlog(CopilotWorkflowResponseV1.model_validate(payload), provider)
    items = _by_title(backlog)
    review = " ".join(items["Review Incoming Work"].acceptance_criteria).casefold()
    status = " ".join(items["Track Delivery Status"].acceptance_criteria).casefold()

    assert "delivery status" not in review
    assert "incoming work" not in status
    assert "incoming work" in review
    assert "delivery status" in status
    assert not any(item.code is FindingCode.DUPLICATE_ACCEPTANCE_CRITERION for item in findings.items)
    assert not any(item.code is FindingCode.OVERLAPPING_STORY_RESPONSIBILITY for item in findings.items)
    assert findings.llm_calls[1].kind == "hard_repair"
    assert findings.llm_calls[1].validation_passed is True
    assert findings.llm_calls[1].accepted is True
    assert findings.llm_calls[1].blocking_quality_findings == 0


def test_integration_criteria_stay_specific_to_each_system() -> None:
    payload = _delivery_envelope()
    fr = payload["requirements"]["functional_requirements"]
    shared = payload["requirements"]["acceptance_criteria"][1]
    stories = [
        _story(
            "story-pms",
            "Project Management System Integration",
            "feat-2",
            2,
            fr[2],
            description="The release integrates with the project management system.",
            acceptance=[shared],
            ac_paths=[(1, shared)],
        ),
        _story(
            "story-repo",
            "Source-Code Repository Integration",
            "feat-3",
            3,
            fr[3],
            description="The release integrates with the source-code repository.",
            acceptance=[shared],
            ac_paths=[(1, shared)],
        ),
    ]
    # Cover the other functional requirements so hard validation can pass.
    stories.extend(
        [
            _story(
                "story-review",
                "Review Incoming Work",
                "feat-0",
                0,
                fr[0],
                description="A reviewer reviews incoming work.",
                acceptance=["A reviewer can review incoming work."],
                ac_paths=[],
            ),
            _story(
                "story-status",
                "Track Delivery Status",
                "feat-1",
                1,
                fr[1],
                description="A reviewer tracks delivery status.",
                acceptance=["A reviewer can track delivery status."],
                ac_paths=[],
            ),
        ]
    )
    provider = MockProvider(responses=[_invalid(payload), _tree(payload, stories)])
    backlog, findings = decompose_backlog(CopilotWorkflowResponseV1.model_validate(payload), provider)
    items = _by_title(backlog)
    pms = " ".join(items["Project Management System Integration"].acceptance_criteria).casefold()
    repo = " ".join(items["Source-Code Repository Integration"].acceptance_criteria).casefold()

    assert "project management" in pms
    assert "source-code" not in pms and "repository" not in pms
    assert "source-code" in repo or "repository" in repo
    assert "project management" not in repo
    for item in (
        items["Project Management System Integration"],
        items["Source-Code Repository Integration"],
    ):
        for criterion in item.acceptance_criteria:
            assert criterion != shared
            assert all(reference.excerpt != criterion for reference in item.provenance)
    assert any(
        item.code is FindingCode.GENERATED_CONTENT
        and "acceptance_criteria" in item.message
        for item in findings.items
    )
    assert findings.llm_calls[1].accepted is True
    assert not blocking_quality_findings(findings.items)


def test_unscoped_blocking_defect_after_hard_repair_is_not_accepted() -> None:
    payload = _delivery_envelope()
    fr = payload["requirements"]["functional_requirements"]
    shared = "The record is stored for later review."
    payload["requirements"]["functional_requirements"][0] = shared
    payload["requirements"]["acceptance_criteria"][0] = shared
    stories = [
        _story(
            "story-review",
            "Review Incoming Work",
            "feat-0",
            0,
            shared,
            description="Track delivery status.",
            acceptance=[shared],
            ac_paths=[(0, shared)],
        ),
        _story(
            "story-status",
            "Track Delivery Status",
            "feat-1",
            1,
            fr[1],
            description="Review incoming work.",
            acceptance=[shared],
            ac_paths=[(0, shared)],
        ),
        _story(
            "story-pms",
            "Project Management System Integration",
            "feat-2",
            2,
            fr[2],
            description="The release integrates with the project management system.",
            acceptance=["The release integrates with the project management system."],
        ),
        _story(
            "story-repo",
            "Source-Code Repository Integration",
            "feat-3",
            3,
            fr[3],
            description="The release integrates with the source-code repository.",
            acceptance=["The release integrates with the source-code repository."],
        ),
    ]
    provider = MockProvider(responses=[_invalid(payload), _tree(payload, stories)])
    backlog, findings = decompose_backlog(CopilotWorkflowResponseV1.model_validate(payload), provider)

    assert findings.llm_calls[1].kind == "hard_repair"
    assert findings.llm_calls[1].validation_passed is True
    assert findings.llm_calls[1].accepted is False
    assert findings.llm_calls[1].blocking_quality_findings >= 1
    assert blocking_quality_findings(findings.items)
    assert any(item.code is FindingCode.QUALITY_REPAIR_REJECTED for item in findings.items)
    assert any(
        item.code
        in {FindingCode.DUPLICATE_ACCEPTANCE_CRITERION, FindingCode.OVERLAPPING_STORY_RESPONSIBILITY}
        for item in findings.items
    )
    assert backlog.items


def test_unmatched_clause_is_not_discarded() -> None:
    payload = _delivery_envelope(integrations=False)
    fr = payload["requirements"]["functional_requirements"]
    shared = "Review incoming work and track delivery status and archive the closed batch."
    payload["requirements"]["acceptance_criteria"][0] = shared
    stories = [
        _story(
            "story-review",
            "Review Incoming Work",
            "feat-0",
            0,
            fr[0],
            description="A reviewer reviews incoming work.",
            acceptance=[shared],
            ac_paths=[(0, shared)],
        ),
        _story(
            "story-status",
            "Track Delivery Status",
            "feat-1",
            1,
            fr[1],
            description="A reviewer tracks delivery status.",
            acceptance=[shared],
            ac_paths=[(0, shared)],
        ),
    ]
    provider = MockProvider(responses=[_invalid(payload), _tree(payload, stories)])
    backlog, findings = decompose_backlog(CopilotWorkflowResponseV1.model_validate(payload), provider)
    items = _by_title(backlog)

    for title, path in (
        ("Review Incoming Work", "$.requirements.functional_requirements[0]"),
        ("Track Delivery Status", "$.requirements.functional_requirements[1]"),
    ):
        assert shared in items[title].acceptance_criteria
        assert path in {reference.json_path for reference in items[title].provenance}
    assert findings.llm_calls[1].validation_passed is True
    assert findings.llm_calls[1].accepted is False
    assert findings.llm_calls[1].blocking_quality_findings >= 1
    assert any(item.code is FindingCode.DUPLICATE_ACCEPTANCE_CRITERION for item in findings.items)


def test_single_outcome_criterion_stays_verbatim_on_its_owner() -> None:
    payload = _delivery_envelope(integrations=False)
    fr = payload["requirements"]["functional_requirements"]
    shared = "A reviewer can review incoming work."
    payload["requirements"]["acceptance_criteria"][0] = shared
    stories = [
        _story(
            "story-review",
            "Review Incoming Work",
            "feat-0",
            0,
            fr[0],
            description=(
                "A reviewer reviews incoming work and track delivery status. "
                "The review decision is stored with the inspection record."
            ),
            acceptance=[shared],
            ac_paths=[(0, shared)],
        ),
        _story(
            "story-status",
            "Track Delivery Status",
            "feat-1",
            1,
            fr[1],
            description="A reviewer tracks delivery status.",
            acceptance=[shared],
            ac_paths=[(0, shared)],
        ),
    ]
    provider = MockProvider(responses=[_invalid(payload), _tree(payload, stories)])
    backlog, findings = decompose_backlog(CopilotWorkflowResponseV1.model_validate(payload), provider)
    items = _by_title(backlog)
    review = items["Review Incoming Work"]
    status = items["Track Delivery Status"]

    assert review.acceptance_criteria == [shared]
    assert any(reference.excerpt == shared for reference in review.provenance)
    assert "$.requirements.functional_requirements[0]" in {
        reference.json_path for reference in review.provenance
    }
    assert shared not in status.acceptance_criteria
    assert fr[1] in status.acceptance_criteria[0]
    assert all(reference.excerpt != status.acceptance_criteria[0] or reference.excerpt == fr[1] for reference in status.provenance)
    assert "delivery status" not in review.description.casefold()
    assert "inspection record" in review.description.casefold()
    assert "$.requirements.functional_requirements[1]" in {
        reference.json_path for reference in status.provenance
    }
    assert findings.llm_calls[1].accepted is True
    assert findings.llm_calls[1].blocking_quality_findings == 0


def test_nonblocking_warnings_can_still_be_accepted() -> None:
    provider = MockProvider(_proposal(_base_items(split_intake=True)))
    _backlog, findings = decompose_backlog(_envelope(), provider)

    assert findings.llm_calls[0].validation_passed is True
    assert findings.llm_calls[0].accepted is True
    assert findings.llm_calls[0].blocking_quality_findings == 0
    assert any(item.code is FindingCode.ATOMIC_STORY for item in findings.items)
    assert any(item.code is FindingCode.GENERATED_CONTENT for item in findings.items)
    assert not blocking_quality_findings(findings.items)
