from __future__ import annotations

import copy

import pytest

from backlog_agent.backlog.decompose import (
    _build_quality_repair_prompt,
    _public_validation_errors,
    decompose_backlog,
)
from backlog_agent.backlog.models import WorkItemType
from backlog_agent.llm.exceptions import DecompositionValidationError
from backlog_agent.contracts.copilot_workflow_response_v1 import CopilotWorkflowResponseV1
from backlog_agent.findings.models import Finding, FindingCode, FindingSeverity
from backlog_agent.llm.mock import MockProvider
from tests.copilot_envelopes import pickup_notice_envelope
from tests.test_decompose_generic import _item, _src
from tests.test_decompose_granularity import _base_items, _envelope, _proposal
from tests.llm_proposal_fixtures import representative_complete_proposal
from tests.test_decompose_backlog import _envelope as _insurance_envelope


def test_quality_repair_prompt_includes_overlap_diagnostics() -> None:
    finding = Finding(
        severity=FindingSeverity.WARNING,
        code=FindingCode.OVERLAPPING_STORY_RESPONSIBILITY,
        message=(
            "Stories 'Send Hold Notice' (cbl_notice) and 'Place a Hold' (cbl_hold) overlap. "
            "'Place a Hold' includes action 'hold notice' owned by 'Send Hold Notice'. "
            "Evidence in 'Place a Hold': 'queue a hold notice after reservation'. "
            "Remediation: remove 'hold notice' from 'Place a Hold' description and acceptance "
            "criteria; keep that outcome on 'Send Hold Notice' only."
        ),
        canonical_ids=["cbl_hold", "cbl_notice"],
        local_ids=["story-hold", "story-notice"],
    )
    prompt = _build_quality_repair_prompt(
        "Normalized project facts: catalog holds.",
        {"items": [{"local_id": "story-hold"}, {"local_id": "story-notice"}]},
        [finding],
    )
    lowered = prompt.lower()
    assert FindingCode.OVERLAPPING_STORY_RESPONSIBILITY.value in prompt
    assert "cbl_hold" in prompt and "cbl_notice" in prompt
    assert "Send Hold Notice" in prompt
    assert "Place a Hold" in prompt
    assert "hold notice" in prompt
    assert "Remediation:" in prompt
    assert "named pairwise" in lowered
    assert "inventing architecture" in lowered
    assert "non-functional requirements" in lowered
    assert "local_id" in lowered
    assert "story-hold" in prompt and "story-notice" in prompt
    assert '"local_ids"' in prompt
    assert "canonical uuid" in lowered or "cbl_" in lowered
    assert "inventing architecture" in lowered
    assert "duplicate submission" in lowered
    assert "auditability" in lowered
    assert "do not invent api endpoint" in lowered
    assert "merely to increase the count" in lowered


def test_quality_repair_runs_once_when_first_proposal_is_valid() -> None:
    oversized = _proposal(_base_items(split_intake=False))
    split = _proposal(_base_items(split_intake=True))
    provider = MockProvider(responses=[oversized, split])
    backlog, findings = decompose_backlog(_envelope(), provider)

    assert len(provider.calls) == 2
    quality_prompt = provider.calls[1][0]
    assert "actionable quality findings" in quality_prompt
    assert FindingCode.OVERSIZED_STORY.value in quality_prompt
    assert "do not invent architecture" in quality_prompt.lower() or "inventing architecture" in quality_prompt
    assert "failed deterministic validation" not in quality_prompt
    assert not any(item.code is FindingCode.OVERSIZED_STORY for item in findings.items)
    assert any("holiday calendar" in item.title.lower() for item in backlog.items)
    assert [call.kind for call in findings.llm_calls] == ["initial", "quality_repair"]
    assert findings.llm_calls[1].accepted is True


def test_empty_quality_repair_cannot_replace_a_valid_tree() -> None:
    oversized = _proposal(_base_items(split_intake=False))
    empty = {"items": [], "prerequisites": [], "uncertainties": []}
    provider = MockProvider(responses=[oversized, empty])
    backlog, findings = decompose_backlog(_envelope(), provider)
    assert any(item.type.value == "user_story" for item in backlog.items)
    assert findings.llm_calls[0].accepted is True
    assert findings.llm_calls[1].accepted is False
    assert any(item.code is FindingCode.QUALITY_REPAIR_REJECTED for item in findings.items)
    assert any(item.code is FindingCode.OVERSIZED_STORY for item in findings.items)


def test_quality_repair_keeps_first_valid_tree_if_second_fails_validation() -> None:
    oversized = _proposal(_base_items(split_intake=False))
    invalid = copy.deepcopy(oversized)
    invalid["items"] = [item for item in invalid["items"] if item["type"] != "user_story"]
    provider = MockProvider(responses=[oversized, invalid])
    backlog, findings = decompose_backlog(_envelope(), provider)

    assert len(provider.calls) == 2
    assert any(item.type.value == "user_story" for item in backlog.items)
    assert any(item.code is FindingCode.OVERSIZED_STORY for item in findings.items)
    assert [call.kind for call in findings.llm_calls] == ["initial", "quality_repair"]
    assert findings.llm_calls[0].usable_proposal is True
    assert findings.llm_calls[0].validation_passed is True
    assert findings.llm_calls[0].accepted is True
    assert findings.llm_calls[1].usable_proposal is True
    assert findings.llm_calls[1].validation_passed is False
    assert findings.llm_calls[1].accepted is False
    assert findings.llm_calls[1].validation_errors
    assert all("Normalized project facts" not in error for error in findings.llm_calls[1].validation_errors)
    assert any(item.code is FindingCode.QUALITY_REPAIR_REJECTED for item in findings.items)


def test_quality_repair_does_not_fail_when_warnings_remain() -> None:
    oversized = _proposal(_base_items(split_intake=False))
    provider = MockProvider(responses=[copy.deepcopy(oversized), copy.deepcopy(oversized)])
    _backlog, findings = decompose_backlog(_envelope(), provider)

    assert len(provider.calls) == 2
    assert any(item.code is FindingCode.OVERSIZED_STORY for item in findings.items)


_QUALITY_CODES = {
    FindingCode.OVERSIZED_STORY,
    FindingCode.MISSING_IMPLEMENTATION_DETAIL,
    FindingCode.OVERLAPPING_STORY_RESPONSIBILITY,
    FindingCode.DUPLICATE_ACCEPTANCE_CRITERION,
}


def test_atomic_and_generated_findings_do_not_trigger_quality_repair() -> None:
    provider = MockProvider(_proposal(_base_items(split_intake=True)))
    _backlog, findings = decompose_backlog(_envelope(), provider)

    assert len(provider.calls) == 1
    assert any(item.code is FindingCode.ATOMIC_STORY for item in findings.items)
    assert any(item.code is FindingCode.GENERATED_CONTENT for item in findings.items)
    assert not any(item.code in _QUALITY_CODES for item in findings.items)


def test_hard_repair_is_not_followed_by_quality_repair() -> None:
    bad = representative_complete_proposal()
    bad["items"] = [item for item in bad["items"] if item["local_id"] != "story-premium"]
    good = representative_complete_proposal()
    extra = copy.deepcopy(good)
    provider = MockProvider(responses=[bad, good, extra])
    _backlog, _findings = decompose_backlog(_insurance_envelope(), provider)

    assert len(provider.calls) == 2
    assert "failed deterministic validation" not in provider.calls[1][0]
    assert "actionable quality findings" in provider.calls[1][0]
    assert [call.kind for call in _findings.llm_calls] == ["initial", "quality_repair"]
    assert _findings.llm_calls[0].validation_passed is True
    assert _findings.llm_calls[0].accepted is False
    assert _findings.llm_calls[1].validation_passed is True
    assert _findings.llm_calls[1].accepted is True


def test_repair_false_skips_quality_repair() -> None:
    provider = MockProvider(_proposal(_base_items(split_intake=False)))
    _backlog, findings = decompose_backlog(_envelope(), provider, repair=False)
    assert len(provider.calls) == 1
    assert any(item.code is FindingCode.OVERSIZED_STORY for item in findings.items)


def _pickup_envelope():
    return CopilotWorkflowResponseV1.model_validate(pickup_notice_envelope())


def _pickup_proposal(*, overlapping: bool) -> dict:
    payload = pickup_notice_envelope()
    fr = payload["requirements"]["functional_requirements"]
    ac = payload["requirements"]["acceptance_criteria"][0]
    caps = payload["solution"]["key_capabilities"]
    nfr = payload["requirements"]["non_functional_requirements"][0]
    hold_description = (
        "A member places a hold on an available copy so the copy is reserved. "
        "Catalog search p95 latency must stay under 400 milliseconds."
        if not overlapping
        else (
            "A member places a hold on an available copy and trigger a pickup notice "
            "via the notice service so the copy is reserved. "
            "Catalog search p95 latency must stay under 400 milliseconds."
        )
    )
    hold_ac = [ac] if overlapping else ["A member can place a hold on an available copy."]
    hold_ac_origins = ["source"] if overlapping else ["generated"]
    items = [
        _item(
            "epic-1",
            "epic",
            "Member holds and pickup notices",
            parent=None,
            provenance=[_src("$.user_request", "user_request", payload["user_request"])],
            title_origin="generated",
            description="",
        ),
        _item(
            "feat-hold",
            "feature",
            caps[0],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[0]", "key_capabilities", caps[0])],
        ),
        _item(
            "feat-notice",
            "feature",
            caps[1],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[1]", "key_capabilities", caps[1])],
        ),
        _item(
            "story-hold",
            "user_story",
            "Place Hold On Available Copy",
            parent="feat-hold",
            provenance=[
                _src("$.requirements.functional_requirements[0]", "functional_requirements", fr[0]),
                _src("$.requirements.non_functional_requirements[0]", "non_functional_requirements", nfr),
            ],
            title_origin="generated",
            description=hold_description,
            ac=hold_ac,
            ac_origins=hold_ac_origins,
            tests=[
                "Place a hold on an available copy and assert it is reserved.",
                "Assert catalog search p95 latency stays under 400 milliseconds.",
            ],
        ),
        _item(
            "story-notice",
            "user_story",
            "Send Pickup Notice via Notice Service",
            parent="feat-notice",
            provenance=[_src("$.requirements.functional_requirements[1]", "functional_requirements", fr[1])],
            title_origin="generated",
            description=(
                "The notice service sends a pickup notice when a held copy is ready so the "
                "member can collect it."
            ),
            ac=["A pickup notice is sent when a held copy is ready."],
            ac_origins=["generated"],
            tests=["Assert a pickup notice is sent after a held copy becomes ready."],
            depends=["story-hold"],
        ),
    ]
    return {"items": items, "prerequisites": []}


def _merged_pickup_proposal() -> dict:
    payload = pickup_notice_envelope()
    fr = payload["requirements"]["functional_requirements"]
    ac = payload["requirements"]["acceptance_criteria"][0]
    caps = payload["solution"]["key_capabilities"]
    nfr = payload["requirements"]["non_functional_requirements"][0]
    items = [
        _item(
            "epic-1",
            "epic",
            "Member holds and pickup notices",
            parent=None,
            provenance=[_src("$.user_request", "user_request", payload["user_request"])],
            title_origin="generated",
            description="",
        ),
        _item(
            "feat-hold",
            "feature",
            caps[0],
            parent="epic-1",
            provenance=[_src("$.solution.key_capabilities[0]", "key_capabilities", caps[0])],
        ),
        _item(
            "story-merged",
            "user_story",
            "Place Hold And Send Pickup Notice",
            parent="feat-hold",
            provenance=[
                _src("$.requirements.functional_requirements[0]", "functional_requirements", fr[0]),
                _src("$.requirements.functional_requirements[1]", "functional_requirements", fr[1]),
                _src("$.requirements.non_functional_requirements[0]", "non_functional_requirements", nfr),
            ],
            title_origin="generated",
            description=(
                "A member places a hold on an available copy and trigger a pickup notice "
                "via the notice service so the copy is reserved. "
                "Catalog search p95 latency must stay under 400 milliseconds."
            ),
            ac=[ac],
            ac_origins=["source"],
            tests=[
                "Walk hold placement and pickup notice in one scenario.",
                "Assert catalog search p95 latency stays under 400 milliseconds.",
            ],
        ),
    ]
    return {"items": items, "prerequisites": []}


def test_overlap_findings_include_proposal_local_ids() -> None:
    provider = MockProvider(_pickup_proposal(overlapping=True))
    backlog, findings = decompose_backlog(_pickup_envelope(), provider, repair=False)
    hold = next(item for item in backlog.items if item.title == "Place Hold On Available Copy")
    assert "pickup notice" not in " ".join(hold.acceptance_criteria).casefold()
    assert "pickup notice" not in hold.description.casefold()
    assert not any(
        item.code is FindingCode.OVERLAPPING_STORY_RESPONSIBILITY for item in findings.items
    )
    assert not any(
        item.code is FindingCode.DUPLICATE_ACCEPTANCE_CRITERION for item in findings.items
    )


def test_quality_repair_prompt_identifies_proposal_items_from_overlap_finding() -> None:
    overlapping = _pickup_proposal(overlapping=True)
    repaired = _pickup_proposal(overlapping=False)
    provider = MockProvider(responses=[overlapping, repaired])
    _backlog, findings = decompose_backlog(_pickup_envelope(), provider)
    assert not any(
        item.code is FindingCode.OVERLAPPING_STORY_RESPONSIBILITY for item in findings.items
    )
    if len(provider.calls) == 1:
        return
    quality_prompt = provider.calls[1][0]
    assert "actionable quality findings" in quality_prompt
    assert FindingCode.OVERLAPPING_STORY_RESPONSIBILITY.value not in quality_prompt
    assert "never treat a canonical" in quality_prompt.lower()
    assert "Previously covered functional requirements" in quality_prompt
    assert "$.requirements.functional_requirements[0]" in quality_prompt
    assert "story-hold" in quality_prompt


def test_repaired_compound_criterion_clears_hold_notice_overlap() -> None:
    overlapping = _pickup_proposal(overlapping=True)
    repaired = _pickup_proposal(overlapping=False)
    provider = MockProvider(responses=[overlapping, repaired])
    backlog, findings = decompose_backlog(_pickup_envelope(), provider)

    assert len(provider.calls) == 2
    hold = next(item for item in backlog.items if item.title == "Place Hold On Available Copy")
    notice = next(
        item for item in backlog.items if item.title == "Send Pickup Notice via Notice Service"
    )
    assert "pickup notice" not in " ".join(hold.acceptance_criteria).lower()
    assert "pickup notice" not in hold.description.lower()
    hold_proposed = next(item for item in repaired["items"] if item["local_id"] == "story-hold")
    assert hold_proposed["acceptance_criteria_origins"] == ["generated"]
    assert notice is not None
    fr_paths = {
        reference.json_path
        for item in (hold, notice)
        for reference in item.provenance
        if reference.json_path.startswith("$.requirements.functional_requirements")
    }
    assert "$.requirements.functional_requirements[0]" in fr_paths
    assert "$.requirements.functional_requirements[1]" in fr_paths
    assert not any(
        item.code is FindingCode.OVERLAPPING_STORY_RESPONSIBILITY
        and {"story-hold", "story-notice"} <= set(item.local_ids)
        for item in findings.items
    )
    assert not any(item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT for item in findings.items)
    assert not any(item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT for item in findings.items)
    nfr_paths = {
        reference.json_path
        for item in backlog.items
        for reference in item.provenance
        if reference.json_path.startswith("$.requirements.non_functional_requirements")
    }
    assert "$.requirements.non_functional_requirements[0]" in nfr_paths
    assert [call.kind for call in findings.llm_calls] == ["initial", "quality_repair"]
    assert findings.llm_calls[1].accepted is True
    assert findings.llm_calls[1].lost_fr_paths == []
    assert "$.requirements.functional_requirements[0]" in findings.llm_calls[1].covered_fr_paths
    assert "$.requirements.functional_requirements[1]" in findings.llm_calls[1].covered_fr_paths
    assert findings.llm_calls[1].item_counts.get("user_story") == 2


def test_remaining_overlap_findings_describe_the_returned_tree() -> None:
    overlapping = _pickup_proposal(overlapping=True)
    provider = MockProvider(responses=[copy.deepcopy(overlapping), copy.deepcopy(overlapping)])
    backlog, findings = decompose_backlog(_pickup_envelope(), provider)

    hold = next(item for item in backlog.items if item.title == "Place Hold On Available Copy")
    assert "pickup notice" not in " ".join(hold.acceptance_criteria).casefold()
    assert "pickup notice" not in hold.description.casefold()
    assert not any(
        item.code is FindingCode.OVERLAPPING_STORY_RESPONSIBILITY for item in findings.items
    )
    assert findings.llm_calls[-1].accepted is True
    assert findings.llm_calls[-1].blocking_quality_findings == 0


def test_merged_distinct_functional_requirements_are_rejected() -> None:
    provider = MockProvider(_merged_pickup_proposal())
    with pytest.raises(DecompositionValidationError) as exc:
        decompose_backlog(_pickup_envelope(), provider, repair=False)
    assert any("distinct functional requirements" in error for error in exc.value.errors)
    assert [call.kind for call in exc.value.findings.llm_calls] == ["initial"]
    assert exc.value.findings.llm_calls[0].validation_passed is False
    assert exc.value.findings.llm_calls[0].accepted is False


def test_hard_repair_splits_merged_functional_requirements() -> None:
    merged = _merged_pickup_proposal()
    split = _pickup_proposal(overlapping=False)
    provider = MockProvider(responses=[merged, split])
    backlog, findings = decompose_backlog(_pickup_envelope(), provider)

    assert len(provider.calls) == 2
    assert "failed deterministic validation" in provider.calls[1][0]
    stories = [item for item in backlog.items if item.type.value == "user_story"]
    assert len(stories) == 2
    fr_paths = {
        reference.json_path
        for item in stories
        for reference in item.provenance
        if reference.json_path.startswith("$.requirements.functional_requirements")
    }
    assert fr_paths == {
        "$.requirements.functional_requirements[0]",
        "$.requirements.functional_requirements[1]",
    }
    assert all(
        len(
            [
                reference
                for reference in item.provenance
                if reference.json_path.startswith("$.requirements.functional_requirements")
            ]
        )
        == 1
        for item in stories
    )
    hold = next(item for item in stories if item.title == "Place Hold On Available Copy")
    hold_proposed = next(item for item in split["items"] if item["local_id"] == "story-hold")
    assert hold_proposed["acceptance_criteria_origins"] == ["generated"]
    assert "pickup notice" not in " ".join(hold.acceptance_criteria).lower()
    assert not any(item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT for item in findings.items)
    assert not any(item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT for item in findings.items)
    assert [call.kind for call in findings.llm_calls] == ["initial", "hard_repair"]
    assert findings.llm_calls[0].accepted is False
    assert findings.llm_calls[1].accepted is True


def test_quality_repair_scoped_generated_ac_keeps_source_mislabeled_origin() -> None:
    overlapping = _pickup_proposal(overlapping=True)
    repaired = _pickup_proposal(overlapping=False)
    hold = next(item for item in repaired["items"] if item["local_id"] == "story-hold")
    hold["acceptance_criteria_origins"] = ["source"]
    provider = MockProvider(responses=[overlapping, repaired])
    backlog, findings = decompose_backlog(_pickup_envelope(), provider)

    assert len(provider.calls) == 2
    assert findings.llm_calls[1].kind == "quality_repair"
    assert findings.llm_calls[1].accepted is True
    assert findings.llm_calls[1].validation_errors == []
    hold_item = next(item for item in backlog.items if item.title == "Place Hold On Available Copy")
    notice_item = next(
        item for item in backlog.items if item.title == "Send Pickup Notice via Notice Service"
    )
    assert hold_item.acceptance_criteria == ["A member can place a hold on an available copy."]
    assert "pickup notice" not in " ".join(hold_item.acceptance_criteria).lower()
    generated = [
        item
        for item in findings.items
        if item.code is FindingCode.GENERATED_CONTENT and hold_item.canonical_id in item.canonical_ids
    ]
    assert generated
    assert "acceptance_criteria" in generated[0].message
    fr_paths = {
        reference.json_path
        for item in (hold_item, notice_item)
        for reference in item.provenance
        if reference.json_path.startswith("$.requirements.functional_requirements")
    }
    assert fr_paths == {
        "$.requirements.functional_requirements[0]",
        "$.requirements.functional_requirements[1]",
    }


def test_uncovered_nfr_remains_visible_and_can_trigger_quality_repair() -> None:
    overlapping = _pickup_proposal(overlapping=False)
    hold = next(item for item in overlapping["items"] if item["local_id"] == "story-hold")
    hold["provenance"] = [
        reference
        for reference in hold["provenance"]
        if not reference["json_path"].startswith("$.requirements.non_functional_requirements")
    ]
    hold["description"] = hold["description"].replace(
        " Catalog search p95 latency must stay under 400 milliseconds.",
        "",
    )
    hold["test_requirements"] = [
        text for text in hold["test_requirements"] if "latency" not in text
    ]
    hold["test_requirements_origins"] = ["generated"] * len(hold["test_requirements"])
    repaired = _pickup_proposal(overlapping=False)
    provider = MockProvider(responses=[copy.deepcopy(overlapping), repaired])
    _backlog, findings = decompose_backlog(_pickup_envelope(), provider)

    assert len(provider.calls) == 2
    assert FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT.value in provider.calls[1][0]
    assert [call.kind for call in findings.llm_calls] == ["initial", "quality_repair"]
    assert findings.llm_calls[1].accepted is True
    assert not any(item.code is FindingCode.UNCOVERED_NON_FUNCTIONAL_REQUIREMENT for item in findings.items)


def test_quality_repair_dropping_covered_fr_keeps_initial_tree() -> None:
    overlapping = _pickup_proposal(overlapping=True)
    dropped = _pickup_proposal(overlapping=False)
    dropped["items"] = [
        item for item in dropped["items"] if item["local_id"] not in {"story-notice", "feat-notice"}
    ]
    provider = MockProvider(responses=[overlapping, dropped])
    backlog, findings = decompose_backlog(_pickup_envelope(), provider)

    assert len(provider.calls) == 2
    assert findings.llm_calls[0].accepted is True
    assert findings.llm_calls[1].kind == "quality_repair"
    assert findings.llm_calls[1].accepted is False
    assert findings.llm_calls[1].validation_passed is True
    assert findings.llm_calls[1].usable_proposal is True
    assert "$.requirements.functional_requirements[1]" in findings.llm_calls[1].lost_fr_paths
    assert any(
        reference.json_path == "$.requirements.functional_requirements[1]"
        for item in backlog.items
        if item.type is WorkItemType.USER_STORY
        for reference in item.provenance
    )
    assert all(
        "Normalized project facts" not in error for error in findings.llm_calls[1].validation_errors
    )
    titles = {item.title for item in backlog.items}
    assert "Send Pickup Notice via Notice Service" in titles
    assert "Place Hold On Available Copy" in titles
    quality_prompt = provider.calls[1][0]
    assert "must not drop a previously covered path" in quality_prompt
    assert "$.requirements.functional_requirements[1]" in quality_prompt


def test_public_validation_errors_redact_secrets_and_bound_length() -> None:
    long_secret = "api_key=AIzaSyDummyValue12 " + ("payload " * 80)
    cleaned = _public_validation_errors(
        [
            long_secret,
            "Normalized project facts: {\"request\": {\"text\": \"hidden\"}}",
            "Uncovered functional requirement: place a hold",
        ]
    )
    assert len(cleaned) == 3
    assert "[redacted]" in cleaned[0]
    assert "AIzaSyDummyValue12" not in cleaned[0]
    assert len(cleaned[0]) <= 300
    assert "omitted" in cleaned[1]
    assert "hidden" not in cleaned[1]
    assert "place a hold" in cleaned[2]


def test_hard_repair_is_the_only_repair_and_keeps_remaining_warnings() -> None:
    merged = _merged_pickup_proposal()
    repaired = _pickup_proposal(overlapping=True)
    shared = pickup_notice_envelope()["requirements"]["acceptance_criteria"][0]
    for local_id in ("story-hold", "story-notice"):
        story = next(item for item in repaired["items"] if item["local_id"] == local_id)
        story["acceptance_criteria"] = [shared]
        story["acceptance_criteria_origins"] = ["source"]
        story["provenance"].append(
            _src("$.requirements.acceptance_criteria[0]", "acceptance_criteria", shared)
        )
    provider = MockProvider(responses=[merged, repaired])
    backlog, findings = decompose_backlog(_pickup_envelope(), provider)

    assert len(provider.calls) == 2
    assert "failed deterministic validation" in provider.calls[1][0]
    assert "only repair" in provider.calls[1][0].lower()
    assert [call.kind for call in findings.llm_calls] == ["initial", "hard_repair"]
    assert findings.llm_calls[0].accepted is False
    assert findings.llm_calls[1].accepted is True
    assert findings.llm_calls[1].validation_passed is True
    assert findings.llm_calls[1].blocking_quality_findings == 0
    assert not any(item.code is FindingCode.DUPLICATE_ACCEPTANCE_CRITERION for item in findings.items)
    assert any(
        item.code is FindingCode.GENERATED_CONTENT or item.code is FindingCode.MISSING_IMPLEMENTATION_DETAIL
        for item in findings.items
    )
    assert not any(item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT for item in findings.items)
    assert any(item.title == "Place Hold On Available Copy" for item in backlog.items)
