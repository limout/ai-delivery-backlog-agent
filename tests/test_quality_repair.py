from __future__ import annotations

import copy

from backlog_agent.backlog.decompose import _build_quality_repair_prompt, decompose_backlog
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


def test_quality_repair_keeps_first_valid_tree_if_second_fails_validation() -> None:
    oversized = _proposal(_base_items(split_intake=False))
    invalid = copy.deepcopy(oversized)
    invalid["items"] = [item for item in invalid["items"] if item["type"] != "user_story"]
    provider = MockProvider(responses=[oversized, invalid])
    backlog, findings = decompose_backlog(_envelope(), provider)

    assert len(provider.calls) == 2
    assert any(item.type.value == "user_story" for item in backlog.items)
    assert any(item.code is FindingCode.OVERSIZED_STORY for item in findings.items)


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
    assert "failed deterministic validation" in provider.calls[1][0]
    assert "actionable quality findings" not in provider.calls[1][0]


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
        "A member places a hold on an available copy so the copy is reserved."
        if not overlapping
        else (
            "A member places a hold on an available copy and trigger a pickup notice "
            "via the notice service so the copy is reserved."
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
            tests=["Place a hold on an available copy and assert it is reserved."],
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


def test_overlap_findings_include_proposal_local_ids() -> None:
    provider = MockProvider(_pickup_proposal(overlapping=True))
    _backlog, findings = decompose_backlog(_pickup_envelope(), provider, repair=False)
    overlap = [
        item for item in findings.items if item.code is FindingCode.OVERLAPPING_STORY_RESPONSIBILITY
    ]
    assert overlap
    assert set(overlap[0].local_ids) == {"story-hold", "story-notice"}
    assert len(overlap[0].canonical_ids) == 2
    assert all(canonical_id.startswith("cbl_") for canonical_id in overlap[0].canonical_ids)
    message = overlap[0].message
    assert "local_id=story-hold" in message
    assert "local_id=story-notice" in message
    assert "Place Hold On Available Copy" in message
    assert "Send Pickup Notice via Notice Service" in message
    assert "pickup notice" in message
    assert "Remediation:" in message


def test_quality_repair_prompt_identifies_proposal_items_from_overlap_finding() -> None:
    overlapping = _pickup_proposal(overlapping=True)
    repaired = _pickup_proposal(overlapping=False)
    provider = MockProvider(responses=[overlapping, repaired])
    decompose_backlog(_pickup_envelope(), provider)
    assert len(provider.calls) == 2
    quality_prompt = provider.calls[1][0]
    assert FindingCode.OVERLAPPING_STORY_RESPONSIBILITY.value in quality_prompt
    assert "local_id=story-hold" in quality_prompt
    assert "local_id=story-notice" in quality_prompt
    assert '"local_ids"' in quality_prompt
    assert "story-hold" in quality_prompt
    assert "never treat a canonical" in quality_prompt.lower()


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


def test_remaining_overlap_findings_describe_the_returned_tree() -> None:
    overlapping = _pickup_proposal(overlapping=True)
    provider = MockProvider(responses=[copy.deepcopy(overlapping), copy.deepcopy(overlapping)])
    backlog, findings = decompose_backlog(_pickup_envelope(), provider)

    hold = next(item for item in backlog.items if item.title == "Place Hold On Available Copy")
    overlap = [
        item for item in findings.items if item.code is FindingCode.OVERLAPPING_STORY_RESPONSIBILITY
    ]
    assert overlap
    assert set(overlap[0].local_ids) == {"story-hold", "story-notice"}
    assert hold.canonical_id in overlap[0].canonical_ids
    assert "pickup notice" in " ".join(hold.acceptance_criteria).lower()
    assert "local_id=story-hold" in overlap[0].message
