"""Unresolved source questions must survive as unique, correctly parented clarification tasks.

Also covers the advisory broad-feature signal and outcome-level verification wording. All
domains here (greenhouses, a ledger) are invented test data; production code must not mention
them.
"""

from __future__ import annotations

from types import SimpleNamespace

from backlog_agent.backlog.decompose import decompose_backlog
from backlog_agent.backlog.granularity import (
    _record_feature_child_alignment,
    record_granularity_findings,
)
from backlog_agent.backlog.models import WorkItemType
from backlog_agent.backlog.quality_gate import is_blocking_quality_finding
from backlog_agent.findings.models import FindingCode
from backlog_agent.llm.mock import MockProvider
from tests.test_material_hierarchy import (
    _QUESTION,
    _catch_all,
    _clarify,
    _env,
    _grouped,
    _payload,
    _run,
)
from tests.test_decompose_generic import _src
from tests.test_story_overlap import _ref, _Sink, _story

_HARVEST_QUESTION = "Which harvest summary file format is required?"
_INVITE_QUESTION = "How long should an invitation to a greenhouse remain valid?"

# -- builders for the broad-feature signal (plain objects, no LLM involved) -----------------

_BROAD_STORIES = [
    ("View Assigned Visits", "Allows a field surveyor to view assigned visits with route and location."),
    ("Open Property Records", "Allows a field surveyor to open property records before a visit."),
    ("Complete Survey Checklists", "Allows a field surveyor to complete the checklist step by step on a visit."),
    ("Record Structural Defects", "Allows a field surveyor to document cracks and damaged beams found on a visit."),
    ("Attach Site Photographs", "Allows a field surveyor to attach photographs to findings on a visit."),
    ("Submit Survey Report", "Allows a field surveyor to submit the finished survey report for sign-off."),
    ("Create Repair Orders", "Allows a field surveyor to create follow-up repair orders for identified defects."),
]
_SAME_ACTION_STORIES = [
    ("Search Visits By Date", "Allows a field surveyor to search assigned visits using a date range."),
    ("Search Property Records By Owner", "Allows a field surveyor to search property records using an owner name."),
    ("Search Survey Reports By Status", "Allows a field surveyor to search survey reports using their review status."),
    ("Search Repair Orders By Contractor", "Allows a field surveyor to search repair orders using a contractor name."),
    ("Search Photographs By Caption", "Allows a field surveyor to search site photographs using caption text."),
]


def broad_feature_items(*, same_action: bool = False, feature_title: str = "Survey Execution Management"):
    stories = _SAME_ACTION_STORIES if same_action else _BROAD_STORIES
    ids = [f"cbl_story_{index}" for index in range(len(stories))]
    feature = SimpleNamespace(
        canonical_id="cbl_feature",
        type=WorkItemType.FEATURE,
        title=feature_title,
        description="",
        parent_id="cbl_epic",
        child_ids=tuple(ids),
        acceptance_criteria=[],
        test_requirements=[],
        provenance=[
            _ref("$.solution.key_capabilities[0]", feature_title)
        ],
    )
    children = [
        SimpleNamespace(
            canonical_id=story_id,
            type=WorkItemType.USER_STORY,
            title=title,
            description=description,
            parent_id="cbl_feature",
            child_ids=(),
            acceptance_criteria=[],
            test_requirements=[],
            provenance=[
                _ref(f"$.requirements.functional_requirements[{index}]", title)
            ],
        )
        for index, (story_id, (title, description)) in enumerate(zip(ids, stories))
    ]
    return [feature, *children]


def _broad(findings) -> list:
    return [
        item
        for item in findings
        if item.code is FindingCode.FEATURE_CHILD_MISALIGNMENT
        and "separate capability clusters" in item.message
    ]


def run_alignment(items) -> list:
    sink = _Sink()
    _record_feature_child_alignment(None, items, sink)
    return sink.items


def _clarifications(backlog) -> list:
    return [
        item
        for item in backlog.items
        if item.type is WorkItemType.TASK
        and any(
            marker in ref.json_path for ref in item.provenance for marker in ("open_question", "unknown")
        )
    ]


def _parent_title(backlog, task) -> str:
    return next(item.title for item in backlog.items if item.canonical_id == task.parent_id)


def _paths(task) -> set[str]:
    return {ref.json_path for ref in task.provenance}


# -- unresolved questions are restored, once, under the story they constrain -----------------


def test_dropped_questions_are_restored_once_under_the_story_they_constrain() -> None:
    questions = [_QUESTION, _HARVEST_QUESTION, _INVITE_QUESTION]
    payload = _payload(questions=questions)
    backlog, findings, provider = _run(payload, _grouped(payload, tasks=[]), repair=False)

    assert len(provider.calls) == 1
    tasks = _clarifications(backlog)
    assert len(tasks) == 3
    by_question = {
        next(iter(_paths(task) & {f"$.requirements.open_questions[{i}]" for i in range(3)})): task
        for task in tasks
    }
    assert "moisture" in _parent_title(backlog, by_question["$.requirements.open_questions[0]"]).lower()
    assert "harvest" in _parent_title(backlog, by_question["$.requirements.open_questions[1]"]).lower()
    assert "co-worker" in _parent_title(backlog, by_question["$.requirements.open_questions[2]"]).lower()
    # Nothing is invented: the task says the answer is unknown and the cited text is the source's.
    for task in tasks:
        assert "do not assume" in task.description.lower()
        assert all(ref.excerpt for ref in task.provenance)


def test_questions_dropped_by_quality_repair_are_restored_in_the_returned_tree() -> None:
    payload = _payload(questions=[_QUESTION, _HARVEST_QUESTION])
    initial = _catch_all(
        payload,
        tasks=[
            _clarify("task-a", "story-1", 0, _QUESTION),
            _clarify("task-b", "story-2", 1, _HARVEST_QUESTION),
        ],
    )
    repaired = _grouped(payload, tasks=[])  # the repair fixes the hierarchy but drops both tasks
    backlog, findings, provider = _run(payload, initial, repaired)

    assert [call.kind for call in findings.llm_calls] == ["initial", "quality_repair"]
    assert len(provider.calls) == 2  # still exactly one repair
    tasks = _clarifications(backlog)
    assert len(tasks) == 2
    assert {_parent_title(backlog, task) for task in tasks} == {
        "The system shall let an agronomist view soil moisture readings for a zone.",
        "The system shall let an owner export a harvest summary.",
    }
    # The returned (final) tree was inspected, not only the repair candidate.
    assert findings.llm_calls[-1].accepted is True


def test_a_question_that_a_story_already_carries_is_not_duplicated() -> None:
    payload = _payload(questions=[_QUESTION])
    backlog, _findings, _provider = _run(payload, _grouped(payload), repair=False)
    assert len(_clarifications(backlog)) == 1


def test_question_without_a_clearly_owning_story_is_reported_not_misparented() -> None:
    unrelated = "Which backup retention window applies to archived invoice statements?"
    payload = _payload(questions=[_QUESTION, unrelated])
    backlog, findings, _provider = _run(
        payload,
        _grouped(payload, tasks=[_clarify("task-calibration", "story-1", 0, _QUESTION)]),
        repair=False,
    )
    assert len(_clarifications(backlog)) == 1
    assert _project_level(findings, "backup retention")


def _project_level(findings, phrase: str) -> list:
    return [
        item
        for item in findings.items
        if item.code is FindingCode.UNRESOLVED_PREREQUISITE and phrase in item.message
    ]


def _epic(backlog):
    return next(item for item in backlog.items if item.type is WorkItemType.EPIC)


# -- project-level questions: kept without inventing or forcing a parent ----------------------


def test_unowned_question_is_kept_at_project_level_with_every_source_path() -> None:
    payload = _payload(questions=["Which backup retention window applies to archived invoice statements?"])
    payload["discovery"] = {"unknowns": ["Backup retention window"]}
    payload["delivery_plan"]["dependencies"] = ["Decision on the invoice statement backup retention window"]
    backlog, findings, _provider = _run(payload, _grouped(payload, tasks=[]), repair=False)

    # No unrelated story received a task, and no story was invented to hold one.
    assert not [item for item in backlog.items if item.type is WorkItemType.TASK]
    assert len([i for i in backlog.items if i.type is WorkItemType.USER_STORY]) == 4
    kept = _project_level(findings, "backup retention")
    assert len(kept) == 1  # one question, however many sections repeat it
    expected = {
        "$.requirements.open_questions[0]",
        "$.discovery.unknowns[0]",
        "$.delivery_plan.dependencies[0]",
    }
    assert {ref.json_path for ref in kept[0].source_references} == expected
    # The returned backlog itself keeps the paths, on the epic that spans the whole project.
    assert expected <= {ref.json_path for ref in _epic(backlog).provenance}
    assert all(ref.excerpt for ref in _epic(backlog).provenance)


def test_question_spanning_several_stories_is_not_forced_under_one_of_them() -> None:
    shared = "Which harvest greenhouse policy applies?"
    payload = _payload(questions=[shared])
    backlog, findings, _provider = _run(payload, _grouped(payload, tasks=[]), repair=False)

    assert _clarifications(backlog) == []  # the harvest and greenhouse stories match equally
    assert len(_project_level(findings, "harvest greenhouse")) == 1
    assert "$.requirements.open_questions[0]" in {ref.json_path for ref in _epic(backlog).provenance}


def test_a_model_supplied_prerequisite_for_the_question_is_not_duplicated() -> None:
    question = "Which backup retention window applies to archived invoice statements?"
    payload = _payload(questions=[question])
    proposal = _grouped(payload, tasks=[])
    proposal["prerequisites"] = [
        {
            "text": question,
            "status": "unresolved",
            "provenance": [_src("$.requirements.open_questions[0]", "open_questions", question)],
        }
    ]
    _backlog, findings, _provider = _run(payload, proposal, repair=False)
    assert len(_project_level(findings, "backup retention")) == 1


def test_a_question_with_a_single_clear_owner_still_gets_a_task_not_a_project_level_entry() -> None:
    payload = _payload(questions=[_HARVEST_QUESTION])
    backlog, findings, _provider = _run(payload, _grouped(payload, tasks=[]), repair=False)
    assert len(_clarifications(backlog)) == 1
    assert not _project_level(findings, "harvest summary file format")
    assert "$.requirements.open_questions[0]" not in {ref.json_path for ref in _epic(backlog).provenance}


# -- the same question from several sources gives one task ----------------------------------


def test_the_same_question_in_several_sections_produces_a_single_task_citing_all() -> None:
    payload = _payload(questions=[_QUESTION])
    payload["discovery"] = {"unknowns": ["Sensor calibration interval"]}
    payload["delivery_plan"]["dependencies"] = ["Confirmation of the sensor calibration interval"]
    backlog, findings, _provider = _run(payload, _grouped(payload, tasks=[]), repair=False)

    tasks = _clarifications(backlog)
    assert len(tasks) == 1
    assert _paths(tasks[0]) == {
        "$.requirements.open_questions[0]",
        "$.discovery.unknowns[0]",
        "$.delivery_plan.dependencies[0]",
    }
    assert not any(
        item.code is FindingCode.AMBIGUOUS_DECOMPOSITION and "Overlapping investigation" in item.message
        for item in findings.items
    )


def test_different_questions_sharing_a_phrase_stay_separate_tasks() -> None:
    first = "Which sensor calibration interval is required for the first release?"
    second = "Which harvest summary file format is required for the first release?"
    payload = _payload(questions=[first, second])
    backlog, _findings, _provider = _run(payload, _grouped(payload, tasks=[]), repair=False)
    assert len(_clarifications(backlog)) == 2


# -- facts the source already confirms do not become investigation tasks --------------------


def test_confirmed_prerequisites_do_not_generate_clarification_tasks() -> None:
    payload = _payload(questions=[])
    payload["delivery_plan"]["dependencies"] = [
        "The existing weather service API is available and documented.",
        "Access to the sensor gateway is confirmed.",
    ]
    backlog, findings, _provider = _run(payload, _grouped(payload, tasks=[]), repair=False)
    assert _clarifications(backlog) == []
    assert not [item for item in backlog.items if item.type is WorkItemType.TASK]
    assert not any(
        item.code is FindingCode.UNRESOLVED_PREREQUISITE and "weather service" in item.message
        for item in findings.items
    )


def test_a_dependency_alone_never_creates_a_question_task() -> None:
    payload = _payload(questions=[])
    payload["delivery_plan"]["dependencies"] = ["Resolution of the sensor calibration interval"]
    backlog, _findings, _provider = _run(payload, _grouped(payload, tasks=[]), repair=False)
    assert _clarifications(backlog) == []


# -- over-broad features: reported when the evidence is material, advisory when weak --------


def test_broad_feature_hidden_by_shared_context_words_is_reported_but_never_blocking() -> None:
    items = broad_feature_items()
    findings = run_alignment(items)
    broad = [item for item in findings if item.code is FindingCode.FEATURE_CHILD_MISALIGNMENT]
    assert len(broad) == 1
    assert "separate capability clusters" in broad[0].message
    assert "advisory" in broad[0].message
    assert not is_blocking_quality_finding(broad[0])
    # Names the feature and every story so a reviewer can act on it.
    assert len(broad[0].canonical_ids) == 1 + len([i for i in items if i.type is WorkItemType.USER_STORY])


def test_sentence_style_titles_do_not_hide_a_broad_feature() -> None:
    items = broad_feature_items()
    for item in items[1:]:
        item.title = f"The system shall let a surveyor {item.title.lower()}"
    assert _broad(run_alignment(items))


def test_variants_of_one_action_are_not_reported_as_a_broad_feature() -> None:
    items = broad_feature_items(same_action=True)
    assert not _broad(run_alignment(items))


def test_a_feature_whose_title_names_its_stories_is_not_reported() -> None:
    items = broad_feature_items(
        feature_title=(
            "View assigned visits, open property records, complete survey checklists, record "
            "structural defects, attach site photographs, submit survey report, create repair orders"
        )
    )
    assert not _broad(run_alignment(items))


def test_clearly_unrelated_stories_are_material_while_a_weak_signal_is_not() -> None:
    from tests.test_material_hierarchy import _material, _weak_grouping

    payload = _payload()
    _backlog, findings, _provider = _run(payload, _catch_all(payload), repair=False)
    assert _material(findings, FindingCode.FEATURE_CHILD_MISALIGNMENT)

    weak_payload = _payload()
    _backlog, weak_findings, _provider = _run(weak_payload, _weak_grouping(weak_payload), repair=False)
    assert not _material(weak_findings)


# -- every distinct outcome of a source requirement is verified -----------------------------

_REQUIREMENT = "The system shall let a clerk open the account summary and past payment receipts from the ledger."


def _verification_findings(acceptance, tests, *, description=None):
    story = _story(
        "cbl_story",
        "Open Account Summary and Receipts",
        description=description
        or "As a clerk, I want to open the account summary and past payment receipts from the ledger.",
        acceptance_criteria=acceptance,
        provenance=[_ref("$.requirements.functional_requirements[0]", _REQUIREMENT)],
    )
    story.test_requirements = list(tests)
    sink = _Sink()
    record_granularity_findings(
        SimpleNamespace(functional_requirements=(), non_functional_requirements=()),
        SimpleNamespace(items=[story]),
        sink,
    )
    return [item for item in sink.items if item.code is FindingCode.MISSING_VERIFICATION_COVERAGE]


def test_related_data_does_not_verify_access_to_a_different_named_outcome() -> None:
    findings = _verification_findings(
        ["A clerk can view the account summary retrieved from the ledger.", "A clerk can view historical ledger data."],
        ["Test the account summary rendering against the ledger."],
    )
    assert len(findings) == 1
    assert is_blocking_quality_finding(findings[0])
    # The missing outcome is quoted in the source's wording, not as a title fragment.
    assert "past payment receipts" in findings[0].message
    assert "source wording" in findings[0].message


def test_each_named_outcome_with_its_own_criterion_is_verified() -> None:
    assert not _verification_findings(
        [
            "A clerk can view the account summary retrieved from the ledger.",
            "A clerk can open past payment receipts from the ledger.",
        ],
        [
            "Test the account summary rendering against the ledger.",
            "Test that past payment receipts open from the ledger.",
        ],
    )
