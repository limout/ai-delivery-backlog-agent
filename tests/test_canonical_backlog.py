from __future__ import annotations

import pytest
from pydantic import ValidationError

from backlog_agent.backlog.ids import new_canonical_id
from backlog_agent.backlog.models import (
    ApprovalState,
    CanonicalBacklogV1,
    Effort,
    Epic,
    Feature,
    Priority,
    SourceReference,
    Subtask,
    Task,
    UserStory,
    WorkItemStatus,
    WorkItemType,
    _reject_cycles,
)
from backlog_agent.contracts.versions import (
    CANONICAL_BACKLOG_V1,
    COPILOT_WORKFLOW_RESPONSE_V1,
)


def _linked_tree() -> tuple[Epic, Feature, UserStory, Task, Subtask]:
    epic = Epic(title="Customer portal")
    feature = Feature(title="Policy renewal", parent_id=epic.canonical_id)
    story = UserStory(
        title="Renew an existing policy",
        parent_id=feature.canonical_id,
        acceptance_criteria=["Confirmation is stored after a successful renewal."],
        test_requirements=["Cover successful renewal and expired-policy rejection."],
        provenance=[
            SourceReference(
                contract_id=COPILOT_WORKFLOW_RESPONSE_V1,
                json_path="$.requirements.functional_requirements[0]",
                field_name="functional_requirements",
                excerpt="The system shall allow a customer to renew an existing policy.",
            )
        ],
    )
    task = Task(title="Implement renewal API", parent_id=story.canonical_id)
    subtask = Subtask(title="Add request validation", parent_id=task.canonical_id)

    epic.child_ids = [feature.canonical_id]
    feature.child_ids = [story.canonical_id]
    story.child_ids = [task.canonical_id]
    task.child_ids = [subtask.canonical_id]
    return epic, feature, story, task, subtask


def test_generated_ids_are_canonical_and_unique() -> None:
    first = new_canonical_id()
    second = new_canonical_id()

    assert first != second
    assert first.startswith("cbl_")
    assert second.startswith("cbl_")
    assert first != "PORTAL-1"
    assert not first.isdigit()


def test_work_item_gets_canonical_id_on_create() -> None:
    epic = Epic(title="Customer portal")

    assert epic.canonical_id.startswith("cbl_")
    assert epic.type is WorkItemType.EPIC
    assert epic.status is WorkItemStatus.TODO
    assert epic.approval_state is ApprovalState.NOT_APPROVED
    assert epic.effort is None
    assert epic.parent_id is None


def test_canonical_backlog_accepts_full_hierarchy() -> None:
    epic, feature, story, task, subtask = _linked_tree()

    backlog = CanonicalBacklogV1(
        items=[epic, feature, story, task, subtask],
        source_contract_id=COPILOT_WORKFLOW_RESPONSE_V1,
    )

    assert backlog.contract_id == CANONICAL_BACKLOG_V1
    assert backlog.backlog_id.startswith("cbl_")
    assert [item.type for item in backlog.items] == [
        WorkItemType.EPIC,
        WorkItemType.FEATURE,
        WorkItemType.USER_STORY,
        WorkItemType.TASK,
        WorkItemType.SUBTASK,
    ]
    assert story.acceptance_criteria
    assert story.test_requirements
    assert story.provenance[0].contract_id == COPILOT_WORKFLOW_RESPONSE_V1


def test_effort_is_optional_and_only_when_explicit() -> None:
    with pytest.raises(ValidationError, match="explicit source-provided value"):
        Effort()

    epic, feature, story, task, subtask = _linked_tree()
    story.effort = Effort(story_points=5, textual_estimate="5 points from source")
    story.priority = Priority.HIGH

    backlog = CanonicalBacklogV1(items=[epic, feature, story, task, subtask])
    stored = backlog.item_map()[story.canonical_id]

    assert stored.effort is not None
    assert stored.effort.story_points == 5
    assert stored.priority is Priority.HIGH
    assert epic.effort is None


def test_allows_unparented_feature() -> None:
    epic = Epic(title="Customer portal")
    feature = Feature(title="Policy renewal")

    backlog = CanonicalBacklogV1(items=[epic, feature])

    assert feature.parent_id is None
    assert backlog.item_map()[feature.canonical_id].parent_id is None


def test_rejects_invalid_parent_reference() -> None:
    epic = Epic(title="Customer portal")
    feature = Feature(title="Policy renewal", parent_id=new_canonical_id())

    with pytest.raises(ValidationError, match="does not exist in the backlog"):
        CanonicalBacklogV1(items=[epic, feature])


def test_rejects_wrong_parent_type() -> None:
    epic = Epic(title="Customer portal")
    story = UserStory(title="Renew policy", parent_id=epic.canonical_id)
    epic.child_ids = [story.canonical_id]

    with pytest.raises(ValidationError, match="parent must be a feature"):
        CanonicalBacklogV1(items=[epic, story])


def test_rejects_tracker_style_ids() -> None:
    with pytest.raises(ValidationError, match="prefix 'cbl_'"):
        Epic(canonical_id="PORTAL-1", title="Customer portal")


def test_rejects_epic_with_parent() -> None:
    with pytest.raises(ValidationError, match="epics cannot have a parent"):
        Epic(title="Customer portal", parent_id=new_canonical_id())


def test_rejects_subtask_with_children() -> None:
    with pytest.raises(ValidationError, match="subtasks cannot have children"):
        Subtask(title="Add validation", child_ids=[new_canonical_id()])


def test_rejects_missing_dependency() -> None:
    epic, feature, story, task, subtask = _linked_tree()
    task.dependencies = [new_canonical_id()]

    with pytest.raises(ValidationError, match="does not exist"):
        CanonicalBacklogV1(items=[epic, feature, story, task, subtask])


def test_rejects_self_dependency() -> None:
    epic, feature, story, task, subtask = _linked_tree()
    task.dependencies = [task.canonical_id]

    with pytest.raises(ValidationError, match="cannot depend on itself"):
        CanonicalBacklogV1(items=[epic, feature, story, task, subtask])


def test_rejects_dependency_cycle() -> None:
    epic, feature, story, task, subtask = _linked_tree()
    extra = Task(title="Write API client", parent_id=story.canonical_id)
    story.child_ids.append(extra.canonical_id)
    task.dependencies = [extra.canonical_id]
    extra.dependencies = [task.canonical_id]

    with pytest.raises(ValidationError, match="dependency cycle"):
        CanonicalBacklogV1(items=[epic, feature, story, task, extra, subtask])


def test_rejects_hierarchy_cycle() -> None:
    class Node:
        def __init__(self, child_ids: list[str]) -> None:
            self.child_ids = child_ids

    with pytest.raises(ValueError, match="parent-child cycle"):
        _reject_cycles({"a": Node(["b"]), "b": Node(["a"])})


def test_round_trips_discriminated_items() -> None:
    epic, feature, story, task, subtask = _linked_tree()
    backlog = CanonicalBacklogV1(items=[epic, feature, story, task, subtask])

    restored = CanonicalBacklogV1.model_validate(backlog.model_dump())

    assert restored.model_dump() == backlog.model_dump()
    assert isinstance(restored.items[0], Epic)
    assert isinstance(restored.items[2], UserStory)
