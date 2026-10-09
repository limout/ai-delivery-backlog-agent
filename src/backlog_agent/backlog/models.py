"""Versioned tracker-independent canonical backlog models."""



from __future__ import annotations



from enum import Enum

from typing import Annotated, Any, Literal, Union



from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator



from backlog_agent.backlog.ids import new_canonical_id

from backlog_agent.contracts.versions import CANONICAL_BACKLOG_V1





class WorkItemType(str, Enum):

    EPIC = "epic"

    FEATURE = "feature"

    USER_STORY = "user_story"

    TASK = "task"

    SUBTASK = "subtask"





class WorkItemStatus(str, Enum):

    TODO = "todo"

    IN_PROGRESS = "in_progress"

    DONE = "done"

    BLOCKED = "blocked"

    CANCELLED = "cancelled"





class ApprovalState(str, Enum):

    NOT_APPROVED = "not_approved"

    APPROVED = "approved"





class Priority(str, Enum):

    CRITICAL = "critical"

    HIGH = "high"

    MEDIUM = "medium"

    LOW = "low"





_PARENT_TYPE: dict[WorkItemType, WorkItemType | None] = {

    WorkItemType.EPIC: None,

    WorkItemType.FEATURE: WorkItemType.EPIC,

    WorkItemType.USER_STORY: WorkItemType.FEATURE,

    WorkItemType.TASK: WorkItemType.USER_STORY,

    WorkItemType.SUBTASK: WorkItemType.TASK,

}



_CHILD_TYPE: dict[WorkItemType, WorkItemType | None] = {

    WorkItemType.EPIC: WorkItemType.FEATURE,

    WorkItemType.FEATURE: WorkItemType.USER_STORY,

    WorkItemType.USER_STORY: WorkItemType.TASK,

    WorkItemType.TASK: WorkItemType.SUBTASK,

    WorkItemType.SUBTASK: None,

}





class SourceReference(BaseModel):

    """Provenance pointer back to a source contract field, without rewriting it."""



    model_config = ConfigDict(extra="forbid", strict=True)



    contract_id: str

    json_path: str

    field_name: str | None = None

    excerpt: str | None = None





class Effort(BaseModel):

    """Optional effort copied only when the source provided it explicitly."""



    model_config = ConfigDict(extra="forbid", strict=True)



    story_points: int | float | None = None

    original_estimate_hours: int | float | None = None

    remaining_hours: int | float | None = None

    textual_estimate: str | None = None



    @model_validator(mode="after")

    def at_least_one_explicit_value(self) -> Effort:

        if (

            self.story_points is None

            and self.original_estimate_hours is None

            and self.remaining_hours is None

            and self.textual_estimate is None

        ):

            raise ValueError(

                "effort requires at least one explicit source-provided value; "

                "do not invent estimates"

            )

        return self





class CanonicalWorkItemBase(BaseModel):

    model_config = ConfigDict(extra="forbid", strict=True)



    canonical_id: str = Field(default_factory=new_canonical_id)

    title: str

    description: str = ""

    status: WorkItemStatus = WorkItemStatus.TODO

    approval_state: ApprovalState = ApprovalState.NOT_APPROVED

    priority: Priority | None = None

    parent_id: str | None = None

    child_ids: list[str] = Field(default_factory=list)

    acceptance_criteria: list[str] = Field(default_factory=list)

    test_requirements: list[str] = Field(default_factory=list)

    dependencies: list[str] = Field(default_factory=list)

    provenance: list[SourceReference] = Field(default_factory=list)

    effort: Effort | None = None



    @field_validator("canonical_id")

    @classmethod

    def canonical_id_must_be_stable_and_non_tracker(cls, value: str) -> str:

        if not value or not value.strip():

            raise ValueError("canonical_id must be a non-empty string")

        if value.startswith(("cbl_",)):

            return value

        raise ValueError(

            "canonical_id must be a generated canonical identifier "

            f"(prefix 'cbl_'), got {value!r}"

        )



    @field_validator("title")

    @classmethod

    def title_must_be_non_empty(cls, value: str) -> str:

        if value.strip() == "":

            raise ValueError("title must be a non-empty string")

        return value



    @field_validator("child_ids", "dependencies")

    @classmethod

    def ids_must_be_unique(cls, value: list[str]) -> list[str]:

        if len(value) != len(set(value)):

            raise ValueError("IDs must be unique")

        return value





class Epic(CanonicalWorkItemBase):

    type: Literal[WorkItemType.EPIC] = WorkItemType.EPIC



    @model_validator(mode="after")

    def epic_must_be_root(self) -> Epic:

        if self.parent_id is not None:

            raise ValueError("epics cannot have a parent")

        return self





class Feature(CanonicalWorkItemBase):

    type: Literal[WorkItemType.FEATURE] = WorkItemType.FEATURE





class UserStory(CanonicalWorkItemBase):

    type: Literal[WorkItemType.USER_STORY] = WorkItemType.USER_STORY





class Task(CanonicalWorkItemBase):

    type: Literal[WorkItemType.TASK] = WorkItemType.TASK





class Subtask(CanonicalWorkItemBase):

    type: Literal[WorkItemType.SUBTASK] = WorkItemType.SUBTASK



    @model_validator(mode="after")

    def subtask_cannot_have_children(self) -> Subtask:

        if self.child_ids:

            raise ValueError("subtasks cannot have children")

        return self





CanonicalWorkItem = Annotated[

    Union[Epic, Feature, UserStory, Task, Subtask],

    Field(discriminator="type"),

]





class CanonicalBacklogV1(BaseModel):

    """Tracker-independent backlog contract `canonical.backlog.v1`."""



    model_config = ConfigDict(extra="forbid", strict=True)



    contract_id: str = Field(default=CANONICAL_BACKLOG_V1, frozen=True)

    backlog_id: str = Field(default_factory=new_canonical_id)

    items: list[CanonicalWorkItem] = Field(default_factory=list)

    source_contract_id: str | None = None

    source_provenance: list[SourceReference] = Field(default_factory=list)



    @field_validator("contract_id")

    @classmethod

    def contract_id_must_match(cls, value: str) -> str:

        if value != CANONICAL_BACKLOG_V1:

            raise ValueError(f"contract_id must be {CANONICAL_BACKLOG_V1!r}, got {value!r}")

        return value



    @field_validator("backlog_id")

    @classmethod

    def backlog_id_must_be_canonical(cls, value: str) -> str:

        if not value.startswith("cbl_"):

            raise ValueError(

                "backlog_id must be a generated canonical identifier "

                f"(prefix 'cbl_'), got {value!r}"

            )

        return value



    @model_validator(mode="after")

    def validate_hierarchy_and_references(self) -> CanonicalBacklogV1:

        items_by_id: dict[str, Any] = {}

        for item in self.items:

            if item.canonical_id in items_by_id:

                raise ValueError(f"duplicate canonical_id {item.canonical_id!r}")

            items_by_id[item.canonical_id] = item



        for item in self.items:

            _validate_parent(item, items_by_id)

        for item in self.items:

            _validate_children(item, items_by_id)

            for dependency_id in item.dependencies:

                if dependency_id not in items_by_id:

                    raise ValueError(

                        f"{item.canonical_id} dependency {dependency_id!r} does not exist"

                    )

                if dependency_id == item.canonical_id:

                    raise ValueError(f"{item.canonical_id} cannot depend on itself")



        _reject_cycles(items_by_id)

        _reject_dependency_cycles(items_by_id)

        return self



    def item_map(self) -> dict[str, Epic | Feature | UserStory | Task | Subtask]:

        return {item.canonical_id: item for item in self.items}





def _validate_parent(

    item: Epic | Feature | UserStory | Task | Subtask,

    items_by_id: dict[str, Any],

) -> None:

    expected_parent_type = _PARENT_TYPE[item.type]

    if expected_parent_type is None:

        return

    if item.parent_id is None:

        return

    parent = items_by_id.get(item.parent_id)

    if parent is None:

        raise ValueError(

            f"{item.canonical_id} parent_id {item.parent_id!r} does not exist in the backlog"

        )

    if parent.type != expected_parent_type:

        raise ValueError(

            f"{item.type.value} {item.canonical_id} parent must be a {expected_parent_type.value}, "

            f"got {parent.type.value}"

        )

    if item.canonical_id not in parent.child_ids:

        raise ValueError(

            f"{item.canonical_id} is missing from parent {parent.canonical_id} child_ids"

        )





def _validate_children(

    item: Epic | Feature | UserStory | Task | Subtask,

    items_by_id: dict[str, Any],

) -> None:

    expected_child_type = _CHILD_TYPE[item.type]

    for child_id in item.child_ids:

        child = items_by_id.get(child_id)

        if child is None:

            raise ValueError(

                f"{item.canonical_id} child_id {child_id!r} does not exist in the backlog"

            )

        if expected_child_type is None:

            raise ValueError(f"{item.type.value} {item.canonical_id} cannot have children")

        if child.type != expected_child_type:

            raise ValueError(

                f"{item.type.value} {item.canonical_id} children must be {expected_child_type.value}, "

                f"got {child.type.value}"

            )

        if child.parent_id != item.canonical_id:

            raise ValueError(

                f"{child.canonical_id} parent_id does not point back to {item.canonical_id}"

            )





def _reject_cycles(items_by_id: dict[str, Any]) -> None:

    visiting: set[str] = set()

    visited: set[str] = set()



    def walk(item_id: str) -> None:

        if item_id in visited:

            return

        if item_id in visiting:

            raise ValueError("canonical backlog contains a parent-child cycle")

        visiting.add(item_id)

        item = items_by_id[item_id]

        for child_id in item.child_ids:

            walk(child_id)

        visiting.remove(item_id)

        visited.add(item_id)



    for item_id in items_by_id:

        walk(item_id)



def _reject_dependency_cycles(items_by_id: dict[str, Any]) -> None:

    visiting: set[str] = set()

    visited: set[str] = set()



    def walk(item_id: str) -> None:

        if item_id in visited:

            return

        if item_id in visiting:

            raise ValueError("canonical backlog contains a dependency cycle")

        visiting.add(item_id)

        item = items_by_id[item_id]

        for dependency_id in item.dependencies:

            if dependency_id in items_by_id:

                walk(dependency_id)

        visiting.remove(item_id)

        visited.add(item_id)



    for item_id in items_by_id:

        walk(item_id)

