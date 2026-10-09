from __future__ import annotations

from types import SimpleNamespace

from backlog_agent.backlog.granularity import (
    looks_broad_task,
    looks_like_implementation_work,
    looks_over_decomposed,
    looks_oversized_story,
    task_restates_parent,
)


def _story(**overrides: object) -> SimpleNamespace:
    values = {
        "title": "Show the office holiday calendar",
        "description": "A resident opens the calendar and sees closed dates.",
        "acceptance_criteria": ["Closed dates are listed."],
        "test_requirements": ["Render the calendar fixture."],
        "provenance": (),
        "child_ids": (),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _task(title: str, description: str = "") -> SimpleNamespace:
    return SimpleNamespace(title=title, description=description)


def test_atomic_story_is_not_oversized() -> None:
    story = _story()
    assert looks_oversized_story(story) is False
    assert looks_like_implementation_work(story) is False


def test_story_with_independent_outcomes_is_oversized() -> None:
    story = _story(
        title="Apply, upload, pay, and track a permit",
        description=(
            "A resident applies for a permit, uploads evidence, pays the fee, "
            "and tracks status until a clerk publishes a decision."
        ),
        acceptance_criteria=[
            "Application is stored.",
            "Evidence is uploaded.",
            "Fee is paid.",
            "Status is visible.",
        ],
        provenance=(
            SimpleNamespace(json_path="$.requirements.functional_requirements[0]"),
            SimpleNamespace(json_path="$.requirements.functional_requirements[1]"),
        ),
    )
    assert looks_oversized_story(story) is True


def test_one_functional_and_one_nfr_path_is_not_oversized() -> None:
    story = _story(
        provenance=(
            SimpleNamespace(json_path="$.requirements.functional_requirements[2]"),
            SimpleNamespace(json_path="$.requirements.non_functional_requirements[1]"),
        ),
    )
    assert looks_oversized_story(story) is False


def test_two_functional_requirement_paths_are_oversized() -> None:
    story = _story(
        title="Show closed dates",
        description="A resident opens the calendar and sees closed dates.",
        acceptance_criteria=["Closed dates are listed."],
        provenance=(
            SimpleNamespace(json_path="$.requirements.functional_requirements[0]"),
            SimpleNamespace(json_path="$.requirements.functional_requirements[1]"),
        ),
    )
    assert looks_oversized_story(story) is True


def test_integration_story_needs_implementation_work() -> None:
    story = _story(
        title="Attach a decision letter",
        description="Store the letter and retry the document webhook on timeout.",
        acceptance_criteria=["Letter is attached."],
    )
    assert looks_like_implementation_work(story) is True


def test_broad_and_restated_tasks() -> None:
    parent = _story(title="Resident permit desk")
    assert looks_broad_task(_task("Implement the feature"), parent) is True
    assert looks_broad_task(_task("Build the entire journey")) is True
    assert task_restates_parent(_task("Implement the resident permit desk"), parent) is True
    assert task_restates_parent(_task("Persist the decision letter PDF"), parent) is False
    assert looks_broad_task(_task("Persist the decision letter PDF"), parent) is False


def test_numbered_duplicate_tasks_are_over_decomposed() -> None:
    assert looks_over_decomposed([f"Save draft {index}" for index in range(1, 5)]) is True
    assert looks_over_decomposed(
        [
            "Persist the decision letter PDF",
            "Virus-scan uploaded evidence",
            "Encrypt evidence at rest",
        ]
    ) is False
