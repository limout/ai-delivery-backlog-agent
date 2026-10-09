"""Scripted LLM proposals for the COMPLETE Copilot fixture. No API calls."""

from __future__ import annotations

from typing import Any

from tests.test_generate_canonical_backlog import load_fixture


def _src(path: str, field: str, excerpt: str) -> dict[str, str]:
    return {"json_path": path, "field_name": field, "excerpt": excerpt}


def representative_complete_proposal() -> dict[str, Any]:
    payload = load_fixture()
    request = payload["user_request"]
    fr = payload["requirements"]["functional_requirements"]
    ac = payload["requirements"]["acceptance_criteria"]
    caps = payload["solution"]["key_capabilities"]
    deliverables = payload["sow"]["deliverables"]
    milestone = payload["delivery_plan"]["milestones"][0]
    return {
        "items": [
            {
                "local_id": "epic-1",
                "type": "epic",
                "title": "Customer self-service policy renewal portal",
                "description": "Replace emailed renewal packs with an authenticated customer portal.",
                "parent_local_id": None,
                "depends_on_local_ids": [],
                "acceptance_criteria": [],
                "test_requirements": [],
                "provenance": [_src("$.user_request", "user_request", request)],
                "title_origin": "generated",
                "description_origin": "generated",
                "uncertainties": [],
            },
            {
                "local_id": "feat-portal",
                "type": "feature",
                "title": caps[0],
                "description": "",
                "parent_local_id": "epic-1",
                "depends_on_local_ids": [],
                "acceptance_criteria": [],
                "test_requirements": [],
                "provenance": [_src("$.solution.key_capabilities[0]", "key_capabilities", caps[0])],
                "title_origin": "source",
                "description_origin": "generated",
                "uncertainties": [],
            },
            {
                "local_id": "feat-renewal",
                "type": "feature",
                "title": caps[1],
                "description": "",
                "parent_local_id": "epic-1",
                "depends_on_local_ids": [],
                "acceptance_criteria": [],
                "test_requirements": [],
                "provenance": [_src("$.solution.key_capabilities[1]", "key_capabilities", caps[1])],
                "title_origin": "source",
                "description_origin": "generated",
                "uncertainties": [],
            },
            {
                "local_id": "story-renew",
                "type": "user_story",
                "title": fr[0],
                "description": (
                    "An authenticated customer reviews eligibility, premium, and coverage "
                    "for a motor or home policy before confirming a renewal online."
                ),
                "parent_local_id": "feat-renewal",
                "depends_on_local_ids": [],
                "acceptance_criteria": [ac[0]],
                "test_requirements": [
                    "Verify an eligible customer can complete renewal and a confirmation record is stored."
                ],
                "provenance": [
                    _src("$.requirements.functional_requirements[0]", "functional_requirements", fr[0])
                ],
                "title_origin": "source",
                "description_origin": "generated",
                "uncertainties": [
                    "Copilot acceptance criteria are not bound to a single functional requirement."
                ],
            },
            {
                "local_id": "story-confirm",
                "type": "user_story",
                "title": fr[1],
                "description": (
                    "After a renewal is accepted, the customer receives confirmation so they "
                    "do not need to call the contact centre for proof of processing."
                ),
                "parent_local_id": "feat-portal",
                "depends_on_local_ids": ["story-renew"],
                "acceptance_criteria": [],
                "test_requirements": [],
                "provenance": [
                    _src("$.requirements.functional_requirements[1]", "functional_requirements", fr[1])
                ],
                "title_origin": "source",
                "description_origin": "generated",
                "uncertainties": [],
            },
            {
                "local_id": "story-premium",
                "type": "user_story",
                "title": fr[2],
                "description": (
                    "Before the customer confirms, the portal shows the proposed premium and "
                    "coverage so they can abort if the terms are unacceptable."
                ),
                "parent_local_id": "feat-renewal",
                "depends_on_local_ids": [],
                "acceptance_criteria": [],
                "test_requirements": [],
                "provenance": [
                    _src("$.requirements.functional_requirements[2]", "functional_requirements", fr[2])
                ],
                "title_origin": "source",
                "description_origin": "generated",
                "uncertainties": [],
            },
            {
                "local_id": "task-mvp",
                "type": "task",
                "title": deliverables[0],
                "description": "",
                "parent_local_id": "story-renew",
                "depends_on_local_ids": [],
                "acceptance_criteria": [],
                "test_requirements": [],
                "provenance": [_src("$.sow.deliverables[0]", "deliverables", deliverables[0])],
                "title_origin": "source",
                "description_origin": "generated",
                "uncertainties": [],
            },
            {
                "local_id": "task-email",
                "type": "task",
                "title": deliverables[1],
                "description": "",
                "parent_local_id": "story-confirm",
                "depends_on_local_ids": ["task-mvp"],
                "acceptance_criteria": [],
                "test_requirements": [],
                "provenance": [_src("$.sow.deliverables[1]", "deliverables", deliverables[1])],
                "title_origin": "source",
                "description_origin": "generated",
                "uncertainties": [],
            },
            {
                "local_id": "sub-mvp",
                "type": "subtask",
                "title": milestone,
                "description": "",
                "parent_local_id": "task-mvp",
                "depends_on_local_ids": [],
                "acceptance_criteria": [],
                "test_requirements": [],
                "provenance": [_src("$.delivery_plan.milestones[0]", "milestones", milestone)],
                "title_origin": "source",
                "description_origin": "generated",
                "uncertainties": [],
            },
        ],
        "prerequisites": [],
        "uncertainties": [
            "Delivery workstreams are scheduling constructs, not product features."
        ],
    }


def proposal_with_out_of_scope_feature() -> dict[str, Any]:
    data = representative_complete_proposal()
    payload = load_fixture()
    claims = payload["solution"]["key_capabilities"][2]
    data["items"].insert(
        3,
        {
            "local_id": "feat-claims",
            "type": "feature",
            "title": claims,
            "description": "",
            "parent_local_id": "epic-1",
            "depends_on_local_ids": [],
            "acceptance_criteria": [],
            "test_requirements": [],
            "provenance": [_src("$.solution.key_capabilities[2]", "key_capabilities", claims)],
            "title_origin": "source",
            "description_origin": "generated",
            "uncertainties": [],
        },
    )
    return data


def execution_ready_proposal() -> dict[str, Any]:
    """Stories with useful text, multiple tasks, generated AC, and no workstream Features."""

    data = representative_complete_proposal()
    payload = load_fixture()
    ac = payload["requirements"]["acceptance_criteria"][0]
    story = next(item for item in data["items"] if item["local_id"] == "story-renew")
    story["acceptance_criteria"] = [
        ac,
        "Ineligible policies show a next-step explanation instead of a confirm action.",
    ]
    story["acceptance_criteria_origins"] = ["source", "generated"]
    story["test_requirements"] = [
        "Automated test: eligible policy shows confirm; ineligible policy hides confirm and shows the explanation."
    ]
    story["test_requirements_origins"] = ["generated"]
    data["items"].insert(
        -1,
        {
            "local_id": "task-eligibility-ui",
            "type": "task",
            "title": "Build eligibility gate on the renewal review screen",
            "description": "Show confirm only when the policy administration response marks the policy eligible.",
            "parent_local_id": "story-renew",
            "depends_on_local_ids": [],
            "acceptance_criteria": [],
            "test_requirements": [],
            "provenance": [_src("$.sow.in_scope[0]", "in_scope", payload["sow"]["in_scope"][0])],
            "title_origin": "generated",
            "description_origin": "generated",
            "uncertainties": [],
        },
    )
    data["prerequisites"] = [
        {
            "text": (
                "Policy administration API documentation and sandbox availability are not confirmed; "
                "integration tasks remain blocked until access is verified."
            ),
            "status": "unresolved",
            "provenance": [_src("$.user_request", "user_request", payload["user_request"])],
        }
    ]
    return data

