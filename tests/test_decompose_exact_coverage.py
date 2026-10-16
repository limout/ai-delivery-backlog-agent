"""Representative coverage fixture, not a captured production envelope.

The production 422 listed uncovered functional requirements and truncated each
message at 180 characters. These sentences keep the observed shapes: a short
requirement stated verbatim on a story, a compound stated only in part, and
requirements no story states.
"""

from __future__ import annotations

from backlog_agent.backlog.decompose import _try_materialize
from backlog_agent.findings.models import FindingCode
from backlog_agent.source import load_project_source

ACCOUNT = "Customers can view their own account."
COMPOUND = (
    "The system must notify customers when a payment fails and provide a way to retry payment."
)
NOTIFY_ONLY = "Notify customers when a payment fails."
MIGRATION = "The system must migrate active membership records from an exported file."
PLANNING = (
    "Use a single migration phase for the initial release as the planning assumption, "
    "but flag scope and data quality before import."
)
ABSENT = "Staff can export a monthly attendance ledger."


def _envelope() -> dict:
    return {
        "workflow_status": "COMPLETE",
        "status": "COMPLETE",
        "user_request": "Build a member portal for accounts, payment notices, and record migration.",
        "executive_summary": "Members view accounts and receive payment notices.",
        "requirements": {
            "functional_requirements": [ACCOUNT, COMPOUND, MIGRATION, PLANNING, ABSENT],
            "acceptance_criteria": [
                "Given a signed-in customer, when they open the portal, then their account is shown."
            ],
            "non_functional_requirements": ["Portal pages must respond within two seconds."],
            "open_questions": [],
        },
        "solution": {"key_capabilities": ["Account access", "Payment notices"]},
        "delivery_plan": {
            "delivery_phases": ["Build"],
            "workstreams": ["Portal delivery"],
            "milestones": ["MVP"],
            "dependencies": [],
        },
        "sow": {
            "in_scope": [ACCOUNT, "Payment failure notices"],
            "out_of_scope": ["In-person branch services"],
            "deliverables": ["Member portal"],
        },
    }


def _src(path: str, field: str, excerpt: str) -> dict[str, str]:
    return {"json_path": path, "field_name": field, "excerpt": excerpt}


def _item(
    local_id: str,
    type_name: str,
    title: str,
    parent: str | None,
    provenance: list[dict[str, str]],
    *,
    description: str = "",
    title_origin: str = "generated",
) -> dict:
    return {
        "local_id": local_id,
        "type": type_name,
        "title": title,
        "description": description,
        "parent_local_id": parent,
        "depends_on_local_ids": [],
        "acceptance_criteria": [],
        "test_requirements": [],
        "provenance": provenance,
        "title_origin": title_origin,
        "description_origin": "generated",
        "uncertainties": [],
    }


def _proposal(*, include_account: bool = True, cite_full_compound: bool = False) -> dict:
    account_title = ACCOUNT
    account_description = (
        "A signed-in customer opens the portal and sees only the account that belongs to them."
    )
    items = [
        _item(
            "epic-1",
            "epic",
            "Member portal",
            None,
            [_src("$.user_request", "user_request", _envelope()["user_request"])],
        ),
        _item(
            "feat-account",
            "feature",
            "Account access",
            "epic-1",
            [_src("$.solution.key_capabilities[0]", "key_capabilities", "Account access")],
        ),
        _item(
            "feat-notice",
            "feature",
            "Payment notices",
            "epic-1",
            [_src("$.solution.key_capabilities[1]", "key_capabilities", "Payment notices")],
        ),
    ]
    if include_account:
        items.append(
            _item(
                "story-account",
                "user_story",
                account_title,
                "feat-account",
                [_src("$.sow.in_scope[0]", "in_scope", ACCOUNT)],
                description=account_description,
                title_origin="source",
            )
        )
    compound_title = COMPOUND if cite_full_compound else NOTIFY_ONLY
    compound_provenance = (
        [_src("$.requirements.functional_requirements[1]", "functional_requirements", COMPOUND)]
        if cite_full_compound
        else [_src("$.sow.in_scope[1]", "in_scope", "Payment failure notices")]
    )
    items.append(
        _item(
            "story-notice",
            "user_story",
            compound_title,
            "feat-notice",
            compound_provenance,
            description=(
                "After a payment fails, the customer receives a notice that names the failed payment."
            ),
            title_origin="source" if cite_full_compound else "generated",
        )
    )
    return {"items": items, "prerequisites": [], "uncertainties": []}


def _uncovered(result) -> list[str]:
    return [
        reference.excerpt or ""
        for item in result.findings.items
        if item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT
        for reference in item.source_references
    ]


def test_exact_story_text_covers_its_requirement_without_covering_a_partial_compound():
    source = load_project_source(_envelope())
    result = _try_materialize(source, _proposal())
    uncovered = _uncovered(result)
    assert result.backlog is not None
    assert not any(ACCOUNT in text for text in uncovered)
    assert any(COMPOUND in text for text in uncovered)
    assert any(MIGRATION in text for text in uncovered)
    assert any(PLANNING in text for text in uncovered)
    assert any(ABSENT in text for text in uncovered)


def test_full_compound_text_is_coverage_and_a_missing_requirement_is_not():
    source = load_project_source(_envelope())
    covered = _try_materialize(source, _proposal(cite_full_compound=True))
    assert not any(COMPOUND in text for text in _uncovered(covered))
    assert any(ABSENT in text for text in _uncovered(covered))

    missing_account = _try_materialize(source, _proposal(include_account=False))
    assert missing_account.backlog is not None
    assert any(ACCOUNT in text for text in _uncovered(missing_account))
    assert any(
        item.code is FindingCode.UNCOVERED_FUNCTIONAL_REQUIREMENT
        for item in missing_account.findings.items
    )
