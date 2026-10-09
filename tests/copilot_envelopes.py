"""Minimal COMPLETE Copilot-shaped envelopes for generic decompose tests."""

from __future__ import annotations

from typing import Any


def _complete(**sections: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "workflow_status": "COMPLETE",
        "status": "COMPLETE",
        "user_request": sections.pop("user_request"),
        "executive_summary": sections.pop("executive_summary", ""),
        "requirements": sections.pop("requirements"),
        "solution": sections.pop("solution"),
        "delivery_plan": sections.pop(
            "delivery_plan",
            {"delivery_phases": ["Build"], "workstreams": ["Delivery"], "milestones": ["MVP"]},
        ),
        "sow": sections.pop("sow"),
    }
    payload.update(sections)
    return payload


def library_holds_envelope() -> dict[str, Any]:
    return _complete(
        user_request="Build a public library catalog so members can search holdings and place holds.",
        executive_summary="Replace the card catalog with an authenticated member search and hold service.",
        requirements={
            "functional_requirements": [
                "The system shall let a member search the catalog by title or author.",
                "The system shall let a member place a hold on an available copy.",
                "The system shall notify the member when a held copy is ready for pickup.",
            ],
            "acceptance_criteria": [
                "Given a matching title, when the member searches, then the matching holdings are listed."
            ],
            "non_functional_requirements": [
                "Catalog search p95 latency must stay under 400 milliseconds.",
                "Launch the member portal within 6 weeks pending feasibility assessment.",
            ],
            "open_questions": ["Identity provider for member login is not specified."],
        },
        solution={
            "key_capabilities": [
                "Catalog search",
                "Hold placement",
                "Pickup notification",
            ]
        },
        delivery_plan={
            "delivery_phases": ["Build"],
            "workstreams": ["Member experience"],
            "milestones": ["Holds MVP"],
            "dependencies": ["Access to the bibliographic vendor MARC feed is not yet granted."],
        },
        sow={
            "in_scope": ["Catalog search", "Member holds"],
            "out_of_scope": ["Interlibrary loan routing"],
            "deliverables": ["Search API", "Hold confirmation email"],
        },
    )


def municipal_permits_envelope() -> dict[str, Any]:
    return _complete(
        user_request="Build a municipal permitting desk so residents can apply, pay, and track decisions.",
        executive_summary="Replace paper permit applications with a resident self-service desk.",
        requirements={
            "functional_requirements": [
                "The system shall let a resident apply for a permit, upload evidence, pay the fee, and track status until a decision is issued.",
                "The system shall publish the office holiday calendar to residents.",
                "The system shall let a clerk attach a decision letter to an application.",
            ],
            "acceptance_criteria": [
                "Given a complete application, when the resident pays the fee, then the application is queued for review."
            ],
            "non_functional_requirements": [
                "Uploaded evidence must be encrypted at rest.",
                "Go-live is desired within 10 weeks pending feasibility assessment.",
            ],
            "open_questions": [
                "The evidence file format and maximum size are unspecified.",
            ],
        },
        solution={
            "key_capabilities": [
                "Resident permit desk",
                "Clerk decision recording",
            ]
        },
        sow={
            "in_scope": ["Resident applications", "Clerk decisions"],
            "out_of_scope": ["Building inspections in the field"],
            "deliverables": ["Application intake form", "Decision letter attachment"],
            "dependencies": ["Payment processor sandbox access is not granted."],
        },
    )


def pickup_notice_envelope() -> dict[str, Any]:
    return _complete(
        user_request="Build a catalog so members can place holds and receive pickup notices.",
        executive_summary="Members place holds and receive pickup notices when copies are ready.",
        requirements={
            "functional_requirements": [
                "The system shall let a member place a hold on an available copy.",
                "The system shall send a pickup notice when a held copy is ready.",
            ],
            "acceptance_criteria": [
                "A member can place a hold on an available copy and trigger a pickup notice "
                "via the notice service."
            ],
            "non_functional_requirements": [
                "Catalog search p95 latency must stay under 400 milliseconds.",
            ],
        },
        solution={
            "key_capabilities": [
                "Place holds on available copies.",
                "Send pickup notices when held copies are ready.",
            ]
        },
        sow={
            "in_scope": ["Member holds", "Pickup notices"],
            "out_of_scope": ["Interlibrary loan routing"],
            "deliverables": ["Hold placement", "Pickup notice delivery"],
        },
    )


def warehouse_picking_envelope() -> dict[str, Any]:
    return _complete(
        user_request="Build a warehouse picking app so operators can receive pick lists and confirm bins.",
        executive_summary="Handheld picking replaces paper pick tickets on the warehouse floor.",
        requirements={
            "functional_requirements": [
                "The system shall display the next pick location and SKU to the operator.",
                "The system shall let the operator confirm a pick by scanning the bin.",
                "The system shall let a supervisor reprint a completed pick list.",
            ],
            "acceptance_criteria": [
                "Given an assigned pick, when the operator scans the correct bin, then the pick is marked complete."
            ],
            "non_functional_requirements": [
                "Handheld sessions must remain usable on the warehouse Wi-Fi with 2% packet loss."
            ],
        },
        solution={
            "key_capabilities": [
                "Pick list playback",
                "Bin scan confirmation",
            ]
        },
        sow={
            "in_scope": ["Handheld picking"],
            "out_of_scope": ["Packing optimization"],
            "deliverables": ["Operator handheld workflow"],
            "dependencies": ["Barcode symbology for existing bins is unconfirmed."],
        },
    )
