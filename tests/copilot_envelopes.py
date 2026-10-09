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
