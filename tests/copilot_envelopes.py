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


def customer_portal_nine_outcomes_envelope() -> dict[str, Any]:
    """Nine independently testable outcomes spanning distinct capability areas."""

    return _complete(
        user_request=(
            "Build a customer self-service portal. API access and technical documentation "
            "for the records system and the payments processor are available. Release the "
            "system incrementally rather than in one large final delivery."
        ),
        executive_summary="Customers manage accounts, orders, service requests, and payments.",
        requirements={
            "functional_requirements": [
                "The system shall authenticate customers using the existing identity provider.",
                "The system shall let a customer update their profile contact details.",
                "The system shall let a customer place and view product orders.",
                "The system shall let a customer submit a service request.",
                "The system shall let a customer pay an outstanding balance.",
                "The system shall notify a customer when an order or request status changes.",
                "The system shall exchange customer and order records with the existing SAP system.",
                "The system shall process card payments through the existing Stripe account.",
                "The system shall let a customer view the status of submitted orders and service requests.",
            ],
            "acceptance_criteria": [
                "Given valid credentials, when the customer signs in, then protected account pages are available.",
                "Given a saved profile, when the customer changes a contact detail, then the updated detail is shown.",
                "Given an in-stock product, when the customer places an order, then the order is listed.",
                "Given a described issue, when the customer submits a service request, then a request identifier is shown.",
                "Given an outstanding balance, when the customer pays, then the balance is reduced.",
                "Given a status change, when the records system reports it, then the customer is notified.",
                "Given a new order, when records are exchanged with SAP, then the order is stored in SAP.",
                "Given a card payment, when Stripe confirms settlement, then the payment is recorded.",
                "Given a submitted order or service request, when the customer opens status, then the current status is shown.",
            ],
            "non_functional_requirements": [
                "The system shall be released incrementally rather than in one large final delivery.",
                "Payment data in transit must be encrypted.",
            ],
            "open_questions": [
                "The field mapping for service-request priority codes in SAP is unspecified."
            ],
        },
        solution={
            "key_capabilities": [
                "Customer authentication and profile",
                "Orders and service requests",
                "Payments and notifications",
            ]
        },
        delivery_plan={
            "delivery_phases": ["Build"],
            "workstreams": ["Customer experience"],
            "milestones": ["Portal MVP"],
            "dependencies": [
                "Access to the SAP API and technical documentation",
                "Access to the Stripe API and technical documentation",
            ],
        },
        sow={
            "in_scope": [
                "Authentication and profile",
                "Orders and service requests",
                "Payments and notifications",
            ],
            "out_of_scope": ["Call-center workforce scheduling"],
            "deliverables": ["Customer portal MVP"],
        },
    )
