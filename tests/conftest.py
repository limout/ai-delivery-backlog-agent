from __future__ import annotations

from typing import Any

import pytest


def minimal_complete_envelope(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "workflow_status": "COMPLETE",
        "status": "COMPLETE",
        "user_request": "Build a customer portal for policy renewals.",
        "requirements": {
            "functional_requirements": [
                "The system shall allow a customer to renew an existing policy."
            ],
            "acceptance_criteria": [
                "Given a valid policy, when the customer submits a renewal, then a confirmation is stored."
            ],
        },
        "solution": {
            "key_capabilities": ["Policy renewal workflow"],
        },
        "delivery_plan": {
            "delivery_phases": ["Discovery", "Build", "Launch"],
            "workstreams": ["Portal", "Integrations"],
            "milestones": ["MVP ready"],
        },
        "sow": {
            "in_scope": ["Customer self-service renewal"],
            "out_of_scope": ["Claims adjudication"],
            "deliverables": ["Working portal MVP"],
        },
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def complete_envelope() -> dict[str, Any]:
    return minimal_complete_envelope()
