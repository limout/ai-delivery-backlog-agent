"""Versioned input contract for Copilot `/analyze` JSON envelopes."""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backlog_agent.contracts.versions import COPILOT_WORKFLOW_RESPONSE_V1

_COMPLETE = "COMPLETE"
_COPILOT_SECTION_KEYS = frozenset(
    {"user_request", "requirements", "solution", "delivery_plan", "sow"}
)


def _is_blank(value: str) -> bool:
    return value.strip() == ""


def _reject_blank_string(value: str, *, field_name: str) -> str:
    if _is_blank(value):
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _looks_like_jira_payload(data: Mapping[str, Any]) -> bool:
    if "issues" in data and "user_request" not in data:
        return True
    if data.get("expand") is not None and "fields" in data and "key" in data:
        return True
    return False


def _looks_like_azure_devops_payload(data: Mapping[str, Any]) -> bool:
    if "workItems" in data and "user_request" not in data:
        return True
    fields = data.get("fields")
    if isinstance(fields, dict) and any(
        key.startswith("System.") or key.startswith("Microsoft.") for key in fields
    ):
        return True
    return False


def _looks_like_tracker_payload(data: Mapping[str, Any]) -> bool:
    return _looks_like_jira_payload(data) or _looks_like_azure_devops_payload(data)


def _parse_raw_payload(data: Any) -> dict[str, Any]:
    if isinstance(data, (bytes, bytearray)):
        try:
            data = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "Rejected non-JSON payload; expected "
                f"{COPILOT_WORKFLOW_RESPONSE_V1} JSON envelope from Copilot /analyze"
            ) from exc

    if isinstance(data, str):
        stripped = data.strip()
        if not stripped:
            raise ValueError(
                "Rejected plain-text payload; expected "
                f"{COPILOT_WORKFLOW_RESPONSE_V1} JSON envelope from Copilot /analyze"
            )
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Rejected plain-text payload; expected "
                f"{COPILOT_WORKFLOW_RESPONSE_V1} JSON envelope from Copilot /analyze, "
                "not a Copilot plain-text export"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                "Rejected non-object JSON payload; expected "
                f"{COPILOT_WORKFLOW_RESPONSE_V1} JSON envelope from Copilot /analyze"
            )
        return parsed

    if not isinstance(data, dict):
        raise ValueError(
            "Rejected unsupported payload type; expected "
            f"{COPILOT_WORKFLOW_RESPONSE_V1} JSON object envelope from Copilot /analyze"
        )
    return data


class RequirementsSection(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    functional_requirements: list[str] = Field(min_length=1)
    acceptance_criteria: list[str]

    @field_validator("functional_requirements")
    @classmethod
    def functional_requirements_must_be_non_empty_strings(cls, value: list[str]) -> list[str]:
        for index, item in enumerate(value):
            if _is_blank(item):
                raise ValueError(
                    "requirements.functional_requirements must be a non-empty list of "
                    f"non-empty strings (blank entry at index {index})"
                )
        return value

    @field_validator("acceptance_criteria")
    @classmethod
    def acceptance_criteria_must_be_strings(cls, value: list[str]) -> list[str]:
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(
                    "requirements.acceptance_criteria must be a list of strings "
                    f"(invalid entry at index {index})"
                )
        return value


class SolutionSection(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    key_capabilities: list[str]


class DeliveryPlanSection(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    delivery_phases: list[str]
    workstreams: list[str]
    milestones: list[str]


class StatementOfWorkSection(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    in_scope: list[str]
    out_of_scope: list[str]
    deliverables: list[str]


class CopilotAnalyzeEnvelope(BaseModel):
    """Validated view of a Copilot `/analyze` JSON envelope."""

    model_config = ConfigDict(extra="allow", strict=True)

    workflow_status: str
    status: str
    user_request: str
    requirements: RequirementsSection
    solution: SolutionSection
    delivery_plan: DeliveryPlanSection
    sow: StatementOfWorkSection

    @field_validator("user_request")
    @classmethod
    def user_request_must_be_non_empty(cls, value: str) -> str:
        return _reject_blank_string(value, field_name="user_request")

    @model_validator(mode="after")
    def statuses_must_be_complete_and_aligned(self) -> CopilotAnalyzeEnvelope:
        if self.workflow_status != self.status:
            raise ValueError(
                "Mismatched status fields: "
                f"workflow_status={self.workflow_status!r}, status={self.status!r}; "
                "both must equal 'COMPLETE'"
            )
        if self.workflow_status != _COMPLETE or self.status != _COMPLETE:
            raise ValueError(
                "Copilot workflow is not COMPLETE "
                f"(workflow_status={self.workflow_status!r}, status={self.status!r}); "
                "NEEDS_INFO, BLOCKED, RUNNING, and incomplete envelopes are rejected"
            )
        return self


class CopilotWorkflowResponseV1(BaseModel):
    """Input contract `copilot.workflow_response.v1`.

    Validates the complete Copilot `/analyze` JSON envelope and preserves the
    original payload for traceability without rewriting or inferring fields.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    contract_id: str = Field(default=COPILOT_WORKFLOW_RESPONSE_V1, frozen=True)
    envelope: CopilotAnalyzeEnvelope
    original_envelope: dict[str, Any]

    @model_validator(mode="before")
    @classmethod
    def wrap_raw_copilot_envelope(cls, data: Any) -> Any:
        if isinstance(data, CopilotWorkflowResponseV1):
            return data
        if (
            isinstance(data, dict)
            and "envelope" in data
            and "original_envelope" in data
            and set(data.keys()) <= {"contract_id", "envelope", "original_envelope"}
        ):
            wrapped = dict(data)
            wrapped.setdefault("contract_id", COPILOT_WORKFLOW_RESPONSE_V1)
            return wrapped

        raw = _parse_raw_payload(data)
        if _looks_like_tracker_payload(raw) and not _COPILOT_SECTION_KEYS <= raw.keys():
            tracker = "Jira" if _looks_like_jira_payload(raw) else "Azure DevOps"
            raise ValueError(
                f"Rejected unrelated {tracker} tracker payload; expected "
                f"{COPILOT_WORKFLOW_RESPONSE_V1} JSON envelope from Copilot /analyze"
            )
        return {
            "contract_id": COPILOT_WORKFLOW_RESPONSE_V1,
            "envelope": raw,
            "original_envelope": copy.deepcopy(raw),
        }

    @field_validator("contract_id")
    @classmethod
    def contract_id_must_match(cls, value: str) -> str:
        if value != COPILOT_WORKFLOW_RESPONSE_V1:
            raise ValueError(
                f"contract_id must be {COPILOT_WORKFLOW_RESPONSE_V1!r}, got {value!r}"
            )
        return value

    @model_validator(mode="after")
    def original_envelope_must_match_source(self) -> CopilotWorkflowResponseV1:
        if not isinstance(self.original_envelope, dict):
            raise ValueError("original_envelope must be the original JSON object")
        return self
