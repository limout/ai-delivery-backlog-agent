"""Structured generation findings. Smallest addition compatible with canonical models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from backlog_agent.backlog.models import SourceReference
from backlog_agent.contracts.versions import GENERATION_FINDINGS_V1


class FindingSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FindingCode(str, Enum):
    UNCOVERED_FUNCTIONAL_REQUIREMENT = "UNCOVERED_FUNCTIONAL_REQUIREMENT"
    COPILOT_CONTRADICTION = "COPILOT_CONTRADICTION"
    SCOPE_CONFLICT = "SCOPE_CONFLICT"
    AMBIGUOUS_DECOMPOSITION = "AMBIGUOUS_DECOMPOSITION"
    OUT_OF_SCOPE_EXCLUDED = "OUT_OF_SCOPE_EXCLUDED"
    ID_COLLISION = "ID_COLLISION"
    LLM_VALIDATION_FAILED = "LLM_VALIDATION_FAILED"
    UNGROUNDED_PROVENANCE = "UNGROUNDED_PROVENANCE"
    GENERATED_CONTENT = "GENERATED_CONTENT"
    UNRESOLVED_PREREQUISITE = "UNRESOLVED_PREREQUISITE"
    UNCOVERED_NON_FUNCTIONAL_REQUIREMENT = "UNCOVERED_NON_FUNCTIONAL_REQUIREMENT"
    PLANNING_CONSTRAINT = "PLANNING_CONSTRAINT"
    ATOMIC_STORY = "ATOMIC_STORY"


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    severity: FindingSeverity
    code: FindingCode
    message: str
    source_references: list[SourceReference] = Field(default_factory=list)
    canonical_ids: list[str] = Field(default_factory=list)


class GenerationFindings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    contract_id: str = Field(default=GENERATION_FINDINGS_V1, frozen=True)
    items: list[Finding] = Field(default_factory=list)
