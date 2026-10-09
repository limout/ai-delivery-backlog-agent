"""LLM-proposed backlog shape. Validated in application code, not trusted as canonical."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backlog_agent.backlog.models import WorkItemType


class ContentOrigin(str, Enum):
    SOURCE = "source"
    GENERATED = "generated"


class ProposedProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    json_path: str
    field_name: str | None = None
    excerpt: str | None = None

    @field_validator("json_path")
    @classmethod
    def json_path_must_be_dollar(cls, value: str) -> str:
        if not value.startswith("$"):
            raise ValueError("json_path must start with '$'")
        return value


class ProposedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_id: str
    type: WorkItemType
    title: str
    description: str = ""
    parent_local_id: str | None = None
    depends_on_local_ids: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    test_requirements: list[str] = Field(default_factory=list)
    provenance: list[ProposedProvenance] = Field(min_length=1)
    title_origin: ContentOrigin
    description_origin: ContentOrigin = ContentOrigin.GENERATED
    uncertainties: list[str] = Field(default_factory=list)

    @field_validator("local_id")
    @classmethod
    def local_id_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("local_id must be a non-empty string")
        return value

    @field_validator("title")
    @classmethod
    def title_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title must be a non-empty string")
        return value

    @field_validator("depends_on_local_ids")
    @classmethod
    def dependency_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("depends_on_local_ids must be unique")
        return value


class BacklogProposal(BaseModel):
    """Structured LLM output. Not a ``canonical.backlog.v1`` document."""

    model_config = ConfigDict(extra="forbid")

    items: list[ProposedItem] = Field(min_length=1)
    uncertainties: list[str] = Field(default_factory=list)

    @field_validator("items")
    @classmethod
    def local_ids_must_be_unique(cls, value: list[ProposedItem]) -> list[ProposedItem]:
        seen: set[str] = set()
        for item in value:
            if item.local_id in seen:
                raise ValueError(f"duplicate local_id {item.local_id!r}")
            seen.add(item.local_id)
        return value


def proposal_json_schema() -> dict[str, Any]:
    """JSON Schema passed to providers. Pydantic still validates the parsed dict."""

    return BacklogProposal.model_json_schema()
